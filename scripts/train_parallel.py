"""Discover documented commands, profile them, and run a bounded CUDA queue."""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _01_code._99_common_utils.training_runtime import code_fingerprint


@dataclass(frozen=True)
class Job:
    id: str
    script: str
    args: tuple


def discover_jobs(root):
    jobs = []
    for directory in sorted((root / '_01_code').iterdir()):
        if not directory.is_dir() or not directory.name[1:3].isdigit():
            continue
        if not 13 <= int(directory.name[1:3]) <= 21:
            continue
        for path in sorted(directory.rglob('*.py')):
            source = path.read_text()
            if 'train' not in path.name or 'trainer' in path.name or 'parse_args(' not in source:
                continue
            commands = [line.strip()[2:] for line in source.splitlines()
                        if line.strip().startswith('# python ')]
            if not commands:
                raise ValueError(f'No documented commands: {path}')
            for index, command in enumerate(commands, 1):
                python, target, *args = shlex.split(command)
                relative = str(path.relative_to(root))
                if target != relative:
                    raise ValueError(f'Wrong documented command in {relative}: {target}')
                jobs.append(Job(f'{directory.name[1:3]}-{path.stem}-{index:02}', relative, tuple(args)))
    return jobs


def can_admit(candidate, reservations, free_mib, budget_mib, slots, headroom):
    # Conservatively count full running peaks even against current free memory:
    # this also reserves their future growth when desktop GPU usage changes.
    return (len(reservations) < slots and sum(reservations) + candidate + headroom <= free_mib
            and sum(reservations) + candidate + headroom <= budget_mib)


def can_admit_exclusive(peak_allocated_mib, reservations, free_mib):
    # User approved original batch 2048 with no other training for the three large CNNs.
    return not reservations and peak_allocated_mib + 512 <= free_mib


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def effective_args(job, phase, batch_sizes):
    args = list(job.args)
    batch_size = batch_sizes.get(job.id, batch_sizes.get(job.script))
    if batch_size is not None:
        args += ['--batch_size', str(batch_size)]
    if phase == 'smoke':
        args += ['--no-wandb', '--epochs', '1', '--validation_intervals', '1']
    return args


def completed_result(directory, script, args):
    path = directory / 'result.json'
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    if (report.get('status') != 'completed' or report.get('exit_code') != 0
            or report.get('script') != script or report.get('args') != args
            or report.get('code_fingerprint') != code_fingerprint()):
        return None
    files = report.get('checkpoints', [])
    if not files or not all(Path(name).is_file() and Path(name).stat().st_size > 0 for name in files):
        return None
    return report


def new_attempt_directory(directory):
    directory.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        attempt = directory / f'attempt-{index:03}'
        try:
            attempt.mkdir()
            return attempt
        except FileExistsError:
            index += 1


def launch(job, output, phase, batch_sizes):
    job_directory = output / phase / job.id
    directory = new_attempt_directory(job_directory)
    args = effective_args(job, phase, batch_sizes)
    command = [sys.executable, '-u', str(ROOT / 'scripts/train_job.py'), str(ROOT / job.script), *args]
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               TRAIN_NUM_WORKERS='2', TRAIN_REQUIRE_CUDA='1', MPLBACKEND='Agg',
               PYTHONUNBUFFERED='1', TRAIN_MAX_BATCHES='2' if phase == 'smoke' else '0',
               TRAIN_CHECKPOINT_DIR=str(directory / 'checkpoints'),
               TRAIN_METRICS_FILE=str(directory / 'metrics.json'),
               TRAIN_RESULT_FILE=str(directory / 'result.json'))
    log = (directory / 'train.log').open('w')
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    metadata = dict(command=command, original_args=job.args, pid=process.pid,
                    started_at=datetime.now().astimezone().isoformat(), attempt=str(directory))
    write_json(directory / 'command.json', metadata)
    write_json(job_directory / 'command.json', metadata)
    write_json(job_directory / 'result.json', dict(status='running', pid=process.pid))
    print(f'START {phase} {job.id} pid={process.pid}', flush=True)
    return process, log, directory


def finish(handle):
    process, log, directory = handle
    code = process.wait()
    log.close()
    path = directory / 'result.json'
    result = json.loads(path.read_text()) if path.exists() else dict(status='failed', error='No child report')
    result['exit_code'] = code
    if code != 0:
        result['status'] = 'failed'
    write_json(path, result)
    write_json(directory.parent / 'result.json', result)
    print(f'END {directory.parent.parent.name} {directory.parent.name} status={result["status"]} exit={code}', flush=True)
    return result


def termination_signal(signum, frame):
    raise KeyboardInterrupt(f'Supervisor received signal {signum}')


def terminate_child(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_queue(jobs, output, phase, slots, headroom, batch_sizes):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; refusing CPU training')
    pending, running = list(jobs), []
    baseline_free = torch.cuda.mem_get_info()[0] / 2**20
    results = {}
    try:
        while pending or running:
            for item in list(running):
                job, handle, reservation = item
                if handle[0].poll() is not None:
                    results[job.id] = finish(handle)
                    running.remove(item)
            for job in list(pending):
                previous = completed_result(output / phase / job.id, str(ROOT / job.script),
                                            effective_args(job, phase, batch_sizes))
                if previous:
                    results[job.id] = previous
                    pending.remove(job)
                    continue
                if phase == 'full':
                    profile = completed_result(output / 'smoke' / job.id, str(ROOT / job.script),
                                               effective_args(job, 'smoke', batch_sizes))
                    if not profile:
                        results[job.id] = dict(status='blocked', error='CUDA smoke test has not passed')
                        pending.remove(job)
                        continue
                    # Reserved CUDA allocator peak plus per-process context/library allowance.
                    reservation = profile['peak_reserved_mib'] * 1.2 + 512
                else:
                    reservation = 0
                free = torch.cuda.mem_get_info()[0] / 2**20
                exclusive = phase == 'full' and job.script in {
                    '_01_code/_16_modern_cnns/_01_vgg/a_cifar10_train_vgg.py',
                    '_01_code/_16_modern_cnns/_02_nin/a_cifar10_train_nin.py',
                    '_01_code/_16_modern_cnns/_03_google_net/a_cifar10_train_google_net.py',
                }
                if phase == 'smoke':
                    admissible = not running
                elif exclusive:
                    admissible = can_admit_exclusive(profile['peak_allocated_mib'],
                                                    [r[2] for r in running], free)
                    reservation = baseline_free
                else:
                    admissible = can_admit(reservation, [r[2] for r in running], free,
                                           baseline_free, slots, headroom)
                if admissible:
                    running.append((job, launch(job, output, phase, batch_sizes), reservation))
                    pending.remove(job)
                elif not running and phase == 'full':
                    results[job.id] = dict(status='blocked', error='Insufficient VRAM with required reserve',
                                          required_mib=reservation, available_mib=free)
                    pending.remove(job)
            states = {job.id: dict(status='pending') for job in pending}
            states.update(results)
            states.update({job.id: dict(status='running', pid=handle[0].pid) for job, handle, _ in running})
            write_json(output / f'{phase}_status.json', states)
            if pending or running:
                time.sleep(3)
    finally:
        for job, handle, reservation in running:
            terminate_child(handle[0])
            results[job.id] = finish(handle)
        if running:
            states = {job.id: dict(status='pending') for job in pending}
            states.update(results)
            write_json(output / f'{phase}_status.json', states)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['smoke', 'full', 'all', 'list'], default='all')
    parser.add_argument('--slots', type=int, default=2)
    parser.add_argument('--headroom-mib', type=int, default=1024)
    parser.add_argument('--batch-sizes', type=Path, help='Explicitly approved batch-size overrides as JSON')
    args = parser.parse_args()
    if args.slots < 1 or args.headroom_mib < 0:
        parser.error('slots must be positive and headroom must be nonnegative')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # One supervisor owns an output directory; prevent duplicate concurrent training.
    import fcntl
    lock = (output / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # After a supervisor crash, do not launch over an orphaned live child.
    for path in output.glob('*/*/command.json'):
        command = json.loads(path.read_text())
        proc_path = Path(f'/proc/{command["pid"]}/cmdline')
        try:
            actual = proc_path.read_bytes().rstrip(b'\0').split(b'\0')
        except OSError:
            continue
        if actual == [str(arg).encode() for arg in command['command']]:
            raise RuntimeError(f'Previous child is still running: pid={command["pid"]}, {path}')
    signal.signal(signal.SIGTERM, termination_signal)
    signal.signal(signal.SIGHUP, termination_signal)
    (output / 'supervisor.pid').write_text(str(os.getpid()) + '\n')
    jobs = discover_jobs(ROOT)
    batch_sizes = json.loads(args.batch_sizes.read_text()) if args.batch_sizes else {}
    write_json(output / 'manifest.json', [asdict(job) for job in jobs])
    if args.phase == 'list':
        print(json.dumps([asdict(job) for job in jobs], indent=2))
        return
    failed = False
    for phase in (['smoke', 'full'] if args.phase == 'all' else [args.phase]):
        results = run_queue(jobs, output, phase, args.slots, args.headroom_mib, batch_sizes)
        failed |= any(result['status'] != 'completed' for result in results.values())
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
