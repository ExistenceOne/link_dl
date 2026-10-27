"""Run one original main entry point with CUDA enforcement and resource telemetry."""
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from _01_code._99_common_utils.training_runtime import code_fingerprint
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; refusing CPU training')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    torch.set_num_interop_threads(1)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    os.environ['TRAIN_REQUIRE_CUDA'] = '1'
    script, *arguments = sys.argv[1:]
    script = str(Path(script).resolve())
    sys.argv = [script, *arguments]
    start = time.monotonic()
    report = dict(status='failed', script=script, args=arguments,
                  cuda_device=torch.cuda.get_device_name(0), pid=os.getpid(),
                  code_fingerprint=code_fingerprint())
    print('CUDA experiment:', json.dumps(report), flush=True)
    try:
        namespace = runpy.run_path(script, run_name='__main__')
        torch.cuda.synchronize()
        report['parsed_args'] = vars(namespace['args'])
        metrics_path = os.environ.get('TRAIN_METRICS_FILE')
        if metrics_path:
            report['metrics'] = json.loads(Path(metrics_path).read_text())
        checkpoint_dir = os.environ.get('TRAIN_CHECKPOINT_DIR')
        if checkpoint_dir:
            files = sorted(Path(checkpoint_dir).glob('*_checkpoint_latest.pt'))
            if not files or not all(path.stat().st_size > 0 for path in files):
                raise RuntimeError('Training returned without a checkpoint')
            report['checkpoints'] = [str(path) for path in files]
        if torch.cuda.max_memory_allocated() == 0:
            raise RuntimeError('Training returned without CUDA allocations')
        report['status'] = 'completed'
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        report['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
        report['peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
        destination = os.environ.get('TRAIN_RESULT_FILE')
        if destination:
            path = Path(destination)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(report, indent=2) + '\n')
            temporary.replace(path)
        print('CUDA result:', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
