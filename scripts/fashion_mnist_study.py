"""Sequential, resumable model-selection/tuning study. Never reads test data."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '_03_homeworks/homework_3/experiments'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def change(arguments, **changes):
    arguments = list(arguments)
    flags = {'batch_size': '-b', 'learning_rate': '-r', 'weight_decay': '-w',
             'seed': '--seed', 'normalization': '-n', 'dropout_rate': '--dropout_rate'}
    for name, value in changes.items():
        if name in ('augment', 'dropout'):
            arguments = [a for a in arguments if a not in ('--' + name, '--no-' + name)]
            arguments.append('--' + name if value else '--no-' + name)
        else:
            flag = flags[name]
            if flag in arguments:
                arguments[arguments.index(flag) + 1] = str(value)
            else:
                arguments.extend([flag, str(value)])
    return arguments


def signature(arguments):
    # Explicit defaults make semantic reuse independent of flag order.
    values = {'dropout_rate': 0.5}
    index = 0
    while index < len(arguments):
        flag = arguments[index]
        if flag in ('--wandb', '--dropout', '--no-dropout', '--augment', '--no-augment'):
            if flag != '--wandb':
                values[flag.replace('--no-', '').replace('--', '')] = not flag.startswith('--no-')
            index += 1
        else:
            value = arguments[index + 1]
            try:
                value = float(value)
            except ValueError:
                pass
            values[flag.lstrip('-')] = value
            index += 2
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:12]


def ensure(arguments, stage, label):
    key = signature(arguments)
    for path in BASE.glob('*/result.json'):
        result = json.loads(path.read_text())
        if result.get('status') == 'completed' and signature(result['configuration']['args']) == key:
            return result
    folder = BASE / f'{stage}_{label}_{key}'
    if stage == 'baseline':
        folder = BASE / ('baseline_' + label)
    folder.mkdir(exist_ok=True)
    configuration = {'id': folder.name, 'stage': stage, 'args': arguments}
    config_path = folder / 'config.json'
    if not config_path.exists():
        write(config_path, configuration)
    write(BASE / 'study_status.json', {'status': 'running', 'stage': stage, 'current': folder.name,
                                      'updated': time.time()})
    status_path = folder / 'status.json'
    # Adopt a worker already launched in this chat instead of duplicating it.
    while status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get('status') != 'running':
            break
        if time.time() - status_path.stat().st_mtime > 300:
            raise RuntimeError(f'Worker stopped reporting: {folder}')
        time.sleep(5)
    result_path = folder / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    if status_path.exists() and json.loads(status_path.read_text()).get('status') == 'failed':
        raise RuntimeError(f'Failed worker; inspect {folder}/train.log')
    print(f'START {folder.name}', flush=True)
    with (folder / 'train.log').open('w') as log:
        completed = subprocess.run([sys.executable, str(ROOT / 'scripts/fashion_mnist_experiment.py'),
                                    str(folder)], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    if completed.returncode:
        raise RuntimeError(f'Worker failed ({completed.returncode}): {folder}')
    result = json.loads(result_path.read_text())
    print(f"DONE {folder.name}: checkpoint accuracy={result['checkpoint_validation_accuracy']:.2f}%", flush=True)
    make_report()
    return result


def ranking(results):
    unique = {signature(r['configuration']['args']): r for r in results}
    return sorted(unique.values(), key=lambda r: (-r['checkpoint_validation_accuracy'], r['elapsed_seconds']))


def make_report():
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/fashion_mnist_report.py')], cwd=ROOT)
    if result.returncode:
        raise RuntimeError('Report generation failed')


def study():
    models = ['cnn', 'vgg', 'nin', 'googlenet', 'resnet']
    baselines = {}
    for model in models:
        config = json.loads((BASE / ('baseline_' + model) / 'config.json').read_text())
        baselines[model] = ensure(config['args'], 'baseline', model)
    selected = [r['args']['model'] for r in ranking(list(baselines.values()))[:2]]
    write(BASE / 'selection.json', {'baseline_candidates': selected, 'baseline_results': baselines})
    make_report()
    best = {}
    for model in selected:
        base_args = baselines[model]['configuration']['args']
        grid = [ensure(change(base_args, batch_size=batch, learning_rate=rate),
                       'batch_lr', model)
                for batch, rate in itertools.product([64, 128, 256], [0.0003, 0.001, 0.003])]
        decay_results = []
        for candidate in ranking(grid)[:2]:
            for decay in [0.0, 0.0001, 0.001]:
                decay_results.append(ensure(change(candidate['configuration']['args'], weight_decay=decay),
                                            'weight_decay', model))
        best[model] = ranking(decay_results)[0]
        make_report()
    for model in selected:
        origin = best[model]['configuration']['args']
        regularization = []
        for augment, rate in itertools.product([False, True], [0.0, 0.2, 0.5]):
            arguments = change(origin, augment=augment, dropout=rate > 0, dropout_rate=rate if rate > 0 else 0.5)
            regularization.append(ensure(arguments, 'regularization', model))
        best[model] = ranking(regularization)[0]
        # BN is an optional tuning addition, not an ablation, in architectures
        # without original BN. Preserve original BN locations in other models.
        if best[model]['args']['normalization'] == 0:
            bn_candidates = [best[model]]
            for rate in sorted({0.0003, 0.001, best[model]['args']['learning_rate']}):
                bn_candidates.append(ensure(change(best[model]['configuration']['args'],
                                                    normalization=1, learning_rate=rate),
                                             'normalization_tuning', model))
            best[model] = ranking(bn_candidates)[0]
        make_report()
    repeats = {}
    for model in selected:
        repeats[model] = [best[model]]
        for seed in [43, 44]:
            repeats[model].append(ensure(change(best[model]['configuration']['args'], seed=seed),
                                         'repeat_seed', model))
    import statistics
    summaries = []
    for model, results in repeats.items():
        scores = [r['checkpoint_validation_accuracy'] for r in results]
        summaries.append({'model': model, 'mean_validation_accuracy': statistics.mean(scores),
                          'std_validation_accuracy': statistics.stdev(scores), 'scores': scores,
                          'seed42_configuration': best[model], 'runs': results})
    summaries.sort(key=lambda r: -r['mean_validation_accuracy'])
    final = {'status': 'completed', 'winner': summaries[0], 'candidates': summaries,
             'test_data_used': False, 'ablation_performed': False,
             'validation_target_met': summaries[0]['mean_validation_accuracy'] >= 94.0,
             'completed_at': time.time()}
    write(BASE / 'final_selection.json', final)
    write(BASE / 'study_status.json', final)
    make_report()


if __name__ == '__main__':
    try:
        study()
    except BaseException as error:
        write(BASE / 'study_status.json', {'status': 'failed', 'error': repr(error), 'updated': time.time()})
        try:
            make_report()
        except Exception as report_error:
            print('Could not refresh the failure report:', repr(report_error), flush=True)
        raise
