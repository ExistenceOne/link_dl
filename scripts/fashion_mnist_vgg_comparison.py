"""Nine fresh, resumable VGG runs; compare validation only, never test data."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '_03_homeworks/homework_3/experiments/vgg_variant_comparison'
VARIANTS = ('resize', 'pad', 'no_last_pool')
SEEDS = (42, 43, 44)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def configurations():
    fixed = ['--model', 'vgg', '--wandb', '-b', '256', '-e', '100', '-v', '1',
             '-p', '10', '-d', '0.00001', '-o', '3', '-r', '0.0003', '-w', '0.0001',
             '--split_seed', '42', '--mean', '0.2855656942074541',
             '--std', '0.3527285864947437', '-n', '1', '--augment', '--no-dropout']
    return [dict(id=f'vgg_{variant}_seed{seed}', stage='vgg_variant_comparison',
                 args=fixed + ['--vgg_variant', variant, '--seed', str(seed)])
            for seed in SEEDS for variant in VARIANTS]


def summarize(results):
    indexed = {(r['args']['vgg_variant'], r['args']['seed']): r for r in results}
    if len(results) != 9 or set(indexed) != {(v, s) for v in VARIANTS for s in SEEDS}:
        raise ValueError('Expected exactly nine variant/seed pairs')
    if any(r['status'] != 'completed' for r in results):
        raise ValueError('All runs must be completed')
    for key in ('training_indices_sha256', 'validation_indices_sha256', 'parameter_count'):
        if len({r[key] for r in results}) != 1:
            raise ValueError(f'Comparison mismatch: {key}')
    for seed in SEEDS:
        for key in ('initial_state_sha256', 'shuffle_state_sha256'):
            if len({indexed[v, seed][key] for v in VARIANTS}) != 1:
                raise ValueError(f'Paired seed {seed} mismatch: {key}')
    variants = {}
    for variant in VARIANTS:
        runs = [indexed[variant, seed] for seed in SEEDS]
        scores = [r['checkpoint_validation_accuracy'] for r in runs]
        deltas = [score - indexed['resize', seed]['checkpoint_validation_accuracy']
                  for score, seed in zip(scores, SEEDS)]
        variants[variant] = dict(
            seeds=list(SEEDS), accuracies=scores, accuracy_mean=statistics.mean(scores),
            accuracy_std=statistics.stdev(scores), paired_delta_pp=deltas,
            paired_delta_mean_pp=statistics.mean(deltas),
            validation_loss_mean=statistics.mean(r['checkpoint_validation_loss'] for r in runs),
            epochs=[r['epochs'] for r in runs],
            wall_seconds_mean=statistics.mean(r['elapsed_seconds'] for r in runs),
            seconds_per_epoch_mean=statistics.mean(r['elapsed_seconds'] / r['epochs'] for r in runs),
            peak_gpu_memory_mb_mean=statistics.mean(r['peak_gpu_memory_mb'] for r in runs),
            parameter_count=runs[0]['parameter_count'])
    return dict(status='completed', variants=variants,
                winner=max(VARIANTS, key=lambda v: variants[v]['accuracy_mean']),
                selection_metric='Mean validation accuracy of minimum-validation-loss checkpoints',
                std_definition='sample standard deviation (ddof=1), three seeds',
                test_data_used=False, paired_initialization_verified=True,
                shuffle_state_verified=True, split_verified=True,
                runs=results)


def export(results):
    summary = summarize(results)
    write_json(BASE / 'comparison.json', summary)
    fields = ['variant', 'seed', 'checkpoint_validation_accuracy', 'checkpoint_validation_loss',
              'epochs', 'elapsed_seconds', 'peak_gpu_memory_mb', 'parameter_count', 'wandb_url']
    with (BASE / 'runs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key) for key in fields}
            row.update(variant=result['args']['vgg_variant'], seed=result['args']['seed'])
            writer.writerow(row)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for i, variant in enumerate(VARIANTS):
        metrics = summary['variants'][variant]
        axes[0].errorbar(i, metrics['accuracy_mean'], yerr=metrics['accuracy_std'],
                        fmt='o', capsize=5, color=f'C{i}')
        axes[0].scatter([i - .12, i, i + .12], metrics['accuracies'], alpha=.6, color=f'C{i}')
        axes[1].bar(i, metrics['wall_seconds_mean'] / 60, color=f'C{i}')
    for axis in axes:
        axis.set_xticks(range(3), ['Resize 32', 'Zero-pad 32', '28, no last pool'])
        axis.grid(axis='y', alpha=.2)
    axes[0].set(ylabel='Validation accuracy (%)', title='Three paired seeds: mean +/- sample SD')
    axes[1].set(ylabel='Wall time per run (minutes)', title='Full run including setup and checkpoint restore')
    figure.savefig(BASE / 'comparison.png', dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for row, seed in enumerate(SEEDS):
        for variant in VARIANTS:
            folder = BASE / 'runs' / f'vgg_{variant}_seed{seed}'
            history = [json.loads(line) for line in (folder / 'history.jsonl').read_text().splitlines()]
            for col, metric in enumerate(('Validation loss', 'Validation accuracy (%)')):
                axes[row, col].plot([h['Epoch'] for h in history], [h[metric] for h in history], label=variant)
                axes[row, col].set(xlabel='Epoch', ylabel=metric, title=f'Seed {seed}')
                axes[row, col].legend()
    figure.savefig(BASE / 'validation_curves.png', dpi=160)
    plt.close(figure)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report_only', action='store_true')
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    results = []
    for config in configurations():
        folder = BASE / 'runs' / config['id']
        folder.mkdir(parents=True, exist_ok=True)
        config_path = folder / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise RuntimeError(f'Configuration mismatch: {folder}')
        write_json(config_path, config)
        result_path = folder / 'result.json'
        if not result_path.exists():
            if args.report_only:
                raise RuntimeError(f'Missing result: {folder}')
            if (folder / 'history.jsonl').exists() or (folder / 'status.json').exists():
                raise RuntimeError(f'Incomplete prior attempt; inspect before retry: {folder}')
            write_json(BASE / 'status.json', dict(status='running', completed=len(results),
                                                 current=config['id'], updated=time.time()))
            print('START', config['id'], flush=True)
            with (folder / 'train.log').open('w') as log:
                process = subprocess.run([sys.executable, str(ROOT / 'scripts/fashion_mnist_experiment.py'),
                                          str(folder)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError(f'Worker failed: {folder}/train.log')
        result = json.loads(result_path.read_text())
        if result['configuration'] != config:
            raise RuntimeError(f'Result configuration mismatch: {folder}')
        results.append(result)
        print(f"DONE {config['id']}: {result['checkpoint_validation_accuracy']:.2f}%", flush=True)
    summary = export(results)
    write_json(BASE / 'status.json', dict(status='completed', completed=9, winner=summary['winner'],
                                         updated=time.time()))
    print(json.dumps(summary['variants'], indent=2), flush=True)


if __name__ == '__main__':
    main()
