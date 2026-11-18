"""Run the approved GoogLeNet and ResNet validation-only sweep."""
from itertools import product
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from scripts.fashion_mnist_study import BASE, change, ranking, signature, write

RUN = Path(__file__).resolve().parent
MODELS = ("googlenet", "resnet")


def read(path):
    return json.loads(path.read_text())


def execute(stage, args):
    model = args[args.index("--model") + 1]
    folder = RUN / f"{model}_{stage}_{signature(args)}"
    result_path = folder / "result.json"
    if result_path.exists():
        result = read(result_path)
        assert result["status"] == "completed"
        return result

    folder.mkdir(exist_ok=True)
    if (folder / "history.jsonl").exists():
        raise RuntimeError(f"Previous incomplete training needs inspection: {folder}")
    write(folder / "config.json", {"id": folder.name, "stage": stage, "args": args})
    write(RUN / "progress.json", {"status": "running", "stage": stage, "run": folder.name})
    print(f"START {folder.name} {args}", flush=True)
    with (folder / "train.log").open("w") as log:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/fashion_mnist_experiment.py"), str(folder)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    result = read(result_path)
    assert result["status"] == "completed"
    assert result["train_samples"] == 55000 and result["validation_samples"] == 5000
    assert math.isfinite(result["checkpoint_validation_accuracy"])
    assert Path(result["checkpoint"]).is_file()
    print(f"DONE {folder.name}: {result['checkpoint_validation_accuracy']:.2f}%", flush=True)
    return result


def arguments(result):
    return result["configuration"]["args"]


def best(results):
    return ranking(results)[0]


def worth_high_rate(grid):
    results = {r["args"]["learning_rate"]: r for r in grid if r["args"]["batch_size"] == 256}
    fast, slow = results[0.001], results[0.0003]
    score = fast["checkpoint_validation_accuracy"]
    if score <= slow["checkpoint_validation_accuracy"]:
        return False
    history_path = Path(fast["checkpoint"]).parent / "history.jsonl"
    history = [json.loads(line) for line in history_path.read_text().splitlines()]
    tail = history[-5:]
    return len(tail) == 5 and all(row["Validation accuracy (%)"] >= score - 2 for row in tail)


def tune(model, baseline):
    origin = arguments(baseline)
    regularized = [baseline]
    if model == "googlenet":
        for bn, augment in ((0, True), (1, False), (1, True)):
            regularized.append(execute("bn_aug", change(origin, normalization=bn, augment=augment)))
    else:
        regularized.append(execute("augmentation", change(origin, augment=True)))

    start = best(regularized)
    origin = arguments(start)
    grid = [start]
    for batch, rate in product((128, 256), (0.0003, 0.001)):
        if (batch, rate) == (128, 0.001):
            continue
        grid.append(execute("batch_lr", change(origin, batch_size=batch, learning_rate=rate)))
    high_rate = worth_high_rate(grid)
    print(f"OPTIONAL {model}: LR 0.003 {'enabled' if high_rate else 'skipped'}", flush=True)
    if high_rate:
        grid.append(execute("batch_lr_probe", change(origin, batch_size=256, learning_rate=0.003)))

    start = best(grid)
    no_decay = execute("weight_decay", change(arguments(start), weight_decay=0))
    start = best([start, no_decay])
    origin = arguments(start)
    dropout_results = [start]
    comparison_rates = (0.0, 0.2) if model == "googlenet" else (0.2, 0.5)
    for rate in comparison_rates:
        args = change(origin, dropout=rate > 0, dropout_rate=rate or 0.5)
        dropout_results.append(execute("dropout", args))
    final = best(dropout_results)

    runs = [final]
    for seed in (43, 44):
        runs.append(execute("repeat_seed", change(arguments(final), seed=seed)))
    scores = [run["checkpoint_validation_accuracy"] for run in runs]
    assert [run["args"]["seed"] for run in runs] == [42, 43, 44]
    return {
        "model": model,
        "configuration": final["args"],
        "baseline_accuracy": baseline["checkpoint_validation_accuracy"],
        "seed_scores": dict(zip((42, 43, 44), scores)),
        "mean_accuracy": statistics.mean(scores),
        "std_accuracy": statistics.stdev(scores),
        "minimum_accuracy": min(scores),
        "run_directories": [str(Path(run["checkpoint"]).parent) for run in runs],
        "optional_high_rate_executed": high_rate,
    }


def study():
    summaries = {}
    for model in MODELS:
        baseline = read(BASE / f"baseline_{model}" / "result.json")
        assert baseline["status"] == "completed"
        summaries[model] = tune(model, baseline)
        write(RUN / "summary.json", summaries)
    completed = list(RUN.glob("*/result.json"))
    planned = 20 + sum(s["optional_high_rate_executed"] for s in summaries.values())
    assert len(completed) == planned, (len(completed), planned)
    write(RUN / "progress.json", {"status": "completed", "new_runs": len(completed)})
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    try:
        study()
    except BaseException as error:
        progress_path = RUN / "progress.json"
        previous = read(progress_path) if progress_path.exists() else {}
        write(progress_path, {**previous, "status": "failed", "error": repr(error)})
        raise

