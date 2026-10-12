"""한 실험 그룹의 36개 결과만 검증하고 최적 모델 순서로 정리한다."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_runs(directory, group):
  records = []
  for path in sorted(Path(directory).glob("*.json")):
    record = json.loads(path.read_text())
    if record.get("group") == group:
      records.append(dict(record, metadata_path=str(path.resolve())))
  table = pd.DataFrame(records)
  expected = set(itertools.product(
    ["sigmoid", "relu", "elu", "leaky_relu"], [32, 64, 128], [0.01, 0.001, 0.0001],
  ))
  keys = ["activation", "batch_size", "learning_rate"]
  if len(table) != 36 or not set(keys).issubset(table.columns):
    raise ValueError("Exactly 36 completed runs are required in the selected group")
  actual = set(table[keys].itertuples(index=False, name=None))
  if actual != expected or table.duplicated(keys).any():
    raise ValueError("Missing, unexpected, or duplicate hyperparameter combinations")
  if not np.isfinite(table[["best_validation_loss", "best_validation_accuracy"]].to_numpy()).all():
    raise ValueError("Non-finite experiment result")
  table = table.sort_values(
    ["best_validation_loss", "best_validation_accuracy", "run_id"],
    ascending=[True, False, True],
  ).reset_index(drop=True)
  table.to_csv(Path(directory) / "experiment_results.csv", index=False)
  return table


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("directory", type=Path)
  parser.add_argument("--group", required=True)
  args = parser.parse_args()
  table = summarize_runs(args.directory, args.group)
  print(table[["activation", "batch_size", "learning_rate", "best_epoch",
               "best_validation_loss", "best_validation_accuracy"]].to_string(index=False))
  print("BEST_METADATA", table.iloc[0].metadata_path)
