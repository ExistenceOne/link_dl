"""선택한 실험의 가중치로 Kaggle 제출 파일을 생성한다."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from _03_homeworks.homework_2.titanic_dataset import get_preprocessed_dataset
from _03_homeworks.homework_2.train_titanic_with_argparse_wandb import MyModel


def predict_submission(run_metadata_path, output_path):
  metadata = json.loads(Path(run_metadata_path).read_text())
  run = SimpleNamespace(config=SimpleNamespace(
    activation=metadata["activation"], n_hidden_unit_list=metadata["hidden_sizes"],
  ))
  model = MyModel(10, 1, run)
  checkpoint = Path(metadata["checkpoint_path"])
  if not checkpoint.is_absolute():
    checkpoint = ROOT / checkpoint
  model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
  model.eval()
  _, _, test = get_preprocessed_dataset(seed=metadata["seed"])
  loader = DataLoader(test, batch_size=128, shuffle=False)
  with torch.no_grad():
    logits = torch.cat([model(batch["input"]).reshape(-1) for batch in loader])
  if not torch.isfinite(logits).all():
    raise ValueError("Test logits must be finite")
  ids = pd.read_csv(Path(__file__).parent / "test.csv")["PassengerId"]
  result = pd.DataFrame({"PassengerId": ids, "Survived": (logits >= 0).long().numpy()})
  if len(result) != len(test) or result.isna().any().any() or ids.duplicated().any():
    raise ValueError("Test rows and PassengerId must align without missing or duplicate values")
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  result.to_csv(output_path, index=False)
  return result


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("metadata", type=Path)
  parser.add_argument("--output", type=Path, default=Path(__file__).parent / "submission.csv")
  args = parser.parse_args()
  result = predict_submission(args.metadata, args.output)
  print(f"Saved {len(result)} predictions: {args.output}")
