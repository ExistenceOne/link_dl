import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from _03_homeworks.homework_2.train_titanic_with_argparse_wandb import MyModel


class PredictionTest(unittest.TestCase):
  def test_metadata_selects_checkpoint_and_preserves_test_ids(self):
    from _03_homeworks.homework_2.predict_titanic import predict_submission

    with tempfile.TemporaryDirectory() as directory:
      directory = Path(directory)
      paths = []
      for label, bias in ((0, -1.0), (1, 1.0)):
        run = SimpleNamespace(config=SimpleNamespace(
          activation="sigmoid", n_hidden_unit_list=[4, 3]))
        model = MyModel(10, 1, run)
        with torch.no_grad():
          for parameter in model.parameters():
            parameter.zero_()
          model.model[-1].bias.fill_(bias)
        checkpoint = directory / f"model_{label}.pt"
        torch.save(model.state_dict(), checkpoint)
        metadata = directory / f"model_{label}.json"
        metadata.write_text(json.dumps({
          "activation": "sigmoid", "hidden_sizes": [4, 3], "seed": 42,
          "checkpoint_path": str(checkpoint),
        }))
        paths.append(metadata)
      original = pd.read_csv(Path(__file__).parent / "test.csv")
      for index in (0, 1, 0):
        output = directory / "submission.csv"
        predict_submission(paths[index], output)
        result = pd.read_csv(output)
        self.assertEqual(list(result.columns), ["PassengerId", "Survived"])
        self.assertEqual(len(result), 418)
        self.assertEqual(result.PassengerId.tolist(), original.PassengerId.tolist())
        self.assertEqual(result.Survived.tolist(), [index] * 418)
        self.assertFalse(result.isna().any().any())


if __name__ == "__main__":
  unittest.main()
