import itertools
import json
import tempfile
import unittest
from pathlib import Path


class SummaryTest(unittest.TestCase):
  def write_grid(self, directory):
    for i, (activation, batch, rate) in enumerate(itertools.product(
        ["sigmoid", "relu", "elu", "leaky_relu"], [32, 64, 128], [0.01, 0.001, 0.0001])):
      record = dict(run_id=f"run{i:02d}", group="grid", activation=activation,
                    batch_size=batch, learning_rate=rate,
                    best_validation_loss=0.4 + i / 1000,
                    best_validation_accuracy=80.0, best_epoch=10,
                    checkpoint_path="unused.pt")
      (directory / f"run{i:02d}.json").write_text(json.dumps(record))

  def test_complete_grid_and_missing_duplicate_rejection(self):
    from _03_homeworks.homework_2.summarize_titanic_runs import summarize_runs

    with tempfile.TemporaryDirectory() as directory:
      directory = Path(directory)
      self.write_grid(directory)
      table = summarize_runs(directory, "grid")
      self.assertEqual(len(table), 36)
      self.assertEqual(table.iloc[0].run_id, "run00")
      duplicate = directory / "duplicate.json"
      duplicate.write_text((directory / "run00.json").read_text())
      with self.assertRaises(ValueError):
        summarize_runs(directory, "grid")
      duplicate.unlink()
      (directory / "run00.json").unlink()
      with self.assertRaises(ValueError):
        summarize_runs(directory, "grid")

  def test_nonfinite_result_is_rejected(self):
    from _03_homeworks.homework_2.summarize_titanic_runs import summarize_runs

    with tempfile.TemporaryDirectory() as directory:
      directory = Path(directory)
      self.write_grid(directory)
      path = directory / "run00.json"
      record = json.loads(path.read_text())
      record["best_validation_loss"] = float("nan")
      path.write_text(json.dumps(record))
      with self.assertRaises(ValueError):
        summarize_runs(directory, "grid")


if __name__ == "__main__":
  unittest.main()
