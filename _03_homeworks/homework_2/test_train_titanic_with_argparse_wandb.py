import math
import unittest
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from _03_homeworks.homework_2.train_titanic_with_argparse_wandb import (
  get_data,
  get_model_and_optimizer,
  training_loop,
)


class FakeRun:
  def __init__(self):
    self.config = SimpleNamespace(
      batch_size=2,
      epochs=1,
      learning_rate=1e-3,
      n_hidden_unit_list=[8, 8],
    )
    self.logged = []

  def log(self, metrics):
    self.logged.append(metrics)


class TitanicTrainingTest(unittest.TestCase):
  def test_data_loaders_use_titanic_classification_samples(self):
    train_loader, validation_loader = get_data(FakeRun())

    self.assertEqual(len(train_loader.dataset) + len(validation_loader.dataset), 891)
    batch = next(iter(train_loader))
    self.assertEqual(batch["input"].shape[1], 10)
    self.assertEqual(batch["target"].dtype, torch.int64)

  def test_one_epoch_logs_finite_loss_and_accuracy(self):
    run = FakeRun()
    samples = [
      {"input": torch.zeros(10), "target": torch.tensor(0)},
      {"input": torch.ones(10), "target": torch.tensor(1)},
      {"input": torch.full((10,), 2.0), "target": torch.tensor(0)},
    ]
    loader = DataLoader(samples, batch_size=2)
    model, optimizer = get_model_and_optimizer(run, n_input=10)

    self.assertEqual(model(torch.zeros(2, 10)).shape, (2, 2))
    training_loop(model, optimizer, loader, loader, run)

    self.assertEqual(len(run.logged), 1)
    metrics = run.logged[0]
    self.assertEqual(metrics["Epoch"], 1)
    for key in ("Training loss", "Validation loss"):
      self.assertTrue(math.isfinite(metrics[key]))
    for key in ("Training accuracy (%)", "Validation accuracy (%)"):
      self.assertGreaterEqual(metrics[key], 0)
      self.assertLessEqual(metrics[key], 100)


if __name__ == "__main__":
  unittest.main()
