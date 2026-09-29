import math
import unittest
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from _03_homeworks.homework_2.train_titanic_with_argparse_wandb import (
  binary_accuracy_f1,
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
      activation="relu",
      n_hidden_unit_list=[8, 8],
    )
    self.logged = []

  def log(self, metrics):
    self.logged.append(metrics)


class TitanicTrainingTest(unittest.TestCase):
  def test_binary_accuracy_and_f1(self):
    logits = torch.tensor([[2.0], [-1.0], [0.2], [-2.0]])
    targets = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
    self.assertEqual(binary_accuracy_f1(logits, targets), (50.0, 0.5))

    logits = torch.tensor([[-2.0], [-1.0]])
    targets = torch.tensor([[0.0], [0.0]])
    self.assertEqual(binary_accuracy_f1(logits, targets), (100.0, 0.0))

  def test_activation_applies_to_both_hidden_layers(self):
    activations = {
      "sigmoid": torch.nn.Sigmoid,
      "relu": torch.nn.ReLU,
      "elu": torch.nn.ELU,
      "leaky_relu": torch.nn.LeakyReLU,
    }
    for name, layer_type in activations.items():
      with self.subTest(activation=name):
        run = FakeRun()
        run.config.activation = name
        model, _ = get_model_and_optimizer(run)
        self.assertIsInstance(model.model[1], layer_type)
        self.assertIsInstance(model.model[3], layer_type)

  def test_data_loaders_use_titanic_classification_samples(self):
    train_loader, validation_loader = get_data(FakeRun())

    self.assertEqual(len(train_loader.dataset) + len(validation_loader.dataset), 891)
    batch = next(iter(train_loader))
    self.assertEqual(batch["input"].shape[1], 10)
    self.assertEqual(batch["target"].dtype, torch.int64)

  def test_one_epoch_logs_finite_classification_loss(self):
    run = FakeRun()
    samples = [
      {"input": torch.zeros(10), "target": torch.tensor(0)},
      {"input": torch.ones(10), "target": torch.tensor(1)},
      {"input": torch.full((10,), 2.0), "target": torch.tensor(0)},
    ]
    loader = DataLoader(samples, batch_size=2)
    model, optimizer = get_model_and_optimizer(run)

    self.assertEqual(model(torch.zeros(2, 10)).shape, (2, 1))
    self.assertIsInstance(optimizer, torch.optim.SGD)
    training_loop(model, optimizer, loader, loader, run)

    self.assertEqual(len(run.logged), 1)
    metrics = run.logged[0]
    self.assertEqual(metrics["Epoch"], 1)
    self.assertEqual(set(metrics), {
      "Epoch", "Training loss", "Validation loss",
      "Training accuracy (%)", "Validation accuracy (%)",
      "Training F1", "Validation F1",
    })
    self.assertTrue(math.isfinite(metrics["Training loss"]))
    self.assertTrue(math.isfinite(metrics["Validation loss"]))
    for key in ("Training accuracy (%)", "Validation accuracy (%)"):
      self.assertGreaterEqual(metrics[key], 0.0)
      self.assertLessEqual(metrics[key], 100.0)
    for key in ("Training F1", "Validation F1"):
      self.assertGreaterEqual(metrics[key], 0.0)
      self.assertLessEqual(metrics[key], 1.0)


if __name__ == "__main__":
  unittest.main()
