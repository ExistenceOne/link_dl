import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from _01_code._99_common_utils.early_stopping import EarlyStopping
from _03_homeworks.homework_2.titanic_dataset import get_preprocessed_dataset

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
      early_stop_patience=2,
      early_stop_delta=1e-5,
      n_hidden_unit_list=[8, 8],
    )
    self.logged = []

  def log(self, metrics):
    self.logged.append(metrics)


class TitanicTrainingTest(unittest.TestCase):
  def test_first_epoch_checkpoint_survives_worsening_loss(self):
    model = torch.nn.Linear(1, 1)
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    with tempfile.TemporaryDirectory() as directory:
      stopper = EarlyStopping(patience=2, project_name="test",
                              checkpoint_file_path=directory, run_time_str="first")
      stopper.check_and_save(0.5, model)
      self.assertTrue(Path(stopper.file_path).is_file())
      with torch.no_grad():
        model.weight.add_(10)
      stopper.check_and_save(0.6, model)
      self.assertTrue(stopper.check_and_save(0.7, model)[1])
      saved = torch.load(stopper.file_path, weights_only=True)
      for key in initial:
        self.assertTrue(torch.equal(saved[key], initial[key]))

  def test_loss_is_sample_weighted_and_returns_saved_epoch(self):
    run = FakeRun()
    samples = [{"input": torch.full((10,), x), "target": torch.tensor(0)}
               for x in (0.0, 0.0, 3.0)]
    model = torch.nn.Linear(10, 1, bias=False)
    with torch.no_grad():
      model.weight.fill_(0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    expected = (2 * math.log(2) + math.log1p(math.exp(3))) / 3
    loader = DataLoader(samples, batch_size=2)
    with tempfile.TemporaryDirectory() as directory:
      result = training_loop(model, optimizer, loader, loader, run, directory, "weighted")
      self.assertAlmostEqual(run.logged[0]["Training loss"], expected, places=6)
      self.assertAlmostEqual(run.logged[0]["Validation loss"], expected, places=6)
      self.assertEqual(result["best_epoch"], 1)
      self.assertTrue(Path(result["checkpoint_path"]).exists())

  def test_split_is_independent_of_global_rng(self):
    first, validation, test = get_preprocessed_dataset()
    torch.rand(57)
    second, second_validation, _ = get_preprocessed_dataset()
    self.assertEqual(first.indices, second.indices)
    self.assertEqual(validation.indices, second_validation.indices)
    self.assertEqual((len(first), len(validation), len(test)), (713, 178, 418))
    self.assertTrue(torch.isfinite(first.dataset.X).all())
    self.assertTrue(torch.isfinite(test.X).all())

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
    with tempfile.TemporaryDirectory() as checkpoint_dir:
      training_loop(model, optimizer, loader, loader, run, checkpoint_dir, "test")

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

  def test_early_stops_when_validation_loss_does_not_improve(self):
    run = FakeRun()
    run.config.epochs = 10
    run.config.learning_rate = 0.0
    samples = [{"input": torch.zeros(10), "target": torch.tensor(0)}]
    loader = DataLoader(samples, batch_size=1)
    model, optimizer = get_model_and_optimizer(run)

    with tempfile.TemporaryDirectory() as checkpoint_dir:
      training_loop(model, optimizer, loader, loader, run, checkpoint_dir, "constant")

    self.assertEqual(len(run.logged), 3)
    self.assertEqual(run.logged[-1]["Epoch"], 3)


if __name__ == "__main__":
  unittest.main()
