import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import wandb


BASE_PATH = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_PATH))

from _03_homeworks.homework_2.titanic_dataset import get_preprocessed_dataset


def get_data(run):
  train_dataset, validation_dataset, _ = get_preprocessed_dataset()
  train_data_loader = DataLoader(
    dataset=train_dataset, batch_size=run.config.batch_size, shuffle=True
  )
  validation_data_loader = DataLoader(
    dataset=validation_dataset, batch_size=run.config.batch_size
  )
  return train_data_loader, validation_data_loader


class MyModel(nn.Module):
  def __init__(self, n_input, n_output, run):
    super().__init__()
    hidden_units = run.config.n_hidden_unit_list
    self.model = nn.Sequential(
      nn.Linear(n_input, hidden_units[0]),
      nn.ReLU(),
      nn.Linear(hidden_units[0], hidden_units[1]),
      nn.ReLU(),
      nn.Linear(hidden_units[1], n_output),
    )

  def forward(self, x):
    return self.model(x)


def get_model_and_optimizer(run, n_input):
  model = MyModel(n_input=n_input, n_output=2, run=run)
  optimizer = optim.Adam(model.parameters(), lr=run.config.learning_rate)
  return model, optimizer


def run_epoch(model, data_loader, loss_fn, optimizer=None):
  is_training = optimizer is not None
  model.train(is_training)
  total_loss = 0.0
  total_correct = 0
  total_samples = 0

  with torch.set_grad_enabled(is_training):
    for batch in data_loader:
      inputs, targets = batch["input"], batch["target"]
      logits = model(inputs)
      loss = loss_fn(logits, targets)

      if is_training:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      batch_size = targets.size(0)
      total_loss += loss.item() * batch_size
      total_correct += (logits.argmax(dim=1) == targets).sum().item()
      total_samples += batch_size

  return total_loss / total_samples, 100.0 * total_correct / total_samples


def training_loop(model, optimizer, train_data_loader, validation_data_loader, run):
  loss_fn = nn.CrossEntropyLoss()

  for epoch in range(1, run.config.epochs + 1):
    train_loss, train_accuracy = run_epoch(model, train_data_loader, loss_fn, optimizer)
    validation_loss, validation_accuracy = run_epoch(model, validation_data_loader, loss_fn)

    run.log({
      "Epoch": epoch,
      "Training loss": train_loss,
      "Training accuracy (%)": train_accuracy,
      "Validation loss": validation_loss,
      "Validation accuracy (%)": validation_accuracy,
    })

    if epoch == 1 or epoch % 100 == 0 or epoch == run.config.epochs:
      print(
        f"Epoch {epoch}, "
        f"Training loss {train_loss:.4f}, accuracy {train_accuracy:.2f}%, "
        f"Validation loss {validation_loss:.4f}, accuracy {validation_accuracy:.2f}%"
      )


def main(args):
  torch.manual_seed(42)
  config = {
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "learning_rate": 1e-3,
    "n_hidden_unit_list": [20, 20],
  }

  with wandb.init(
    mode="online" if args.wandb else "disabled",
    project="titanic_survival",
    tags=["titanic", "classification"],
    name=datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S"),
    config=config,
  ) as run:
    run.define_metric("Training loss", step_metric="Epoch")
    run.define_metric("Validation loss", step_metric="Epoch", summary="min")
    run.define_metric("Training accuracy (%)", step_metric="Epoch")
    run.define_metric("Validation accuracy (%)", step_metric="Epoch", summary="max")

    train_data_loader, validation_data_loader = get_data(run)
    n_input = train_data_loader.dataset[0]["input"].numel()
    model, optimizer = get_model_and_optimizer(run, n_input)
    training_loop(model, optimizer, train_data_loader, validation_data_loader, run)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--wandb", action=argparse.BooleanOptionalAction, default=False,
    help="Enable Weights & Biases logging (default: disabled)",
  )
  parser.add_argument(
    "-b", "--batch_size", type=int, default=32, help="Batch size (default: 32)"
  )
  parser.add_argument(
    "-e", "--epochs", type=int, default=100, help="Training epochs (default: 100)"
  )
  main(parser.parse_args())
