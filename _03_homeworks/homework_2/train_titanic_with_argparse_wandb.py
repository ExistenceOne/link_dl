import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import datetime
import wandb
import argparse

from pathlib import Path
BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
print("BASE_PATH:", BASE_PATH)

import sys
sys.path.append(BASE_PATH)

from _03_homeworks.homework_2.titanic_dataset import get_preprocessed_dataset


def get_data(run):
  train_dataset, validation_dataset, _ = get_preprocessed_dataset()
  print(len(train_dataset), len(validation_dataset))

  train_data_loader = DataLoader(dataset=train_dataset, batch_size=run.config.batch_size, shuffle=True)
  validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset))

  return train_data_loader, validation_data_loader


class MyModel(nn.Module):
  def __init__(self, n_input, n_output, run):
    super().__init__()
    activation_fn = {
      "sigmoid": nn.Sigmoid,
      "relu": nn.ReLU,
      "elu": nn.ELU,
      "leaky_relu": nn.LeakyReLU,
    }[run.config.activation]

    self.model = nn.Sequential(
      nn.Linear(n_input, run.config.n_hidden_unit_list[0]),
      activation_fn(),
      nn.Linear(run.config.n_hidden_unit_list[0], run.config.n_hidden_unit_list[1]),
      activation_fn(),
      nn.Linear(run.config.n_hidden_unit_list[1], n_output),
    )

  def forward(self, x):
    x = self.model(x)
    return x


def get_model_and_optimizer(run):
  my_model = MyModel(n_input=10, n_output=1, run=run)
  optimizer = optim.SGD(my_model.parameters(), lr=run.config.learning_rate)

  return my_model, optimizer


def training_loop(model, optimizer, train_data_loader, validation_data_loader, run):
  n_epochs = run.config.epochs
  loss_fn = nn.BCEWithLogitsLoss()
  next_print_epoch = 100

  for epoch in range(1, n_epochs + 1):
    loss_train = 0.0
    num_trains = 0
    for train_batch in train_data_loader:
      input = train_batch['input']
      target = train_batch['target'].float().unsqueeze(1)
      output_train = model(input)
      loss = loss_fn(output_train, target)
      loss_train += loss.item()
      num_trains += 1

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    loss_validation = 0.0
    num_validations = 0
    with torch.no_grad():
      for validation_batch in validation_data_loader:
        input = validation_batch['input']
        target = validation_batch['target'].float().unsqueeze(1)
        output_validation = model(input)
        loss = loss_fn(output_validation, target)
        loss_validation += loss.item()
        num_validations += 1

    run.log({
      "Epoch": epoch,
      "Training loss": loss_train / num_trains,
      "Validation loss": loss_validation / num_validations
    })

    if epoch >= next_print_epoch:
      print(
        f"Epoch {epoch}, "
        f"Training loss {loss_train / num_trains:.4f}, "
        f"Validation loss {loss_validation / num_validations:.4f}"
      )
      next_print_epoch += 100


def main(args):
  torch.manual_seed(42)
  current_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

  config = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'learning_rate': args.learning_rate,
    'activation': args.activation,
    'n_hidden_unit_list': [20, 20],
  }

  with wandb.init(
    mode="online" if args.wandb else "disabled",
    project="my_model_training",
    notes="Titanic survival classification",
    tags=["my_model", "titanic"],
    name=f"{current_time_str}_{args.activation}_bs{args.batch_size}_lr{args.learning_rate:g}",
    config=config
  ) as run:
    print(args)
    print(run.config)

    # 가로축을 Epoch, 세로축을 Training loss로 그려줘.”
    run.define_metric("Training loss", step_metric="Epoch")

    # 가로축을 Epoch, 세로축을 Validation loss로 그려줘, 실험 요약에는 가장 낮았던 값을 남겨줘.”
    run.define_metric("Validation loss", step_metric="Epoch", summary="min")

    train_data_loader, validation_data_loader = get_data(run)

    linear_model, optimizer = get_model_and_optimizer(run)

    print("#" * 50, 1)

    training_loop(
      model=linear_model,
      optimizer=optimizer,
      train_data_loader=train_data_loader,
      validation_data_loader=validation_data_loader,
      run=run
    )


# https://docs.wandb.ai/models/track/config
if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--wandb", action=argparse.BooleanOptionalAction, default=True, help="True or False"
  )

  parser.add_argument(
    "-b", "--batch_size", type=int, default=128, help="Batch size (int, default: 128)"
  )

  parser.add_argument(
    "-e", "--epochs", type=int, default=10_000, help="Number of training epochs (int, default:10_000)"
  )

  parser.add_argument(
    "-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate (float, default: 1e-3)"
  )

  parser.add_argument(
    "-a", "--activation", choices=["sigmoid", "relu", "elu", "leaky_relu"],
    default="relu", help="Activation function (default: relu)"
  )

  args = parser.parse_args()

  main(args)
