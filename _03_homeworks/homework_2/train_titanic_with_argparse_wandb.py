import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import datetime
import wandb
import argparse
import csv
import json
import math
from uuid import uuid4

from pathlib import Path
BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl

import sys
sys.path.append(BASE_PATH)

from _03_homeworks.homework_2.titanic_dataset import get_preprocessed_dataset
from _01_code._99_common_utils.early_stopping import EarlyStopping


def get_data(run):
  seed = getattr(run.config, "seed", 42)
  train_dataset, validation_dataset, _ = get_preprocessed_dataset(seed=seed)

  train_data_loader = DataLoader(
    dataset=train_dataset, batch_size=run.config.batch_size, shuffle=True,
    generator=torch.Generator().manual_seed(seed),
  )
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


def binary_accuracy_f1(logits, targets):
  predicted = logits.reshape(-1) >= 0
  actual = targets.reshape(-1).bool()
  accuracy = 100.0 * (predicted == actual).float().mean().item()

  true_positives = (predicted & actual).sum().item()
  false_positives = (predicted & ~actual).sum().item()
  false_negatives = (~predicted & actual).sum().item()
  denominator = 2 * true_positives + false_positives + false_negatives
  f1 = 0.0 if denominator == 0 else 2 * true_positives / denominator
  return accuracy, f1


def training_loop(model, optimizer, train_data_loader, validation_data_loader, run,
                  checkpoint_dir, run_time_str):
  Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
  early_stopping = EarlyStopping(
    patience=run.config.early_stop_patience,
    delta=run.config.early_stop_delta,
    project_name="titanic",
    checkpoint_file_path=str(checkpoint_dir),
    run_time_str=run_time_str,
  )
  n_epochs = run.config.epochs
  if n_epochs < 1 or len(train_data_loader.dataset) == 0 or len(validation_data_loader.dataset) == 0:
    raise ValueError("Training requires positive epochs and nonempty datasets")
  loss_fn = nn.BCEWithLogitsLoss()
  best = {}
  history = []

  for epoch in range(1, n_epochs + 1):
    model.train()
    loss_train = 0.0
    num_trains = 0
    train_logits = []
    train_targets = []
    for train_batch in train_data_loader:
      input = train_batch['input']
      target = train_batch['target'].float().unsqueeze(1)
      output_train = model(input)
      loss = loss_fn(output_train, target)
      # 마지막 작은 배치도 표본 수에 비례해서 반영한다.
      loss_train += loss.item() * target.numel()
      num_trains += target.numel()
      train_logits.append(output_train.detach())
      train_targets.append(target)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    train_accuracy, train_f1 = binary_accuracy_f1(
      torch.cat(train_logits), torch.cat(train_targets)
    )
    loss_validation = 0.0
    num_validations = 0
    validation_logits = []
    validation_targets = []
    model.eval()
    with torch.no_grad():
      for validation_batch in validation_data_loader:
        input = validation_batch['input']
        target = validation_batch['target'].float().unsqueeze(1)
        output_validation = model(input)
        loss = loss_fn(output_validation, target)
        loss_validation += loss.item() * target.numel()
        num_validations += target.numel()
        validation_logits.append(output_validation)
        validation_targets.append(target)

    validation_accuracy, validation_f1 = binary_accuracy_f1(
      torch.cat(validation_logits), torch.cat(validation_targets)
    )

    validation_loss = loss_validation / num_validations
    if not math.isfinite(validation_loss) or not math.isfinite(loss_train):
      raise ValueError(f"Non-finite loss at epoch {epoch}")
    previous_best = early_stopping.val_loss_min
    message, early_stop = early_stopping.check_and_save(validation_loss, model)
    if previous_best is None or early_stopping.val_loss_min < previous_best:
      best = {
        "best_epoch": epoch,
        "best_validation_loss": validation_loss,
        "best_validation_accuracy": validation_accuracy,
        "best_validation_f1": validation_f1,
        "checkpoint_path": str(Path(early_stopping.file_path).resolve()),
      }

    metrics = {
      "Epoch": epoch,
      "Training loss": loss_train / num_trains,
      "Validation loss": validation_loss,
      "Training accuracy (%)": train_accuracy,
      "Validation accuracy (%)": validation_accuracy,
      "Training F1": train_f1,
      "Validation F1": validation_f1,
    }
    run.log(metrics)
    history.append(metrics)

    if epoch == 1 or epoch % 1000 == 0 or early_stop or epoch == n_epochs:
      print(
        f"Epoch {epoch}, "
        f"Training loss {loss_train / num_trains:.4f}, "
        f"accuracy {train_accuracy:.2f}%, F1 {train_f1:.4f}, "
        f"Validation loss {validation_loss:.4f}, "
        f"accuracy {validation_accuracy:.2f}%, F1 {validation_f1:.4f}, "
        f"{message}"
      )

    if early_stop:
      break
  return dict(best, stopped_epoch=epoch, history=history)


def main(args):
  seed = getattr(args, "seed", 42)
  torch.manual_seed(seed)
  torch.set_num_threads(1)
  current_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')
  run_time_str = f"{current_time_str}_{args.activation}_bs{args.batch_size}_lr{args.learning_rate:g}_{uuid4().hex[:8]}"
  output_dir = Path(getattr(args, "output_dir", Path(BASE_PATH) / "_03_homeworks/homework_2/results"))
  output_dir.mkdir(parents=True, exist_ok=True)

  config = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'learning_rate': args.learning_rate,
    'activation': args.activation,
    'early_stop_patience': args.early_stop_patience,
    'early_stop_delta': args.early_stop_delta,
    'n_hidden_unit_list': [20, 20],
    'seed': seed,
    'optimizer': 'SGD',
  }

  with wandb.init(
    mode="online" if args.wandb else "disabled",
    project="my_model_training",
    notes="Titanic survival classification",
    tags=["my_model", "titanic"],
    name=run_time_str,
    group=getattr(args, "group", "hw2"),
    config=config
  ) as run:
    print(run_time_str, config, flush=True)

    # 가로축을 Epoch, 세로축을 Training loss로 그려줘.”
    run.define_metric("Training loss", step_metric="Epoch")

    # 가로축을 Epoch, 세로축을 Validation loss로 그려줘, 실험 요약에는 가장 낮았던 값을 남겨줘.”
    run.define_metric("Validation loss", step_metric="Epoch", summary="min")
    run.define_metric("Training accuracy (%)", step_metric="Epoch")
    run.define_metric("Validation accuracy (%)", step_metric="Epoch", summary="max")
    run.define_metric("Training F1", step_metric="Epoch")
    run.define_metric("Validation F1", step_metric="Epoch", summary="max")

    train_data_loader, validation_data_loader = get_data(run)

    linear_model, optimizer = get_model_and_optimizer(run)

    result = training_loop(
      model=linear_model,
      optimizer=optimizer,
      train_data_loader=train_data_loader,
      validation_data_loader=validation_data_loader,
      run=run,
      checkpoint_dir=output_dir / "checkpoints",
      run_time_str=run_time_str,
    )
    history = result.pop("history")
    history_path = output_dir / f"{run_time_str}.csv"
    with history_path.open("w", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(history[0]))
      writer.writeheader()
      writer.writerows(history)
    result.update(config)
    result.update({
      "run_id": run_time_str, "group": getattr(args, "group", "hw2"),
      "wandb_url": run.url if args.wandb else None,
      "history_path": str(history_path.resolve()),
      "hidden_sizes": config["n_hidden_unit_list"], "max_epochs": args.epochs,
    })
    # 프로젝트를 이동해도 결과 파일과 체크포인트의 연결을 유지한다.
    for key in ("history_path", "checkpoint_path"):
      path = Path(result[key])
      if Path(BASE_PATH) in path.parents:
        result[key] = str(path.relative_to(BASE_PATH))
    run.summary.update({key: result[key] for key in (
      "best_epoch", "stopped_epoch", "best_validation_loss",
      "best_validation_accuracy", "best_validation_f1",
    )})
    metadata_path = output_dir / f"{run_time_str}.json"
    metadata_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("RESULT", metadata_path, flush=True)
    return result


# https://docs.wandb.ai/models/track/config
def build_parser():
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

  parser.add_argument(
    "--early_stop_patience", type=int, default=100,
    help="Epochs to wait without validation loss improvement (default: 100)"
  )

  parser.add_argument(
    "--early_stop_delta", type=float, default=1e-5,
    help="Minimum validation loss improvement (default: 1e-5)"
  )

  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--group", default="hw2")
  parser.add_argument("--output_dir", type=Path,
                      default=Path(BASE_PATH) / "_03_homeworks/homework_2/results")
  return parser


if __name__ == "__main__":
  main(build_parser().parse_args())
