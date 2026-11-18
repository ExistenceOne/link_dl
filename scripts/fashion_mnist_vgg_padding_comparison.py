"""Train and evaluate paired Fashion-MNIST VGG input-padding experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, stdev

import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torchvision import transforms
import wandb

from _01_code._09_fcn_best_practice.c_trainer import ClassificationTrainer
from _03_homeworks.homework_3.a_fashion_mnist_data import (
  get_fashion_mnist_data,
  get_fashion_mnist_test_data,
)
from _03_homeworks.homework_3.b_fashion_mnist_train import TrainingOnlyAugmentation
from _03_homeworks.homework_3.c_fashion_mnist_test import predict
from _03_homeworks.homework_3.d_vgg_train import (
  TRAIN_MEAN,
  TRAIN_STD,
)

SEEDS = (42, 43, 44)
INPUT_SIZES = {"preprocess_pad": 32, "first_conv_pad": 28, "third_pool_ceil": 28}
SPLIT_SEED = 42
BATCH_SIZE = 256
DROPOUT = 0.5


def make_model(condition, dropout_rate=DROPOUT):
  layers = []
  in_channels = 1
  blocks = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
  for block_number, (layer_count, out_channels) in enumerate(blocks, start=1):
    for _ in range(layer_count):
      padding = 3 if not layers and condition == "first_conv_pad" else 1
      layers.extend([
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
      ])
      in_channels = out_channels
    ceil_mode = condition == "third_pool_ceil" and block_number == 3
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))

  return nn.Sequential(
    *layers,
    nn.Flatten(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(dropout_rate) if dropout_rate else nn.Identity(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(dropout_rate) if dropout_rate else nn.Identity(),
    nn.Linear(512, 10),
  )


def make_transforms(condition, model, augment=False):
  layers = []
  if condition == "preprocess_pad":
    layers.append(transforms.Pad(2))
  if augment:
    layers.append(TrainingOnlyAugmentation(lambda: model.training, INPUT_SIZES[condition]))
  layers.append(transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD))
  return nn.Sequential(*layers)


def load_subset_tensors(subset, condition):
  indices = torch.as_tensor(subset.indices)
  pixels = subset.dataset.data[indices].unsqueeze(1)
  if condition == "preprocess_pad":
    pixels = transforms.Pad(2)(pixels)
  labels = subset.dataset.targets[indices]
  return pixels, labels


def make_concat_loaders(train_loader, validation_loader, condition, seed):
  pixels, labels = load_subset_tensors(train_loader.dataset, condition)
  augment = TrainingOnlyAugmentation(lambda: True, INPUT_SIZES[condition])
  augmented_pixels = torch.empty_like(pixels)
  with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(seed)
    for start in range(0, len(pixels), BATCH_SIZE):
      end = start + BATCH_SIZE
      augmented_pixels[start:end] = augment(pixels[start:end])

  originals = TensorDataset(pixels.float().div_(255), labels)
  augmented = TensorDataset(augmented_pixels.float().div_(255), labels)
  assert len(originals) == len(augmented) == 55_000
  combined = ConcatDataset([originals, augmented])
  validation_pixels, validation_labels = load_subset_tensors(validation_loader.dataset, condition)
  validation = TensorDataset(validation_pixels.float().div_(255), validation_labels)

  training_batches = DataLoader(
    combined, batch_size=BATCH_SIZE, shuffle=True,
    generator=torch.Generator().manual_seed(seed), pin_memory=True,
    num_workers=train_loader.num_workers,
  )
  validation_batches = DataLoader(
    validation, batch_size=BATCH_SIZE, pin_memory=True,
    num_workers=validation_loader.num_workers,
  )
  return training_batches, validation_batches


def write_json(path, payload):
  temporary = path.with_suffix(".tmp")
  temporary.write_text(json.dumps(payload, indent=2) + "\n")
  temporary.replace(path)


def train_and_evaluate(condition, seed, output, device, smoke, concat, dropout_rate):
  run_name = f"{condition}_seed{seed}"
  run_dir = output / run_name
  run_dir.mkdir(exist_ok=True)
  result_path = run_dir / "result.json"
  settings = dict(
    condition=condition,
    seed=seed,
    split_seed=SPLIT_SEED,
    batch_size=BATCH_SIZE,
    epochs=1 if smoke else 100,
    early_stop_patience=10,
    early_stop_delta=0.00001,
    validation_intervals=1,
    learning_rate=0.0003,
    weight_decay=0.0001,
    optimizer="Adam",
    batch_norm=True,
    dropout=dropout_rate,
    augment=True,
    normalize=True,
    mean=TRAIN_MEAN,
    std=TRAIN_STD,
  )
  if concat:
    settings.update(
      augmentation_mode="fixed_concat",
      augmentation_seed=seed,
      train_original_samples=55_000,
      train_augmented_samples=55_000,
    )
  if result_path.exists():
    result = json.loads(result_path.read_text())
    if result["settings"] != settings:
      raise ValueError(f"Existing result has different settings: {result_path}")
    return result

  with wandb.init(project="cnn_fashion_mnist", name=run_name, config=settings, mode="disabled") as run:
    torch.manual_seed(SPLIT_SEED)
    train_loader, validation_loader, _ = get_fashion_mnist_data()
    train_loader.num_workers = 4
    validation_loader.num_workers = 4
    train_loader.sampler.generator = torch.Generator().manual_seed(seed)

    torch.manual_seed(seed)
    model = make_model(condition, dropout_rate)
    initial_weights = hashlib.sha256()
    for parameter in model.parameters():
      initial_weights.update(parameter.detach().numpy().tobytes())
    model.to(device=device, memory_format=torch.channels_last)
    train_transforms = make_transforms(condition, model, augment=True)

    test_transforms = make_transforms(condition, model)
    if concat:
      train_loader, validation_loader = make_concat_loaders(
        train_loader, validation_loader, condition, seed
      )
      train_transforms = transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD)
      print(f"{run_name}: {len(train_loader.dataset)} training examples", flush=True)

    model.eval()
    with torch.inference_mode():
      sample = torch.zeros(2, 1, 28, 28, device=device)
      prepared_sample = test_transforms(sample)
      assert prepared_sample.shape[-2:] == (INPUT_SIZES[condition],) * 2
      first_conv_size = 28 if condition == "third_pool_ceil" else 32
      assert model[0](prepared_sample).shape == (2, 64, first_conv_size, first_conv_size)
      assert model(prepared_sample).shape == (2, 10)

    optimizer = optim.Adam(
      model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    os.environ["TRAIN_CHECKPOINT_DIR"] = str(run_dir)
    os.environ["TRAIN_METRICS_FILE"] = str(run_dir / "progress.json")
    trainer = ClassificationTrainer(
      run_name, model, optimizer, train_loader, validation_loader,
      train_transforms, "selected", run, device, str(run_dir),
    )
    print(f"Starting {run_name} on {device}", flush=True)
    trainer.train_loop()

    checkpoint = run_dir / f"{run_name}_checkpoint_selected.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    validation_loss, validation_accuracy = trainer.do_validation()
    _, test_loader, _ = get_fashion_mnist_test_data()
    targets, probabilities = predict(model, test_loader, test_transforms, device, BATCH_SIZE)
    assert len(targets) == 10_000
    correct = int(probabilities.argmax(1).eq(targets).sum())
    progress = json.loads((run_dir / "progress.json").read_text())
    result = dict(
      settings=settings,
      checkpoint=str(checkpoint),
      initial_weights_sha256=initial_weights.hexdigest(),
      final_epoch=progress["epoch"],
      validation_loss=validation_loss,
      validation_accuracy=validation_accuracy,
      test_correct=correct,
      test_samples=len(targets),
      test_accuracy=100 * correct / len(targets),
    )
    write_json(result_path, result)
    print(f"Finished {run_name}: test accuracy {result['test_accuracy']:.2f}%", flush=True)
    return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("output", type=Path)
  parser.add_argument(
    "--concat", action="store_true", help="Append one fixed augmented copy of each training image"
  )
  parser.add_argument("--condition", choices=INPUT_SIZES, help="Run one condition instead of the padding pair")
  parser.add_argument("--no-dropout", action="store_true", help="Disable classifier Dropout")
  parser.add_argument("--smoke", action="store_true")
  args = parser.parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=True)
  os.environ["TRAIN_MAX_BATCHES"] = "2" if args.smoke else "0"
  os.environ["TRAIN_REQUIRE_CUDA"] = "1"
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this experiment")
  device = torch.device("cuda:0")

  dropout_rate = 0.0 if args.no_dropout else DROPOUT
  seeds = (SEEDS[0],) if args.smoke else SEEDS
  if args.condition:
    results = [
      train_and_evaluate(args.condition, seed, output, device, args.smoke, args.concat, dropout_rate)
      for seed in seeds
    ]
    summary = {"condition": args.condition, "runs": results}
    if not args.smoke:
      accuracies = [result["test_accuracy"] for result in results]
      summary.update(mean=mean(accuracies), sample_sd=stdev(accuracies))
    write_json(output / "summary.json", summary)
    return

  pairs = []
  for seed in seeds:
    preprocess = train_and_evaluate(
      "preprocess_pad", seed, output, device, args.smoke, args.concat, dropout_rate
    )
    first_conv = train_and_evaluate(
      "first_conv_pad", seed, output, device, args.smoke, args.concat, dropout_rate
    )
    assert preprocess["initial_weights_sha256"] == first_conv["initial_weights_sha256"]
    pairs.append(dict(
      seed=seed,
      preprocess=preprocess,
      first_conv=first_conv,
      difference_pp=first_conv["test_accuracy"] - preprocess["test_accuracy"],
    ))

  summary = {"pairs": pairs}
  if not args.smoke:
    for condition in ("preprocess", "first_conv"):
      accuracies = [pair[condition]["test_accuracy"] for pair in pairs]
      summary[condition] = dict(mean=mean(accuracies), sample_sd=stdev(accuracies))
    summary["mean_difference_pp"] = mean(pair["difference_pp"] for pair in pairs)
  write_json(output / "summary.json", summary)


if __name__ == "__main__":
  main()
