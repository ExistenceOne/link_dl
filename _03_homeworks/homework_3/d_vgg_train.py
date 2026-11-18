"""Train the final Fashion MNIST VGG pad model or one ablation condition."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

import torch
from torch import nn, optim
from torchinfo import summary
from torchvision import transforms
import wandb

BASE_PATH = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_PATH))
CHECKPOINT_ROOT = Path(__file__).resolve().parent / "checkpoints"

from _01_code._09_fcn_best_practice.c_trainer import ClassificationTrainer
from _01_code._16_modern_cnns._01_vgg.a_cifar10_train_vgg import get_vgg_model
from _03_homeworks.homework_3.a_fashion_mnist_data import get_fashion_mnist_data

# Pixel statistics from the 55,000 training images with split seed 42.
TRAIN_MEAN = 0.2855656942074541
TRAIN_STD = 0.3527285864947437
IMAGE_SIZE = 32
PROJECT_NAME = "cnn_fashion_mnist"


def get_parser():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("-b", "--batch_size", type=int, default=256)
  parser.add_argument("-e", "--epochs", type=int, default=100)
  parser.add_argument("-r", "--learning_rate", type=float, default=0.0003)
  parser.add_argument("-w", "--weight_decay", type=float, default=0.0001)
  parser.add_argument("-p", "--early_stop_patience", type=int, default=10)
  parser.add_argument("-d", "--early_stop_delta", type=float, default=0.00001)
  parser.add_argument("-n", "--normalization", type=int, choices=(0, 1), default=1,
                      help="Model BatchNorm: 0 = off, 1 = on")
  parser.add_argument("--dropout", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument("--dropout_rate", type=float, default=0.5)
  parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--standardization", dest="normalize", action=argparse.BooleanOptionalAction,
                      default=True, help="Standardize input pixels using the training mean and std")
  parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument("--seed", type=int, default=42, help="Model and training shuffle seed")
  parser.set_defaults(split_seed=42, mean=TRAIN_MEAN, std=TRAIN_STD,
                      validation_intervals=1, model="vgg", vgg_variant="pad")
  return parser


def get_fashion_mnist_model(args):
  model = get_vgg_model(num_classes=10)

  if args.normalization:
    for block in model.model[:5]:
      for index, layer in enumerate(block):
        if isinstance(layer, nn.Conv2d):
          # Keep the Conv -> BN layout used by the saved VGG pad checkpoint.
          block[index] = nn.Sequential(layer, nn.BatchNorm2d(layer.out_channels))

  for index, layer in enumerate(model.model):
    if isinstance(layer, nn.Dropout):
      model.model[index] = nn.Dropout(args.dropout_rate) if args.dropout else nn.Identity()

  return model


class TrainingOnlyAugmentation(nn.Module):
  def __init__(self, is_training):
    super().__init__()
    self.is_training = is_training
    self.augmentation = transforms.Compose([
      transforms.RandomCrop(IMAGE_SIZE, padding=2),
      transforms.RandomHorizontalFlip(),
    ])

  def forward(self, images):
    # The shared trainer switches the model's mode for validation.
    if not self.is_training():
      return images
    return torch.stack([self.augmentation(image) for image in images])


def configure_fashion_mnist_transforms(image_transforms, args, model, augment=False):
  if args.normalize:
    image_transforms[1] = transforms.Normalize(mean=args.mean, std=args.std)
  else:
    image_transforms[1] = nn.Identity()

  # Zero-pad 28x28 to 32x32 before augmentation and standardization.
  image_transforms.insert(1, transforms.Pad(2))
  if augment and args.augment:
    image_transforms.insert(2, TrainingOnlyAugmentation(lambda: model.training))
  return image_transforms


def main(args):
  run_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S-%f")
  condition = (f"vgg_pad_bn{args.normalization}_drop{int(args.dropout)}"
               f"_aug{int(args.augment)}_norm{int(args.normalize)}_seed{args.seed}")
  run_name = f"{condition}_{run_time}"
  checkpoint_dir = Path(os.environ.get("TRAIN_CHECKPOINT_DIR", CHECKPOINT_ROOT / run_name)).resolve()

  with wandb.init(mode="online" if args.wandb else "disabled", project=PROJECT_NAME,
                  name=run_name, tags=["fashion_mnist", "vgg", "pad"], config=vars(args)) as run:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}: {run_name}")

    torch.manual_seed(args.split_seed)
    train_loader, validation_loader, image_transforms = get_fashion_mnist_data()
    train_loader.sampler.generator = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)

    model = get_fashion_mnist_model(args).to(device)
    configure_fashion_mnist_transforms(image_transforms, args, model, augment=True)
    summary(model, input_size=(2, 1, IMAGE_SIZE, IMAGE_SIZE), device=device, mode="eval")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    trainer = ClassificationTrainer(
      condition, model, optimizer, train_loader, validation_loader, image_transforms,
      run_time, run, device, str(checkpoint_dir),
    )
    torch.manual_seed(args.seed)
    trainer.train_loop()

    checkpoint = checkpoint_dir / f"{condition}_checkpoint_{run_time}.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    validation_loss, validation_accuracy = trainer.do_validation()
    result = {
      "args": vars(args),
      "checkpoint": str(checkpoint),
      "checkpoint_validation_loss": validation_loss,
      "checkpoint_validation_accuracy": validation_accuracy,
    }
    result_path = checkpoint_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Checkpoint validation accuracy: {validation_accuracy:.2f}%")
    print(f"Evaluation input: {result_path}")


if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()
  if min(args.batch_size, args.epochs, args.early_stop_patience) <= 0:
    parser.error("batch size, epochs and early stopping patience must be positive")
  if not 0 <= args.dropout_rate < 1:
    parser.error("dropout rate must be in [0, 1)")
  if args.normalization and (args.batch_size == 1 or 55_000 % args.batch_size == 1):
    parser.error("BatchNorm needs at least two images in each training batch")
  main(args)
