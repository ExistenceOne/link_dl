import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import sys

import torch
from torch import nn, optim
from torchvision import transforms
import wandb

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(BASE_PATH)

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")

from _01_code._09_fcn_best_practice.c_trainer import ClassificationTrainer
from _01_code._14_normalization.a_cifar10_train_cnn_with_normalization import (
  get_cnn_model_with_dropout_and_batch_normalization,
)
from _01_code._15_data_augmentation.a_arg_parser import get_parser as get_base_parser
from _01_code._16_modern_cnns._01_vgg.a_cifar10_train_vgg import get_vgg_model
from _01_code._16_modern_cnns._02_nin.a_cifar10_train_nin import get_nin_model
from _01_code._16_modern_cnns._03_google_net.a_cifar10_train_google_net import get_google_net_model
from _01_code._16_modern_cnns._03_google_net.b_google_net_trainer import GoogLeNetClassificationTrainer
from _01_code._16_modern_cnns._04_resnet.a_cifar10_train_resnet import get_resnet_model
from _03_homeworks.homework_3.a_fashion_mnist_data import get_fashion_mnist_data


def get_parser():
  parser = get_base_parser()
  parser.description = "Fashion MNIST training and ablation with existing link_dl models"
  parser.set_defaults(batch_size=128, epochs=100, validation_intervals=1,
                      optimizer=3, normalization=1, dropout=True)
  # The inherited help strings contain old defaults; match this script's settings.
  for action in parser._actions:
    if action.dest in ("batch_size", "epochs", "validation_intervals"):
      action.help = action.dest.replace("_", " ") + " (default: %(default)s)"
    elif action.dest == "optimizer":
      action.help = "0: SGD, 1: Momentum, 2: RMSprop, 3: Adam (default: %(default)s)"
    elif action.dest == "normalization":
      action.choices = [0, 1]
      action.help = "Model normalization: 0: none, 1: BatchNorm (default: %(default)s)"
  parser.add_argument("--model", choices=["cnn", "vgg", "nin", "googlenet", "resnet"], default="cnn")
  parser.add_argument("--vgg_variant", choices=["resize", "pad", "no_last_pool"], default="resize",
                      help="VGG: resize to 32, zero-pad to 32, or keep 28 and remove the last pool")
  parser.add_argument("--dropout_rate", type=float, default=0.5)
  parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                      help="Input normalization; separate from model BatchNorm")
  parser.add_argument("--mean", type=float, help="Mean calculated in problem 1")
  parser.add_argument("--std", type=float, help="Std calculated in problem 1")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--split_seed", type=int, default=42)
  return parser


def add_conv_batch_norm(module):
  # Used only for feature blocks that originally have no BN.
  for name, child in list(module.named_children()):
    if isinstance(child, nn.Conv2d):
      setattr(module, name, nn.Sequential(child, nn.BatchNorm2d(child.out_channels)))
    else:
      add_conv_batch_norm(child)


def configure_existing_layers(module, args):
  for name, child in list(module.named_children()):
    if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LazyBatchNorm2d)):
      if args.normalization == 0:
        setattr(module, name, nn.Identity())
    elif isinstance(child, nn.Dropout):
      setattr(module, name, nn.Dropout(args.dropout_rate) if args.dropout else nn.Identity())
    else:
      configure_existing_layers(child, args)


def get_fashion_mnist_model(args):
  if args.model == "cnn":
    model = get_cnn_model_with_dropout_and_batch_normalization()
    model.model[0] = nn.Conv2d(1, 6, kernel_size=5)
    model.model[10] = nn.Linear(256, 128)
    # Preserve Conv -> Pool -> BN -> ReLU and the hidden FC's BatchNorm1d.
  elif args.model == "vgg":
    model = get_vgg_model(num_classes=10)
    if getattr(args, "vgg_variant", "resize") == "no_last_pool":
      model.model[4][-1] = nn.Identity()
    if args.normalization == 1:
      for block in list(model.model.children())[:5]:
        add_conv_batch_norm(block)
  elif args.model == "nin":
    model = get_nin_model()
    if args.normalization == 1:
      for index in (0, 2, 4):  # Feature blocks only, not the class-score block.
        add_conv_batch_norm(model.model[index])
  elif args.model == "googlenet":
    model = get_google_net_model(num_classes=10)
    model.inception_block_3.add_module("classifier", nn.Linear(1024, 10))
    if args.normalization == 1:
      for name in ("conv_block", "inception_block_1", "inception_block_2", "inception_block_3"):
        add_conv_batch_norm(getattr(model, name))
    # Existing Dropout is in the two auxiliary classifiers only.
  elif args.model == "resnet":
    model = get_resnet_model(num_classes=10)
    model.fc = nn.Sequential(
      nn.Dropout(args.dropout_rate) if args.dropout else nn.Identity(), model.fc,
    )
  else:
    raise ValueError(f"Unknown model: {args.model}")
  configure_existing_layers(model, args)
  return model


class TrainingOnlyAugmentation(nn.Module):
  def __init__(self, is_training, image_size):
    super().__init__()
    # A callback avoids registering the classifier as a transform submodule.
    self.is_training = is_training
    self.augmentation = transforms.Compose([
      transforms.RandomCrop(image_size, padding=2),
      transforms.RandomHorizontalFlip(),
    ])

  def forward(self, images):
    # The unchanged Trainer calls model.train()/eval(), not transforms.train()/eval().
    if self.is_training():
      return torch.stack([self.augmentation(image) for image in images])
    return images


def configure_fashion_mnist_transforms(f_mnist_transforms, args, model, augment=False):
  # Extend the supplied transform object; validation/test never add augmentation.
  f_mnist_transforms[1] = (
    transforms.Normalize(mean=args.mean, std=args.std) if args.normalize else nn.Identity()
  )
  image_size = 28
  if args.model == "vgg":
    variant = getattr(args, "vgg_variant", "resize")
    if variant == "resize":
      f_mnist_transforms.insert(1, transforms.Resize((32, 32), antialias=True))
      image_size = 32
    elif variant == "pad":
      f_mnist_transforms.insert(1, transforms.Pad(2, fill=0, padding_mode="constant"))
      image_size = 32
    elif variant != "no_last_pool":
      raise ValueError(f"Unknown VGG variant: {variant}")
  if augment and args.augment:
    f_mnist_transforms.insert(-1, TrainingOnlyAugmentation(lambda: model.training, image_size))
  return f_mnist_transforms


def main(args):
  if args.normalization not in (0, 1):
    raise ValueError("--normalization supports 0 (none) or 1 (BatchNorm)")
  if args.optimizer not in (0, 1, 2, 3):
    raise ValueError("--optimizer must be 0, 1, 2, or 3")
  if not 0 <= args.dropout_rate < 1:
    raise ValueError("--dropout_rate must be in [0, 1)")
  if args.normalize and (args.mean is None or args.std is None or
                         not math.isfinite(args.mean) or not math.isfinite(args.std) or args.std <= 0):
    raise ValueError("Pass problem 1 values with --mean and --std, or use --no-normalize")
  if min(args.batch_size, args.epochs, args.validation_intervals, args.early_stop_patience) <= 0:
    raise ValueError("batch_size, epochs, validation_intervals, and early_stop_patience must be positive")

  config = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'validation_intervals': args.validation_intervals,
    'learning_rate': args.learning_rate,
    'early_stop_patience': args.early_stop_patience,
    'early_stop_delta': args.early_stop_delta,
    'weight_decay': args.weight_decay,
    'optimizer': args.optimizer,
    'model': args.model,
    'vgg_variant': args.vgg_variant,
    'normalization': args.normalization,
    'dropout': args.dropout,
    'dropout_rate': args.dropout_rate,
    'augment': args.augment,
    'normalize': args.normalize,
    'mean': args.mean,
    'std': args.std,
    'seed': args.seed,
    'split_seed': args.split_seed,
  }

  run_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')
  technique_name = (f"{args.model}_bn{args.normalization}_drop{int(args.dropout)}"
                    f"_aug{int(args.augment)}_norm{int(args.normalize)}_seed{args.seed}")
  if args.model == "vgg":
    technique_name += f"_{args.vgg_variant}"
  project_name = "cnn_fashion_mnist"
  with wandb.init(
    mode="online" if args.wandb else "disabled",
    project=project_name,
    notes="fashion mnist experiment with existing link_dl models",
    tags=["fashion_mnist", args.model],
    name=f"{technique_name}_{run_time_str}",
    config=config,
  ) as run:
    print(args)
    print(run.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device {device}.")

    torch.manual_seed(args.split_seed)
    train_data_loader, validation_data_loader, f_mnist_transforms = get_fashion_mnist_data()
    # Keep the supplied loaders. Isolate shuffle randomness from model/augmentation.
    train_data_loader.sampler.generator = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)

    model = get_fashion_mnist_model(args)
    model.to(device)

    if args.normalization == 1 and args.model in ("cnn", "googlenet", "resnet"):
      if args.batch_size == 1 or len(train_data_loader.dataset) % args.batch_size == 1:
        raise ValueError("This model's BatchNorm needs more than one sample in each training batch")

    configure_fashion_mnist_transforms(f_mnist_transforms, args, model, augment=True)
    image_size = 32 if args.model == "vgg" and args.vgg_variant != "no_last_pool" else 28

    from torchinfo import summary
    summary(model=model, input_size=(2, 1, image_size, image_size), device=device, mode="eval")

    optimizers = [
      optim.SGD(model.parameters(), lr=run.config.learning_rate, weight_decay=args.weight_decay),
      optim.SGD(model.parameters(), lr=run.config.learning_rate, momentum=0.9, weight_decay=args.weight_decay),
      optim.RMSprop(model.parameters(), lr=run.config.learning_rate, weight_decay=args.weight_decay),
      optim.Adam(model.parameters(), lr=run.config.learning_rate, weight_decay=args.weight_decay),
    ]
    print("Optimizer:", optimizers[args.optimizer])

    trainer_class = GoogLeNetClassificationTrainer if args.model == "googlenet" else ClassificationTrainer
    classification_trainer = trainer_class(
      project_name + "_" + technique_name, model, optimizers[args.optimizer],
      train_data_loader, validation_data_loader, f_mnist_transforms,
      run_time_str, run, device, CHECKPOINT_FILE_PATH,
    )
    torch.manual_seed(args.seed)
    classification_trainer.train_loop()


if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()
  main(args)
  # Supply --mean and --std calculated in problem 1 when using --normalize.
  # python _03_homeworks/homework_3/b_fashion_mnist_train.py --wandb --mean MEAN --std STD
  # Compare by adding: -n 0 / --no-dropout / --augment / --no-normalize
  # Select a model with: --model cnn|vgg|nin|googlenet|resnet
