import torch
from torch import nn, optim
from datetime import datetime
import os
import wandb
from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import datasets
from torchvision.transforms import transforms, v2

BASE_PATH = Path(__file__).resolve().parents[2]
import sys
sys.path.append(str(BASE_PATH))

from _01_code._99_common_utils.utils import get_num_cpu_cores

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")
if not os.path.isdir(CHECKPOINT_FILE_PATH):
  os.makedirs(os.path.join(CURRENT_FILE_PATH, "checkpoints"), exist_ok=True)

from _01_code._09_fcn_best_practice.c_trainer import ClassificationTrainer
# from _03_homeworks.homework_3.a_fashion_mnist_data import get_fashion_mnist_data
# from _01_code._16_modern_cnns.a_arg_parser import get_parser

class TrainingOnlyAugmentation(nn.Module):
  def __init__(self, is_training):
    super().__init__()
    # A callback avoids registering the classifier as a transform submodule.
    self.is_training = is_training
    self.augmentation = v2.Compose([
      v2.RandomCrop(28, padding=2),
      v2.RandomHorizontalFlip(),
    ])

  def forward(self, images):
    # The unchanged Trainer calls model.train()/eval(), not transforms.train()/eval().
    if self.is_training():
      return torch.stack([self.augmentation(image) for image in images])
    return images

# class AugmentedDataset(Dataset):
#   def __init__(self, dataset, transform):
#       self.dataset = dataset
#       self.transform = transform

#   def __len__(self):
#       return len(self.dataset)

#   def __getitem__(self, index):
#       image, label = self.dataset[index]
#       return self.transform(image), label

def get_fashion_mnist_data(run):
  data_path = os.path.join(BASE_PATH, "_00_data", "j_fashion_mnist")
  
  f_mnist_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=v2.ToTensor())
  f_mnist_train, f_mnist_validation = random_split(f_mnist_train, [55_000, 5_000])
  
  print("Num Train Samples: ", len(f_mnist_train))
  print("Num Validation Samples: ", len(f_mnist_validation))
  print("Sample Data Shape: ", f_mnist_train[0][0].shape)  # torch.Size([1, 28, 28])
  print("Sample Data Target: ", f_mnist_train[0][1])  # 9

  num_data_loading_workers = get_num_cpu_cores()
  print("Number of Data Loading Workers:", num_data_loading_workers)

  train_data_loader = DataLoader(
    dataset=f_mnist_train, batch_size=run.config.batch_size, shuffle=True,
    pin_memory=True, num_workers=num_data_loading_workers,
  )

  validation_data_loader = DataLoader(
    dataset=f_mnist_validation, batch_size=run.config.batch_size,
    pin_memory=True, num_workers=num_data_loading_workers,
  )

  f_mnist_transforms = nn.Sequential(
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=0.2855656942074541, std=0.3527285864947437),
  )

  return train_data_loader, validation_data_loader, f_mnist_transforms

def get_fashion_mnist_test_data():
  data_path = os.path.join(BASE_PATH, "_00_data", "j_fashion_mnist")

  f_mnist_test_images = datasets.FashionMNIST(data_path, train=False, download=True)
  f_mnist_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=v2.ToTensor())

  print("Num Test Samples: ", len(f_mnist_test))
  print("Sample Shape: ", f_mnist_test[0][0].shape)  # torch.Size([1, 28, 28])

  test_data_loader = DataLoader(
    dataset=f_mnist_test, batch_size=len(f_mnist_test)
  )

  f_mnist_transforms = nn.Sequential(
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=0.2855656942074541, std=0.3527285864947437),
  )

  return f_mnist_test_images, test_data_loader, f_mnist_transforms

def get_vgg_model(num_classes=10):
  def vgg_block(num_conv_layers, out_channels, ceil_mode=False):
    layers = []

    for _ in range(num_conv_layers):
      layers.append(nn.LazyConv2d(out_channels=out_channels, kernel_size=3, padding=1))
      layers.append(nn.BatchNorm2d(out_channels))
      layers.append(nn.ReLU())

    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))

    block = nn.Sequential(*layers)
    return block

  class VGG(nn.Module):
    def __init__(self, block_info):
      super().__init__()

      conv_blocks = []
      for (num_conv_layers, out_channels, *args) in block_info:
        conv_blocks.append(vgg_block(num_conv_layers, out_channels, *args))

      self.model = nn.Sequential(
        *conv_blocks,
        nn.Flatten(),
        nn.LazyLinear(out_features=512),
        nn.ReLU(),
        #nn.Dropout(0.5),
        nn.LazyLinear(out_features=512),
        nn.ReLU(),
        #nn.Dropout(0.5),
        nn.LazyLinear(out_features=num_classes)
      )

    def forward(self, x):
      x = self.model(x)
      return x

  my_model = VGG(
    block_info=((1, 64), (1, 128), (2, 256, True), (2, 512), (2, 512))
  )

  return my_model


def main():
  run_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

  config = {
    'wandb': True,
    'epochs': 500,
    'batch_size': 256,
    'validation_intervals': 10,
    'learning_rate': 3e-4,
    'early_stop_patience': 10,
    'early_stop_delta': 1e-5,
    'weight_decay': 1e-4,
    'seed': 42
  }

  project_name = "modern_fashion_mnist"
  name = "vgg_{0}".format(run_time_str)
  with wandb.init(
    mode="online" if config['wandb'] else "disabled",
    project=project_name,
    notes="fashion_mnist experiment with vgg",
    tags=["vgg", "fashion_mnist"],
    name=name,
    config=config
  ) as run:
    print(run.config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device {device}.")

    torch.manual_seed(42)
    train_data_loader, validation_data_loader, fashion_mnist_transforms = get_fashion_mnist_data(run)
    
    torch.manual_seed(run.config.seed)

    model = get_vgg_model(num_classes=10)
    model.to(device)
    
    # Add training-only augmentation
    fashion_mnist_transforms.insert(-1, TrainingOnlyAugmentation(lambda: model.training))

    from torchinfo import summary
    summary(
      model=model, input_size=(1, 1, 28, 28),
      col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"]
    )

    optimizer = optim.Adam(model.parameters(), lr=run.config.learning_rate, weight_decay=run.config.weight_decay)

    classification_trainer = ClassificationTrainer(
      project_name + "_vgg", model, optimizer, train_data_loader,
      validation_data_loader, fashion_mnist_transforms,
      run_time_str, run, device, CHECKPOINT_FILE_PATH
    )
    
    torch.manual_seed(run.config.seed)
    classification_trainer.train_loop()


if __name__ == "__main__":
  main()
