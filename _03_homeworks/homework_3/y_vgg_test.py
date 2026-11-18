import numpy as np
import torch
import os

from matplotlib import pyplot as plt
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[2]
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")

import sys
sys.path.append(str(BASE_PATH))

from _01_code._09_fcn_best_practice.d_tester import ClassificationTester
from _03_homeworks.homework_3.x_vgg_train import get_fashion_mnist_test_data
from _03_homeworks.homework_3.x_vgg_train import get_vgg_model

import torchvision

def main():
  test_model = get_vgg_model(num_classes=10)
  fashion_mnist_test_images, test_data_loader, fashion_mnist_transforms = get_fashion_mnist_test_data()

  project_name = "modern_fashion_mnist"
  classification_tester = ClassificationTester(
    project_name + "_vgg", test_model, test_data_loader,
    fashion_mnist_transforms, CHECKPOINT_FILE_PATH
  )
  classification_tester.test()

  print()

  img, label = fashion_mnist_test_images[0]
  print("     LABEL:", label)
  plt.imshow(img)
  plt.show()

  output = classification_tester.test_single(
    torch.tensor(np.array(fashion_mnist_test_images[0][0])).unsqueeze(dim=0).unsqueeze(dim=0)
  )
  print("PREDICTION:", output)


if __name__ == "__main__":
  main()
