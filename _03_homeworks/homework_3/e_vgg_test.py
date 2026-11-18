"""Evaluate the final Fashion MNIST VGG pad model or a saved ablation run."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import wandb

BASE_PATH = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_PATH))
HOMEWORK_PATH = Path(__file__).resolve().parent
DEFAULT_RESULT = HOMEWORK_PATH / "experiments/input_normalization_vgg_pad/normalize_on/result.json"

from _03_homeworks.homework_3.a_fashion_mnist_data import get_fashion_mnist_test_data
from _03_homeworks.homework_3.c_fashion_mnist_test import (
  predict, save_results, select_sample_indices, summarize_predictions,
)
from _03_homeworks.homework_3.d_vgg_train import (
  PROJECT_NAME, configure_fashion_mnist_transforms, get_fashion_mnist_model,
)


def get_parser():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--result", type=Path, default=DEFAULT_RESULT,
                      help="Training result.json; defaults to the final standardized VGG pad model")
  parser.add_argument("--output_dir", type=Path, help="Folder for test plots, predictions and metrics")
  parser.add_argument("--batch_size", type=int, default=256)
  parser.add_argument("--sample_seed", type=int, default=42)
  parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
  return parser


def main(args):
  result_path = args.result.resolve()
  result = json.loads(result_path.read_text())
  settings = SimpleNamespace(**result["args"])
  if settings.model != "vgg" or settings.vgg_variant != "pad":
    raise ValueError("The result must describe a Fashion MNIST VGG pad checkpoint")

  checkpoint = Path(result["checkpoint"])
  model = get_fashion_mnist_model(settings)
  model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device).eval()

  images, test_loader, image_transforms = get_fashion_mnist_test_data()
  configure_fashion_mnist_transforms(image_transforms, settings, model)
  labels, probabilities = predict(model, test_loader, image_transforms, device, args.batch_size)
  metrics = summarize_predictions(labels, probabilities)
  metrics.update(sample_seed=args.sample_seed, checkpoint=str(checkpoint), training_args=result["args"],
                 source_result=str(result_path))
  sample_indices = select_sample_indices(len(labels), args.sample_seed)

  run_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S-%f")
  run_name = f"test_vgg_pad_{result_path.parent.name}_{run_time}"
  output_dir = args.output_dir or HOMEWORK_PATH / "evaluation" / run_name
  save_results(output_dir, images, labels, probabilities, metrics, sample_indices)

  if args.wandb:
    with wandb.init(project=PROJECT_NAME, job_type="test", name=run_name, config=result["args"]) as run:
      run.log({
        "Test accuracy (%)": metrics["accuracy"],
        "Confusion matrix": wandb.Image(str(output_dir / "confusion_matrix_counts.png")),
        "Random samples": wandb.Image(str(output_dir / "random_samples.png")),
        "Most ambiguous": wandb.Image(str(output_dir / "most_ambiguous.png")),
      })
      metrics["wandb_url"] = run.url

  (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
  print(f"Test accuracy: {metrics['accuracy']:.2f}% ({metrics['num_correct']}/{metrics['num_samples']})")
  print(f"Evaluation results: {output_dir.resolve()}")


if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()
  if args.batch_size <= 0:
    parser.error("batch size must be positive")
  main(args)
