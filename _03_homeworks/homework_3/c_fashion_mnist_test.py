"""Evaluate a validation-selected Fashion MNIST checkpoint (HW problems 3 and 4)."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import textwrap
import time
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
import wandb

BASE_PATH = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_PATH))
CURRENT_PATH = Path(__file__).resolve().parent

from _03_homeworks.homework_3.a_fashion_mnist_data import (
  get_fashion_mnist_data, get_fashion_mnist_test_data,
)
from _03_homeworks.homework_3.b_fashion_mnist_train import (
  get_fashion_mnist_model, configure_fashion_mnist_transforms,
)

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def get_parser():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--result', type=Path, help='Training result.json; default: selected seed-42 run')
  parser.add_argument('--batch_size', type=int, default=256, help='Inference batch size')
  parser.add_argument('--sample_seed', type=int, default=42)
  parser.add_argument('--output_dir', type=Path, default=CURRENT_PATH / 'evaluation')
  parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)
  return parser


def predict(model, test_data_loader, image_transforms, device, batch_size):
  if batch_size <= 0:
    raise ValueError('Inference batch_size must be positive')
  model.eval()
  targets, probabilities = [], []
  with torch.inference_mode():
    for images, labels in test_data_loader:
      # The supplied test loader returns 10,000 CPU images. Only move one
      # inference chunk to the GPU, preserving the original loader and order.
      for start in range(0, len(images), batch_size):
        inputs = image_transforms(images[start:start + batch_size].to(device))
        logits = model(inputs)
        if isinstance(logits, tuple):
          logits = logits[0]  # GoogLeNet main classifier, excluding auxiliary heads.
        if logits.ndim != 2 or logits.shape != (len(inputs), 10) or not torch.isfinite(logits).all():
          raise ValueError('Expected finite logits with shape (batch, 10)')
        targets.append(labels[start:start + batch_size].cpu())
        probabilities.append(logits.softmax(dim=1).cpu())
  if not targets:
    raise ValueError('Cannot evaluate an empty dataset')
  return torch.cat(targets), torch.cat(probabilities)


def summarize_predictions(targets, probabilities):
  targets = torch.as_tensor(targets, dtype=torch.long).cpu()
  probabilities = torch.as_tensor(probabilities).cpu()
  if (targets.ndim != 1 or len(targets) == 0 or probabilities.shape != (len(targets), 10)
      or not torch.isfinite(probabilities).all() or torch.any(probabilities < 0)
      or not torch.allclose(probabilities.sum(1), torch.ones_like(probabilities[:, 0]), atol=1e-5)
      or torch.any(targets < 0) or torch.any(targets > 9)):
    raise ValueError('Expected labels 0..9 and finite normalized probabilities (N,10)')
  predicted = probabilities.argmax(1)
  correct = predicted.eq(targets)
  confidence = probabilities.max(1).values
  top_two = probabilities.topk(2, dim=1)
  margins = top_two.values[:, 0] - top_two.values[:, 1]
  matrix = confusion_matrix(targets.numpy(), predicted.numpy(), labels=range(10))
  report = classification_report(targets.numpy(), predicted.numpy(), labels=list(range(10)),
                                 target_names=CLASS_NAMES, output_dict=True, zero_division=0)
  classes = [dict(label=i, name=name, **report[name]) for i, name in enumerate(CLASS_NAMES)]
  off_diagonal = matrix.copy()
  np.fill_diagonal(off_diagonal, 0)
  directed = np.unravel_index(off_diagonal.argmax(), off_diagonal.shape)
  paired = np.triu(off_diagonal + off_diagonal.T, k=1)
  pair = np.unravel_index(paired.argmax(), paired.shape)
  ambiguous = int(margins.argmin())  # First index wins when margins tie.
  groups = {}
  for high, confidence_name in [(True, 'high'), (False, 'low')]:
    for right, outcome_name in [(True, 'correct'), (False, 'incorrect')]:
      groups[confidence_name + '_' + outcome_name] = int(((confidence >= .9).eq(high) & correct.eq(right)).sum())
  return dict(
    accuracy=100.0 * int(correct.sum()) / len(targets), num_samples=len(targets),
    num_correct=int(correct.sum()), num_incorrect=int((~correct).sum()),
    confusion_matrix=matrix.tolist(), per_class=classes,
    macro_average=report['macro avg'], weighted_average=report['weighted avg'],
    largest_directed_confusion=dict(true_label=int(directed[0]), predicted_label=int(directed[1]),
                                    count=int(off_diagonal[directed])),
    largest_confusion_pair=dict(labels=[int(pair[0]), int(pair[1])], count=int(paired[pair])),
    ambiguous_index=ambiguous, minimum_margin=float(margins[ambiguous]),
    ambiguity_definition='minimum top1-top2 softmax probability margin; index tie-break',
    confidence_threshold=.9, confidence_groups=groups,
  )


def select_sample_indices(num_samples, seed):
  if num_samples < 10:
    raise ValueError('Need at least 10 samples for problem 4')
  return np.random.default_rng(seed).choice(num_samples, size=10, replace=False).tolist()


def plot_samples(dataset, indices, targets, probabilities, path, heading):
  columns = min(5, len(indices))
  rows = math.ceil(len(indices) / columns)
  figure, axes = plt.subplots(rows, columns, figsize=(max(4.5, 3 * columns), 4 * rows),
                              squeeze=False, constrained_layout=True)
  for axis in axes.flat:
    axis.axis('off')
  details = []
  for axis, index in zip(axes.flat, indices):
    image, label = dataset[index]
    values, labels = probabilities[index].topk(2)
    predicted, confidence = int(labels[0]), float(values[0])
    correct = predicted == int(label)
    axis.imshow(np.asarray(image), cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    axis.set_title(f'#{index} | {"correct" if correct else "incorrect"}\n'
                   f'True: {CLASS_NAMES[int(label)]}\nPred: {CLASS_NAMES[predicted]} ({confidence:.1%})',
                   fontsize=9, color='darkgreen' if correct else 'darkred')
    details.append(dict(index=index, label=int(label), label_name=CLASS_NAMES[int(label)],
                        prediction=predicted, prediction_name=CLASS_NAMES[predicted], correct=correct,
                        confidence=confidence, confidence_level='high' if confidence >= .9 else 'low',
                        second_label=int(labels[1]), second_name=CLASS_NAMES[int(labels[1])],
                        second_probability=float(values[1]), margin=float(values[0] - values[1])))
  figure.suptitle(textwrap.fill(heading, width=44) if columns == 1 else heading, fontsize=11)
  figure.savefig(path, dpi=170)
  plt.close(figure)
  return details


def save_results(output_dir, dataset, targets, probabilities, metrics, sample_indices):
  output_dir.mkdir(parents=True, exist_ok=True)
  predictions = probabilities.argmax(1)
  top_two = probabilities.topk(2, dim=1).values
  with (output_dir / 'predictions.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['index', 'label', 'prediction', 'correct', 'confidence', 'margin'] + [f'p_{i}' for i in range(10)])
    for index in range(len(targets)):
      writer.writerow([index, int(targets[index]), int(predictions[index]),
                       bool(targets[index] == predictions[index]), float(top_two[index, 0]),
                       float(top_two[index, 0] - top_two[index, 1])] + probabilities[index].tolist())
  with (output_dir / 'class_metrics.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['label', 'name', 'precision', 'recall', 'f1-score', 'support'])
    writer.writeheader()
    writer.writerows(metrics['per_class'])
  matrix = np.array(metrics['confusion_matrix'])
  for normalized in [False, True]:
    shown = matrix / np.maximum(matrix.sum(1, keepdims=True), 1) if normalized else matrix
    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(shown, cmap='Blues', vmin=0, vmax=1 if normalized else None)
    figure.colorbar(image, ax=axis, fraction=.046, pad=.04)
    axis.set(xticks=range(10), yticks=range(10), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
             xlabel='Predicted class', ylabel='True class',
             title='Confusion matrix: row proportions' if normalized else 'Confusion matrix: sample counts')
    plt.setp(axis.get_xticklabels(), rotation=45, ha='right')
    for row in range(10):
      for column in range(10):
        text = f'{shown[row, column]:.1%}' if normalized else str(matrix[row, column])
        axis.text(column, row, text, ha='center', va='center', fontsize=8,
                  color='white' if shown[row, column] > shown.max() / 2 else 'black')
    figure.tight_layout()
    figure.savefig(output_dir / ('confusion_matrix_normalized.png' if normalized else 'confusion_matrix_counts.png'), dpi=170)
    plt.close(figure)
  metrics['random_samples'] = plot_samples(dataset, sample_indices, targets, probabilities,
                                           output_dir / 'random_samples.png',
                                           f"Ten random test samples (seed {metrics['sample_seed']})")
  metrics['ambiguous_sample'] = plot_samples(dataset, [metrics['ambiguous_index']], targets, probabilities,
                                            output_dir / 'most_ambiguous.png', 'Minimum top-1 / top-2 probability margin')[0]
  weakest = min(metrics['per_class'], key=lambda c: c['recall'])['label']
  errors = torch.where((targets == weakest) & (predictions != targets))[0].tolist()[:10]
  metrics['lowest_recall_error_samples'] = plot_samples(dataset, errors, targets, probabilities,
      output_dir / 'lowest_recall_errors.png', f'Errors for lowest-recall class: {CLASS_NAMES[weakest]}') if errors else []


def main(args):
  if args.batch_size <= 0:
    raise ValueError('Inference batch_size must be positive')
  torch.set_num_threads(4)
  if args.result is None:
    selection = json.loads((CURRENT_PATH / 'experiments/final_selection.json').read_text())
    result = selection['winner']['seed42_configuration']
    source_result = Path(result['checkpoint']).parent / 'result.json'
  else:
    source_result = args.result.resolve()
    result = json.loads(source_result.read_text())
  model_args = SimpleNamespace(**result['args'])
  checkpoint = Path(result['checkpoint'])
  checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  torch.manual_seed(model_args.seed)
  model = get_fashion_mnist_model(model_args)
  model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
  model.to(device).eval()
  bn_before = {key: value.cpu().clone() for key, value in model.state_dict().items()
               if 'running_' in key or 'num_batches_tracked' in key}
  run_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')
  config = dict(result['args'], evaluation_batch_size=args.batch_size, sample_seed=args.sample_seed,
                checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash, source_result=str(source_result))
  args.output_dir.mkdir(parents=True, exist_ok=True)
  with wandb.init(mode='online' if args.wandb else 'disabled', project='cnn_fashion_mnist', job_type='test',
                   name=f'test_{model_args.model}_seed{model_args.seed}_{run_time_str}', config=config) as run:
    # Verify restoration on the same validation split before reading test data.
    torch.manual_seed(model_args.split_seed)
    _, validation_loader, image_transforms = get_fashion_mnist_data()
    configure_fashion_mnist_transforms(image_transforms, model_args, model)
    validation_targets, validation_probabilities = predict(model, validation_loader, image_transforms, device, args.batch_size)
    validation_accuracy = 100.0 * int(validation_probabilities.argmax(1).eq(validation_targets).sum()) / len(validation_targets)
    if not math.isclose(validation_accuracy, result['checkpoint_validation_accuracy'], abs_tol=1e-6):
      raise RuntimeError(f'Restore mismatch: expected {result["checkpoint_validation_accuracy"]}%, got {validation_accuracy}%')
    print(f'Restored validation accuracy: {validation_accuracy:.2f}% ({len(validation_targets)} samples)', flush=True)

    test_images, test_data_loader, test_transforms = get_fashion_mnist_test_data()
    if len(test_data_loader.dataset) != 10000:
      raise RuntimeError('Expected 10,000 test samples')
    configure_fashion_mnist_transforms(test_transforms, model_args, model)
    sample_indices = select_sample_indices(10000, args.sample_seed)
    started = time.perf_counter()
    targets, probabilities = predict(model, test_data_loader, test_transforms, device, args.batch_size)
    inference_seconds = time.perf_counter() - started
    metrics = summarize_predictions(targets, probabilities)
    if metrics['num_samples'] != 10000:
      raise RuntimeError('Incomplete test inference')
    for key, before in bn_before.items():
      torch.testing.assert_close(model.state_dict()[key].cpu(), before, rtol=0, atol=0)
    metrics.update(model=model_args.model, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                   training_args=result['args'], source_result=str(source_result), device=str(device),
                   evaluation_batch_size=args.batch_size, sample_seed=args.sample_seed,
                   validation_restore_accuracy=validation_accuracy, inference_seconds=inference_seconds,
                   test_target_met=metrics['accuracy'] >= 93, wandb_url=run.url, wandb_id=run.id,
                   completed_at=datetime.now().astimezone().isoformat(), bn_statistics_unchanged=True)
    save_results(args.output_dir, test_images, targets, probabilities, metrics, sample_indices)
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    run.log({'Test accuracy (%)': metrics['accuracy'], 'Validation restore accuracy (%)': validation_accuracy,
             'Confusion matrix': wandb.Image(str(args.output_dir / 'confusion_matrix_counts.png')),
             'Class metrics': wandb.Table(columns=['Class', 'Precision', 'Recall', 'F1', 'Support'],
                 data=[[c['name'], c['precision'], c['recall'], c['f1-score'], c['support']] for c in metrics['per_class']]),
             'Random samples': wandb.Image(str(args.output_dir / 'random_samples.png')),
             'Most ambiguous': wandb.Image(str(args.output_dir / 'most_ambiguous.png'))})
    print(f'Test accuracy: {metrics["accuracy"]:.2f}% ({metrics["num_correct"]}/10000)', flush=True)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
  return metrics


if __name__ == '__main__':
  args = get_parser().parse_args()
  main(args)
