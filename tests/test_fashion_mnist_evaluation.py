"""Behavioral checks before using the frozen checkpoint on the test set."""
import importlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class EvaluationTests(unittest.TestCase):
    def evaluation(self):
        name = '_03_homeworks.homework_3.c_fashion_mnist_test'
        self.assertIsNotNone(importlib.util.find_spec(name), 'Evaluation module is not implemented yet')
        return importlib.import_module(name)

    def test_shared_factory_loads_existing_bn_vgg_without_missing_keys(self):
        training = importlib.import_module('_03_homeworks.homework_3.b_fashion_mnist_train')
        self.assertTrue(hasattr(training, 'get_fashion_mnist_model'), 'Shared model factory is missing')
        import json
        selected = json.loads(Path('_03_homeworks/homework_3/experiments/final_selection.json').read_text())
        result = selected['winner']['seed42_configuration']
        model = training.get_fashion_mnist_model(SimpleNamespace(**result['args']))
        state = torch.load(result['checkpoint'], map_location='cpu', weights_only=True)
        restored = model.load_state_dict(state, strict=True)
        self.assertEqual(restored.missing_keys, [])
        self.assertEqual(restored.unexpected_keys, [])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    def test_predict_splits_last_batch_and_preserves_order(self):
        evaluation = self.evaluation()
        logits = torch.full((5, 10), -10.)
        logits[torch.arange(5), torch.tensor([0, 0, 1, 2, 2])] = 10
        labels = torch.tensor([0, 1, 1, 2, 3])
        class RecordingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.sizes = []
            def forward(self, x):
                self.sizes.append(len(x))
                self.test_case.assertFalse(self.training)
                self.test_case.assertFalse(torch.is_grad_enabled())
                return x
        model = RecordingModel()
        model.test_case = self
        targets, probabilities = evaluation.predict(model, DataLoader(TensorDataset(logits, labels), batch_size=5),
                                                     nn.Identity(), torch.device('cpu'), 2)
        self.assertEqual(model.sizes, [2, 2, 1])
        torch.testing.assert_close(targets, labels)
        torch.testing.assert_close(probabilities, logits.softmax(1))
        metrics = evaluation.summarize_predictions(targets, probabilities)
        self.assertEqual(metrics['num_samples'], 5)
        self.assertEqual(metrics['num_correct'], 3)
        self.assertEqual(metrics['accuracy'], 60.)
        self.assertEqual(np.trace(metrics['confusion_matrix']), 3)
        self.assertEqual(np.sum(metrics['confusion_matrix']), 5)
        self.assertEqual(metrics['per_class'][0]['precision'], .5)
        self.assertEqual(metrics['per_class'][0]['recall'], 1.)
        self.assertEqual(metrics['per_class'][1]['precision'], 1.)
        self.assertEqual(metrics['per_class'][1]['recall'], .5)
        self.assertEqual(metrics['per_class'][4]['support'], 0)
        self.assertEqual(metrics['per_class'][4]['precision'], 0)
        self.assertEqual(metrics['per_class'][4]['recall'], 0)

    def test_main_head_selected_for_googlenet_outputs(self):
        evaluation = self.evaluation()
        class ThreeHeads(nn.Module):
            def forward(self, x):
                return x, -x, -x
        logits = torch.zeros(3, 10)
        logits[:, 2] = 5
        targets, probabilities = evaluation.predict(ThreeHeads(), DataLoader(TensorDataset(logits, torch.tensor([2, 2, 2])), batch_size=3),
                                                     nn.Identity(), torch.device('cpu'), 2)
        self.assertEqual(evaluation.summarize_predictions(targets, probabilities)['accuracy'], 100.)

    def test_inference_keeps_batch_norm_running_statistics(self):
        evaluation = self.evaluation()
        model = nn.Sequential(nn.BatchNorm1d(10), nn.Dropout(.5))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        images = torch.randn(7, 10)
        loader = DataLoader(TensorDataset(images, torch.arange(7)), batch_size=7)
        _, split = evaluation.predict(model, loader, nn.Identity(), torch.device('cpu'), 3)
        _, whole = evaluation.predict(model, loader, nn.Identity(), torch.device('cpu'), 7)
        torch.testing.assert_close(split, whole)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_margin_tie_chooses_first_index_and_samples_are_fixed(self):
        evaluation = self.evaluation()
        probabilities = torch.tensor([[.45, .44] + [.11/8]*8, [.51, .49] + [0.]*8,
                                      [.45, .44] + [.11/8]*8])
        metrics = evaluation.summarize_predictions(torch.tensor([0, 1, 0]), probabilities)
        self.assertEqual(metrics['ambiguous_index'], 0)
        self.assertAlmostEqual(metrics['minimum_margin'], .01, places=6)
        selected = evaluation.select_sample_indices(10000, 42)
        np.testing.assert_array_equal(selected, evaluation.select_sample_indices(10000, 42))
        self.assertEqual(len(set(selected)), 10)
        self.assertTrue(all(0 <= i < 10000 for i in selected))

    def test_rejects_bad_inference_batch_and_invalid_probabilities(self):
        evaluation = self.evaluation()
        loader = DataLoader(TensorDataset(torch.zeros(1, 10), torch.zeros(1, dtype=torch.long)))
        with self.assertRaises(ValueError):
            evaluation.predict(nn.Identity(), loader, nn.Identity(), torch.device('cpu'), 0)
        with self.assertRaises(ValueError):
            evaluation.summarize_predictions(torch.tensor([0]), torch.full((1, 10), float('nan')))
        with self.assertRaises(ValueError):
            evaluation.summarize_predictions(torch.tensor([0]), torch.ones(1, 10))

    def test_metrics_accept_float64_probabilities_loaded_from_csv(self):
        evaluation = self.evaluation()
        probabilities = np.zeros((2, 10), dtype=np.float64)
        probabilities[0, 0], probabilities[1, 1] = 1., 1.
        metrics = evaluation.summarize_predictions(torch.tensor([0, 2]), probabilities)
        self.assertEqual(metrics['accuracy'], 50.)
        self.assertEqual(metrics['num_correct'], 1)


if __name__ == '__main__':
    unittest.main()
