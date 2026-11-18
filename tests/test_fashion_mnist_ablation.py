"""Offline regression checks for the simplified Fashion MNIST training script."""

from contextlib import redirect_stdout
import gc
import importlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

training = importlib.import_module('_03_homeworks.homework_3.b_fashion_mnist_train')


class FashionMNISTAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def args(self, *options):
        return training.get_parser().parse_args(list(options))

    def run_main(self, args, inspect_trainer=None):
        supplied = []
        metrics = []

        def fake_data():
            images = torch.arange(10 * 784, dtype=torch.float).reshape(10, 1, 28, 28) / (10 * 784)
            dataset = TensorDataset(images, torch.arange(10))
            train, val = random_split(dataset, [8, 2])
            supplied.extend([
                DataLoader(train, batch_size=args.batch_size, shuffle=True),
                DataLoader(val, batch_size=args.batch_size),
                nn.Sequential(training.transforms.ConvertImageDtype(torch.float),
                              training.transforms.Normalize(0.0, 0.1)),
            ])
            return supplied

        real_init = training.wandb.init
        real_loop = training.ClassificationTrainer.train_loop

        def capture_init(**kwargs):
            run = real_init(**kwargs)
            run.log = lambda data: metrics.append(data)
            return run

        def check_trainer(trainer):
            self.assertIs(trainer.train_data_loader, supplied[0])
            self.assertIs(trainer.validation_data_loader, supplied[1])
            self.assertIs(trainer.transforms, supplied[2])
            if inspect_trainer:
                inspect_trainer(trainer)
            else:
                real_loop(trainer)

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'TRAIN_CHECKPOINT_DIR': directory}), \
                patch.object(training, 'get_fashion_mnist_data', side_effect=fake_data), \
                patch.object(training.wandb, 'init', side_effect=capture_init), \
                patch.object(training.torch.cuda, 'is_available', return_value=False), \
                patch.object(training.ClassificationTrainer, 'train_loop', autospec=True,
                             side_effect=check_trainer), redirect_stdout(io.StringIO()):
            training.main(args)
            if inspect_trainer is None:
                self.assertEqual(len(list(Path(directory).rglob('*latest.pt'))), 1)
        return metrics

    def test_all_models_and_bn_dropout_combinations(self):
        expected_bn = {'cnn': 3, 'vgg': 8, 'nin': 9, 'googlenet': 57, 'resnet': 20}
        expected_dropout = {'cnn': 2, 'vgg': 2, 'nin': 1, 'googlenet': 2, 'resnet': 1}
        for name in expected_bn:
            for bn in (0, 1):
                for dropout in (False, True):
                    with self.subTest(model=name, bn=bn, dropout=dropout):
                        args = self.args('--model', name, '-n', str(bn), '-b', '2', '-e', '1',
                                         '--mean', '0.3', '--std', '0.2',
                                         '--dropout' if dropout else '--no-dropout')

                        def inspect(trainer):
                            model = trainer.model
                            self.assertEqual(sum(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
                                                 for m in model.modules()), expected_bn[name] if bn else 0)
                            self.assertEqual(sum(isinstance(m, nn.Dropout) for m in model.modules()),
                                             expected_dropout[name] if dropout else 0)
                            if name == 'cnn':
                                self.assertIsInstance(model.model[1], nn.MaxPool2d)
                                self.assertIsInstance(model.model[2], nn.BatchNorm2d if bn else nn.Identity)
                                self.assertIsInstance(model.model[6], nn.BatchNorm2d if bn else nn.Identity)
                                self.assertIsInstance(model.model[11], nn.BatchNorm1d if bn else nn.Identity)
                                self.assertEqual(model.model[10].in_features, 256)
                            if name == 'nin':
                                self.assertFalse(any(isinstance(m, nn.BatchNorm2d)
                                                     for m in model.model[7].modules()))
                            if name == 'googlenet':
                                self.assertIsInstance(trainer, training.GoogLeNetClassificationTrainer)
                                self.assertFalse(any(isinstance(m, nn.BatchNorm2d)
                                                     for m in model.aux_1.modules()))
                            model.train()
                            images, labels = next(iter(trainer.train_data_loader))
                            images = trainer.transforms(images)
                            self.assertEqual(images.shape[-1], 32 if name == 'vgg' else 28)
                            result = model(images)
                            outputs = result if isinstance(result, tuple) else (result,)
                            self.assertEqual(len(outputs), 3 if name == 'googlenet' else 1)
                            self.assertTrue(all(tuple(t.shape) == (2, 10) for t in outputs))
                            loss = sum(nn.functional.cross_entropy(t, labels) for t in outputs)
                            self.assertTrue(torch.isfinite(loss))
                            loss.backward()
                            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                                for p in model.parameters() if p.requires_grad))
                            model.eval()
                            with torch.no_grad():
                                result = model(images[:1])
                            outputs = result if isinstance(result, tuple) else (result,)
                            self.assertTrue(all(tuple(t.shape) == (1, 10) for t in outputs))

                        self.run_main(args, inspect)
                        gc.collect()

    def test_augmentation_is_per_image_and_training_only(self):
        model = nn.Linear(1, 1)
        augmentation = training.TrainingOnlyAugmentation(lambda: model.training, 28)
        images = torch.zeros(3, 1, 28, 28)
        with patch.object(augmentation, 'augmentation', side_effect=lambda x: x + 1) as augment:
            model.train()
            torch.testing.assert_close(augmentation(images), torch.ones_like(images))
            self.assertEqual(augment.call_count, 3)
            model.eval()
            self.assertIs(augmentation(images), images)
            self.assertEqual(augment.call_count, 3)
        self.assertEqual(list(augmentation.parameters()), [])

    def test_transform_order_and_normalization_switch(self):
        for name in ('cnn', 'vgg'):
            for normalize in (False, True):
                args = self.args('--model', name, '-b', '2', '--augment', '--mean', '0.3', '--std', '0.2',
                                 '--normalize' if normalize else '--no-normalize')

                def inspect(trainer):
                    trainer.model.eval()
                    pipeline = trainer.transforms
                    self.assertIsInstance(pipeline[-2], training.TrainingOnlyAugmentation)
                    self.assertIsInstance(pipeline[-1], training.transforms.Normalize if normalize else nn.Identity)
                    images = torch.ones(2, 1, 28, 28)
                    output = pipeline(images)
                    self.assertEqual(output.shape[-1], 32 if name == 'vgg' else 28)
                    torch.testing.assert_close(output, torch.full_like(output, 3.5 if normalize else 1.0))

                with self.subTest(model=name, normalize=normalize):
                    self.run_main(args, inspect)

    def test_split_and_shuffle_remain_identical_across_models(self):
        observations = []
        def inspect(trainer):
            observations.append((trainer.train_data_loader.dataset.indices,
                                 trainer.validation_data_loader.dataset.indices,
                                 torch.cat([y for _, y in trainer.train_data_loader]).tolist()))
        for name in ('cnn', 'resnet', 'cnn'):
            self.run_main(self.args('--model', name, '-b', '2', '--no-normalize'), inspect)
        self.assertEqual(observations[0], observations[1])
        self.assertEqual(observations[0], observations[2])

    def test_existing_trainers_log_four_metrics_and_save_checkpoint(self):
        for name in ('cnn', 'googlenet'):
            with self.subTest(model=name):
                metrics = self.run_main(self.args('--model', name, '-b', '2', '-e', '1',
                                                  '--mean', '0.3', '--std', '0.2', '--augment'))
                self.assertEqual(len(metrics), 1)
                self.assertTrue({'Training loss', 'Validation loss', 'Training accuracy (%)',
                                 'Validation accuracy (%)'} <= metrics[0].keys())

    def test_normalization_requires_problem_one_statistics(self):
        for options in ([], ['--mean', '0.3'], ['--mean', '0.3', '--std', '0']):
            with self.subTest(options=options), self.assertRaises(ValueError):
                training.main(self.args(*options))


if __name__ == '__main__':
    unittest.main()
