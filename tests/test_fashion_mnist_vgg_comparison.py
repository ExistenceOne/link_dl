"""Ensure variant comparisons use paired seeds and selected checkpoints."""
import importlib
import unittest

study = importlib.import_module('scripts.fashion_mnist_vgg_comparison')


class VGGComparisonTests(unittest.TestCase):
    def fixture(self):
        results = []
        for variant, scores in [('resize', [90., 92., 94.]), ('pad', [91., 93., 95.]),
                                ('no_last_pool', [89., 91., 93.])]:
            for seed, score in zip((42, 43, 44), scores):
                results.append(dict(
                    args=dict(vgg_variant=variant, seed=seed),
                    checkpoint_validation_accuracy=score, best_logged_validation_accuracy=100.,
                    checkpoint_validation_loss=.2, elapsed_seconds=60., epochs=10,
                    peak_gpu_memory_mb=100., parameter_count=123,
                    initial_state_sha256=str(seed), shuffle_state_sha256=str(seed),
                    training_indices_sha256='train', validation_indices_sha256='val',
                    status='completed'))
        return results

    def test_summary_uses_checkpoint_accuracy_and_sample_std(self):
        summary = study.summarize(self.fixture())
        resize = summary['variants']['resize']
        self.assertEqual(resize['accuracy_mean'], 92.)
        self.assertEqual(resize['accuracy_std'], 2.)
        self.assertEqual(summary['variants']['pad']['paired_delta_pp'], [1., 1., 1.])
        self.assertEqual(summary['variants']['no_last_pool']['paired_delta_pp'], [-1., -1., -1.])
        self.assertEqual(summary['winner'], 'pad')

    def test_rejects_unpaired_initialization_and_incomplete_runs(self):
        for field in ('initial_state_sha256', 'shuffle_state_sha256', 'training_indices_sha256',
                      'validation_indices_sha256', 'parameter_count', 'status'):
            with self.subTest(field=field):
                results = self.fixture()
                results[-1][field] = 'different'
                with self.assertRaises(ValueError):
                    study.summarize(results)
        with self.assertRaises(ValueError):
            study.summarize(self.fixture()[:-1])

    def test_configs_are_nine_fresh_fixed_condition_runs(self):
        configurations = study.configurations()
        self.assertEqual(len(configurations), 9)
        self.assertEqual(len({c['id'] for c in configurations}), 9)
        training = importlib.import_module('_03_homeworks.homework_3.b_fashion_mnist_train')
        parsed = [vars(training.get_parser().parse_args(c['args'])) for c in configurations]
        reference = {k: v for k, v in parsed[0].items() if k not in ('vgg_variant', 'seed')}
        for args in parsed:
            self.assertEqual({k: v for k, v in args.items() if k not in ('vgg_variant', 'seed')}, reference)
        self.assertEqual(reference['batch_size'], 256)
        self.assertEqual(reference['learning_rate'], .0003)
        self.assertEqual(reference['weight_decay'], .0001)
        self.assertEqual(reference['split_seed'], 42)
        self.assertTrue(reference['augment'])
        self.assertFalse(reference['dropout'])
