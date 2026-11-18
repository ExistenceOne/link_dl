"""Check VGG variants preserve pixels, classifier size and paired initialization."""
import importlib
import unittest

import torch
from torch import nn

training = importlib.import_module('_03_homeworks.homework_3.b_fashion_mnist_train')


class VGGVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def args(self, variant):
        return training.get_parser().parse_args([
            '--model', 'vgg', '--vgg_variant', variant, '--mean', '0.3',
            '--std', '0.2', '--no-dropout', '--augment'])

    def pipeline(self, args, model, augment=False):
        return training.configure_fashion_mnist_transforms(
            nn.Sequential(training.transforms.ConvertImageDtype(torch.float),
                          training.transforms.Normalize(0., .1)), args, model, augment=augment)

    def test_padding_preserves_original_pixels_and_pads_before_normalizing(self):
        args = self.args('pad')
        image = torch.arange(784).reshape(1, 1, 28, 28).float() / 784
        model = training.get_fashion_mnist_model(args).eval()
        result = self.pipeline(args, model)(image)
        self.assertEqual(tuple(result.shape), (1, 1, 32, 32))
        torch.testing.assert_close(result[:, :, 2:30, 2:30], (image - .3) / .2)
        torch.testing.assert_close(result[:, :, :2], torch.full((1, 1, 2, 32), -1.5))
        torch.testing.assert_close(result[:, :, :, :2], torch.full((1, 1, 32, 2), -1.5))
        torch.testing.assert_close(result[:, :, -2:], torch.full((1, 1, 2, 32), -1.5))
        torch.testing.assert_close(result[:, :, :, -2:], torch.full((1, 1, 32, 2), -1.5))

    def test_variants_have_same_initial_weights_and_classifier_input(self):
        reference = None
        for variant, size, pools in [('resize', 32, 5), ('pad', 32, 5), ('no_last_pool', 28, 4)]:
            with self.subTest(variant=variant):
                torch.manual_seed(42)
                args = self.args(variant)
                model = training.get_fashion_mnist_model(args).eval()
                images = self.pipeline(args, model)(torch.ones(2, 1, 28, 28))
                self.assertEqual(images.shape[-1], size)
                with torch.no_grad():
                    logits = model(images)
                self.assertEqual(tuple(logits.shape), (2, 10))
                self.assertEqual(sum(isinstance(m, nn.MaxPool2d) for m in model.modules()), pools)
                self.assertEqual(model.model[6].in_features, 512)
                state = {k: v.clone() for k, v in model.state_dict().items()}
                if reference is None:
                    reference = state
                else:
                    self.assertEqual(set(reference), set(state))
                    for key in reference:
                        torch.testing.assert_close(state[key], reference[key], rtol=0, atol=0)
                model.train()
                nn.functional.cross_entropy(model(images), torch.tensor([0, 1])).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))

    def test_augmentation_is_skipped_in_eval_and_uses_variant_size(self):
        for variant, size in [('resize', 32), ('pad', 32), ('no_last_pool', 28)]:
            args = self.args(variant)
            model = training.get_fashion_mnist_model(args)
            images = torch.rand(2, 1, 28, 28)
            pipeline = self.pipeline(args, model, augment=True)
            model.eval()
            torch.testing.assert_close(pipeline(images), self.pipeline(args, model)(images), rtol=0, atol=0)
            model.train()
            self.assertEqual(tuple(pipeline(images).shape), (2, 1, size, size))

    def test_old_args_default_to_resize(self):
        args = training.get_parser().parse_args(['--model', 'vgg', '--mean', '.3', '--std', '.2'])
        del args.vgg_variant
        model = training.get_fashion_mnist_model(args).eval()
        self.assertEqual(self.pipeline(args, model)(torch.ones(1, 1, 28, 28)).shape[-1], 32)


if __name__ == '__main__':
    unittest.main()
