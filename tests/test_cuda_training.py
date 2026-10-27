import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


class TrainingTests(unittest.TestCase):
    def test_all_documented_jobs_have_existing_paths_and_preserved_flags(self):
        from scripts.train_parallel import discover_jobs
        jobs = discover_jobs(ROOT)
        self.assertEqual(len(jobs), 21)
        self.assertEqual(len({job.script for job in jobs}), 12)
        self.assertEqual(len({job.id for job in jobs}), 21)
        for job in jobs:
            self.assertTrue((ROOT / job.script).is_file(), job.script)
        self.assertEqual(sum('--wandb' in job.args for job in jobs), 19)
        decay = [job.args[job.args.index('-w') + 1] for job in jobs
                 if 'weight_decay.py' in job.script]
        self.assertEqual(decay, ['0.0', '0.001', '0.002', '0.005', '0.01', '0.02'])

    def test_checkpoint_directories_do_not_overwrite_other_experiments(self):
        from _01_code._99_common_utils.early_stopping import EarlyStopping
        model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            for name, loss in [('a', 1.0), ('b', 2.0)]:
                destination = Path(directory) / name
                with patch.dict(os.environ, {'TRAIN_CHECKPOINT_DIR': str(destination)}):
                    stopping = EarlyStopping(project_name='same', checkpoint_file_path=directory,
                                             run_time_str='same-second')
                    stopping.check_and_save(loss, model)
                self.assertTrue((destination / 'same_checkpoint_latest.pt').is_file())
            self.assertFalse((Path(directory) / 'same_checkpoint_latest.pt').exists())

    def test_worker_limit_and_smoke_batch_limit(self):
        from _01_code._99_common_utils.training_runtime import data_workers, training_batches
        with patch.dict(os.environ, {'TRAIN_NUM_WORKERS': '2', 'TRAIN_MAX_BATCHES': '2'}):
            self.assertEqual(data_workers(), 2)
            self.assertEqual(list(training_batches(range(10))), [0, 1])
        with patch.dict(os.environ, {'TRAIN_MAX_BATCHES': '0'}):
            self.assertEqual(list(training_batches(range(4))), [0, 1, 2, 3])

    def test_launcher_refuses_cpu_before_executing_script(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'should_not_run.py'
            marker = Path(directory) / 'marker'
            script.write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/train_job.py'), str(script)],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('CUDA is required', result.stderr)
            self.assertFalse(marker.exists())

    def test_vram_admission_respects_running_reservation_and_headroom(self):
        from scripts.train_parallel import can_admit
        self.assertTrue(can_admit(1500, [1000], 6000, 7000, 2, 1024))
        self.assertFalse(can_admit(4000, [2500], 6000, 7000, 2, 1024))
        self.assertFalse(can_admit(1000, [1000, 1000], 6000, 7000, 2, 1024))
        self.assertFalse(can_admit(5500, [], 6000, 7000, 2, 1024))
        # A growing desktop workload must not consume a running job's unused reserve.
        self.assertFalse(can_admit(1000, [2500], 4000, 7000, 2, 1024))

    def test_resume_rejects_missing_checkpoints_and_changed_arguments(self):
        import json
        from scripts.train_parallel import completed_result
        from _01_code._99_common_utils.training_runtime import code_fingerprint
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = directory / 'model.pt'
            report = dict(status='completed', exit_code=0, script='/model.py', args=['-b', '32'],
                          checkpoints=[str(checkpoint)], code_fingerprint=code_fingerprint())
            (directory / 'result.json').write_text(json.dumps(report))
            self.assertIsNone(completed_result(directory, '/model.py', ['-b', '32']))
            checkpoint.write_bytes(b'checkpoint')
            self.assertIsNone(completed_result(directory, '/model.py', ['-b', '64']))
            self.assertEqual(completed_result(directory, '/model.py', ['-b', '32'])['status'], 'completed')
            report['code_fingerprint'] = 'old-code'
            (directory / 'result.json').write_text(json.dumps(report))
            self.assertIsNone(completed_result(directory, '/model.py', ['-b', '32']))

    def test_retry_gets_fresh_metrics_and_preserves_previous_attempt(self):
        from scripts.train_parallel import new_attempt_directory
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = new_attempt_directory(root)
            (first / 'metrics.json').write_text('{"best_validation_loss": 0.01}')
            second = new_attempt_directory(root)
            self.assertNotEqual(first, second)
            self.assertFalse((second / 'metrics.json').exists())
            self.assertTrue((first / 'metrics.json').exists())

    def test_exclusive_large_job_keeps_original_batch_without_other_training(self):
        from scripts.train_parallel import can_admit_exclusive
        self.assertTrue(can_admit_exclusive(5500, [], 6300))
        self.assertFalse(can_admit_exclusive(5500, [800], 6300))
        self.assertFalse(can_admit_exclusive(5500, [], 5800))

    def test_termination_handler_stops_detached_child_group(self):
        import signal
        from scripts.train_parallel import terminate_child, termination_signal
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        try:
            with self.assertRaises(KeyboardInterrupt):
                termination_signal(signal.SIGTERM, None)
            terminate_child(process)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_lstm_main_passes_run_to_data_loader(self):
        # Stop at the data boundary: no training or external W&B requests.
        from types import SimpleNamespace
        for module_name in ['g_crypto_currency_regression_train_lstm',
                            'i_crypto_currency_classification_train_lstm']:
            module = importlib.import_module('_01_code._18_lstm_and_its_application.' + module_name)
            run = SimpleNamespace(config=SimpleNamespace())
            class StopAtData(Exception):
                pass
            def data(*args, **kwargs):
                self.assertTrue(args, 'missing run argument')
                self.assertIs(args[0], run)
                raise StopAtData
            args = module.get_parser().parse_args([])
            with patch.object(module.wandb, 'init') as init, patch.object(module, 'get_btc_krw_data', data):
                init.return_value.__enter__.return_value = run
                with self.assertRaises(StopAtData):
                    module.main(args)

    def test_googlenet_validation_averages_all_batches(self):
        from _01_code._16_modern_cnns._03_google_net.b_google_net_trainer import GoogLeNetClassificationTrainer
        import math
        class ThreeHeads(torch.nn.Module):
            def forward(self, x):
                return x, x, x
        batches = [(torch.tensor([[2., 0.]]), torch.tensor([0])),
                   (torch.tensor([[0., 2.]]), torch.tensor([0]))]
        trainer = GoogLeNetClassificationTrainer('test', ThreeHeads(), None, [], batches,
                                                lambda x: x, 'now', None, torch.device('cpu'), '.')
        loss, accuracy = trainer.do_validation()
        expected = 1.6 * (math.log1p(math.exp(-2)) + math.log1p(math.exp(2))) / 2
        self.assertAlmostEqual(float(loss), expected, places=5)
        self.assertEqual(accuracy, 50.)

    def test_googlenet_main_calls_factory_with_supported_arguments(self):
        from types import SimpleNamespace
        module = importlib.import_module('_01_code._16_modern_cnns._03_google_net.a_cifar10_train_google_net')
        class StopAtModel(Exception):
            pass
        def factory(num_classes=10):
            self.assertEqual(num_classes, 10)
            raise StopAtModel
        with patch.object(module.wandb, 'init') as init, \
             patch.object(module, 'get_cifar10_data', return_value=([], [], None)), \
             patch.object(module, 'get_google_net_model', factory):
            init.return_value.__enter__.return_value = SimpleNamespace(config=SimpleNamespace())
            with self.assertRaises(StopAtModel):
                module.main(module.get_parser().parse_args([]))


if __name__ == '__main__':
    unittest.main()
