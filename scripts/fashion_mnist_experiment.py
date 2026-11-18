"""Run one recorded Fashion MNIST experiment without changing the teaching code."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def run_experiment(directory):
    import torch
    from torch.utils.data import TensorDataset
    import wandb

    directory = directory.resolve()
    configuration = json.loads((directory / 'config.json').read_text())
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for these experiments')
    if int(os.environ.get('TRAIN_MAX_BATCHES', '0')):
        raise RuntimeError('TRAIN_MAX_BATCHES must be zero: full datasets are required')
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    train_module = importlib.import_module('_03_homeworks.homework_3.b_fashion_mnist_train')
    data_module = importlib.import_module('_03_homeworks.homework_3.a_fashion_mnist_data')
    args = train_module.get_parser().parse_args(configuration['args'])
    actual_init = wandb.init
    actual_data = train_module.get_fashion_mnist_data
    actual_trainer = train_module.ClassificationTrainer
    actual_google_trainer = train_module.GoogLeNetClassificationTrainer
    captured = {}
    runtime = {'cuda': torch.cuda.get_device_name(0), 'torch': torch.__version__,
               'tensor_cache': 'GPU; equivalent ToTensor pixels, same Subset indices',
               'training_code_sha256': hashlib.sha256(Path(train_module.__file__).read_bytes()).hexdigest()}
    write_json(directory / 'status.json', dict(status='running', started=started, **runtime))

    def cached_data():
        with patch.object(data_module, 'get_num_cpu_cores', return_value=0):
            train, validation, pipeline = actual_data()
        original = train.dataset.dataset
        # ToTensor is the only dataset transform in the supplied data module.
        # Cache that deterministic conversion once, preserving loaders and split.
        # Divide on CPU exactly as torchvision ToTensor does, then copy to GPU.
        images = original.data.unsqueeze(1).float().div_(255).to('cuda')
        labels = original.targets.to('cuda')
        torch.testing.assert_close(images[0].cpu(), original[0][0], rtol=0, atol=0)
        tensor_dataset = TensorDataset(images, labels)
        train.dataset.dataset = tensor_dataset
        validation.dataset.dataset = tensor_dataset
        train.pin_memory = validation.pin_memory = False
        captured['indices'] = train.dataset.indices
        captured['validation_indices'] = validation.dataset.indices
        return train, validation, pipeline

    def capture_trainer(*a, **kw):
        captured['trainer'] = actual_trainer(*a, **kw)
        trainer = captured['trainer']
        # torchinfo already materialized lazy layers in eval mode. These hashes
        # prove paired seeds started from identical weights and shuffle state.
        state_hash = hashlib.sha256()
        for name, tensor in trainer.model.state_dict().items():
            state_hash.update(name.encode())
            state_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        runtime.update(
            initial_state_sha256=state_hash.hexdigest(),
            shuffle_state_sha256=hashlib.sha256(
                trainer.train_data_loader.sampler.generator.get_state().numpy().tobytes()).hexdigest(),
            training_indices_sha256=hashlib.sha256(json.dumps(captured['indices']).encode()).hexdigest(),
            validation_indices_sha256=hashlib.sha256(json.dumps(captured['validation_indices']).encode()).hexdigest(),
            parameter_count=sum(p.numel() for p in trainer.model.parameters()),
        )
        trainer.run.config.update(runtime)
        return captured['trainer']

    def capture_google_trainer(*a, **kw):
        captured['trainer'] = actual_google_trainer(*a, **kw)
        return captured['trainer']

    def recorded_init(**kwargs):
        kwargs['config'] = dict(kwargs['config'], **runtime)
        kwargs['dir'] = str(directory)
        kwargs['settings'] = wandb.Settings(init_timeout=30)
        try:
            run = actual_init(**kwargs)
        except Exception as error:
            if kwargs['mode'] != 'online':
                raise
            print('W&B online unavailable; preserving an offline run:', type(error).__name__, flush=True)
            kwargs['mode'] = 'offline'
            run = actual_init(**kwargs)
        captured['run'] = run
        actual_log = run.log

        def log(data, *a, **kw):
            record = dict(data, elapsed_seconds=time.time() - started)
            with (directory / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
                stream.flush()
            write_json(directory / 'status.json', dict(status='running', started=started,
                       latest=record, wandb_url=run.url, wandb_id=run.id, **runtime))
            return actual_log(data, *a, **kw)

        run.log = log

        class Context:
            def __enter__(self):
                return run

            def __exit__(self, error_type, error, traceback):
                if error_type is None:
                    trainer = captured['trainer']
                    checkpoint = next(directory.glob('*checkpoint_latest.pt'))
                    trainer.model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True))
                    validation_loss, validation_accuracy = trainer.do_validation()
                    history = [json.loads(line) for line in (directory / 'history.jsonl').read_text().splitlines()]
                    result = dict(
                        status='completed', configuration=configuration, args=vars(args),
                        checkpoint=str(checkpoint), checkpoint_validation_loss=validation_loss,
                        checkpoint_validation_accuracy=validation_accuracy,
                        best_logged_validation_accuracy=max(h['Validation accuracy (%)'] for h in history),
                        epochs=len(history), elapsed_seconds=time.time() - started,
                        peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / 2**20,
                        wandb_url=run.url, wandb_id=run.id,
                        wandb_mode='offline' if run.offline else 'online',
                        train_samples=len(trainer.train_data_loader.dataset),
                        validation_samples=len(trainer.validation_data_loader.dataset), **runtime,
                    )
                    write_json(directory / 'result.json', result)
                    write_json(directory / 'status.json', result)
                    run.summary.update({'checkpoint_validation_accuracy': validation_accuracy,
                                        'checkpoint_validation_loss': validation_loss,
                                        'elapsed_seconds': result['elapsed_seconds']})
                run.finish(exit_code=0 if error_type is None else 1)
                return False

        return Context()

    os.environ['TRAIN_CHECKPOINT_DIR'] = str(directory)
    os.environ['TRAIN_REQUIRE_CUDA'] = '1'
    try:
        with patch.object(train_module, 'get_fashion_mnist_data', cached_data), \
                patch.object(train_module, 'ClassificationTrainer', capture_trainer), \
                patch.object(train_module, 'GoogLeNetClassificationTrainer', capture_google_trainer), \
                patch.object(train_module.wandb, 'init', recorded_init):
            train_module.main(args)
    except BaseException as error:
        write_json(directory / 'status.json', dict(status='failed', error=repr(error),
                   elapsed_seconds=time.time() - started, **runtime))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    run_experiment(parser.parse_args().directory)
