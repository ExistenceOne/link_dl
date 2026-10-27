"""Runtime controls for scheduled experiments; defaults preserve standalone runs."""
import itertools
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path


def code_fingerprint():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / '_01_code').rglob('*.py')) + [root / 'scripts/train_job.py', root / 'scripts/train_parallel.py']
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def data_workers():
    value = os.environ.get('TRAIN_NUM_WORKERS')
    if value is None:
        return multiprocessing.cpu_count() if os.name == 'nt' or os.sys.platform.startswith('linux') else 0
    workers = int(value)
    if workers < 0:
        raise ValueError('TRAIN_NUM_WORKERS must be nonnegative')
    return workers


def training_batches(loader):
    limit = int(os.environ.get('TRAIN_MAX_BATCHES', '0'))
    return itertools.islice(loader, limit) if limit > 0 else iter(loader)


def record_validation(epoch, train_loss, validation_loss, run, device):
    """Persist machine-readable progress without depending on W&B being enabled."""
    if not math.isfinite(float(train_loss)) or not math.isfinite(float(validation_loss)):
        raise RuntimeError('Non-finite training or validation loss')
    if os.environ.get('TRAIN_REQUIRE_CUDA') == '1' and device.type != 'cuda':
        raise RuntimeError('CUDA is required; refusing CPU training')
    destination = os.environ.get('TRAIN_METRICS_FILE')
    if not destination:
        return
    path = Path(destination)
    previous = json.loads(path.read_text()) if path.exists() else {}
    data = dict(epoch=epoch, train_loss=float(train_loss), validation_loss=float(validation_loss),
                best_validation_loss=min(float(validation_loss), previous.get('best_validation_loss', float('inf'))),
                device=str(device), wandb_id=run.id, wandb_url=run.url)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)
