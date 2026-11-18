# Fashion MNIST VGG variant comparison

User-approved design: compare existing resize VGG, input zero-padding (2 pixels on every side), and native 28x28 VGG with the final MaxPool replaced by Identity. Import the existing custom VGG factory; retain all convolutions, BN and FC layers. Do not modify the supplied data module or Trainer.do_train().

Fixed conditions: Adam, LR 0.0003, weight decay 0.0001, batch 256, BN on, Dropout off, augmentation on, input normalization mean 0.2855656942074541/std 0.3527285864947437. Split seed 42; 55,000 train and 5,000 validation samples. Max 100 epochs, patience 10, delta 1e-5, validation each epoch. Select minimum validation-loss checkpoint using existing EarlyStopping. Seeds 42,43,44 for each variant (nine fresh runs). No test data or retuning.

Transforms: ConvertImageDtype -> Resize32 / Pad(2,fill=0) / unchanged28 -> training-only RandomCrop(native variant size,padding=2) and HorizontalFlip -> Normalize. Evaluation excludes augmentation. Preserve legacy result args without vgg_variant as resize.

Tasks:
- [x] 1. Add --vgg_variant resize|pad|no_last_pool, shared train/evaluation transforms, regression tests for exact pad pixels, no augmentation at eval, 512 FC input, equal initial weights and backward compatibility.
- [x] 2. Extend existing experiment recording with initial model/shuffle/split hashes and parameter count. Add resumable nine-run comparison coordinator and tests with hand-computed summary fixtures. Verify all tests before GPU runs.
- [x] 3. Execute nine fresh runs sequentially. Record W&B runs as run, full data use and per-epoch histories. Confirm matching initialization and sampler state for each seed, split constant for all runs, equal parameter counts, completed runs only.
- [x] 4. Compare selected-checkpoint validation accuracy mean/sample std, seed-paired differences, loss, epochs, wall time and GPU allocation. Export CSV, JSON and plots. Compare resize reproduction against existing scores. Independently review changes, verify outputs and report conclusions.

Review focus: legacy checkpoint loading; pad before normalization (zero border in original intensity space); removing only final pool; train/eval augmentation separation; no stale-run reuse or test-based selection. Preserve previous training and evaluation artifacts.


## Completed results (2026-11-05 KST)

All nine fresh CUDA runs completed; no test data read. Fixed training source fingerprint and paired initial state/shuffle/split fingerprints verified. All models have 9,755,274 parameters. Checkpoint accuracy is from restoration of the existing delta-based minimum-validation-loss selection, not peak epoch accuracy. Full regression suite: 34 tests passed in 6.070 sec; diff check clean. Fresh read-only review found no Critical/Important issues. Visual inspection passed for both exported plots.

| Variant | Seed 42 | Seed 43 | Seed 44 | Mean +/- sample SD (%) | Mean validation loss | Mean wall min | Wall sec / epoch | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| resize | 93.84 | 93.86 | 93.50 | 93.73 +/- 0.20 | 0.1761 | 6.21 | 10.76 | 750.50 |
| pad | 94.20 | 93.92 | 93.70 | 93.94 +/- 0.25 | 0.1803 | 7.57 | 10.73 | 750.50 |
| no_last_pool | 93.84 | 93.04 | 93.46 | 93.45 +/- 0.40 | 0.1843 | 5.04 | 7.96 | 646.68 |

Padding accuracy is higher in all three paired seeds (+0.36,+0.06,+0.20 pp; mean +0.2067 pp), but three seeds and a small gain do not establish a decisive general advantage. Resize has the lower mean validation loss. Removing the final pool at native 28x28 reduces mean accuracy by 0.2867 pp; wall seconds per epoch decrease 26.01%, allocated peak memory decreases 13.83%. Per-epoch timing includes setup/restore amortized over epochs. Pad wall time is longer because early stopping ran more epochs, not because each epoch was slower. Three-seed resize mean reproduces prior mean to -0.0333 pp (prior 93.7667%, fresh 93.7333%); individual seeded GPU results are not bitwise reproduced.

Artifacts: `_03_homeworks/homework_3/experiments/vgg_variant_comparison/{comparison.json,runs.csv,comparison.png,validation_curves.png,audit.json}`; individual run configs, histories, checkpoints and W&B links in `runs/`. Total worker wall time 3387.28 sec (56.45 min). Prior final_selection.json, test evaluation, PDF and notebook preserved; this is a validation comparison, not a new test evaluation.

Review minor deferred: automatic rejection of changed code on resume is not implemented; these nine runs were separately verified against the identical current b_fashion_mnist_train.py SHA256. No other review findings or unresolved execution failures.

Run/rebuild: `.venv/bin/python scripts/fashion_mnist_vgg_comparison.py` (resumes completed identical configurations) or append `--report_only` for export without training. Select a training variant with `--model vgg --vgg_variant resize|pad|no_last_pool` in b_fashion_mnist_train.py.
