# CUDA parallel training execution plan

Goal: Run all 21 commands documented under the main guards of 12 argparse
training programs in chapters 13–21, using CUDA and independent outputs.

User-approved constraints: new branch; preserve documented W&B flags and
training hyperparameters; smoke tests do not count as full runs; CPU fallback
is forbidden; confirm batch-size changes if the original size cannot fit.

- [x] Fix stale command paths and reproducible training entry-point errors.
- [x] Add bounded data workers and experiment-local checkpoint directories.
- [x] Add a CUDA-only child launcher and a two-slot, VRAM-aware queue, with
      command manifests, per-job logs, measured profiles and resumable status.
- [x] Test command discovery, GPU enforcement, checkpoint isolation, queue
      admission, LSTM loading and identified trainer regressions.
- [x] Run one-epoch, two-batch offline CUDA smoke tests for all 12 programs.
- [x] Start the full-training queue with original settings and documented W&B.
- [x] Review the branch and report live progress, failures and output paths.

Review focus: concurrent checkpoint writes; failed child exit codes; no-CUDA
startup; VRAM reserve and oversubscription; smoke/full-run separation.

## Execution ledger

- Branch: codex/train-13plus-cuda, starting from jyj.
- Working in the existing checkout as requested (new branch, no extra worktree).
- Initial checks: clean git status; CUDA available; RTX 3060 8GB; W&B API key
  configured; existing environment has no pytest, so use stdlib unittest.
- Additional identified errors: GoogLeNet passes an extra positional argument
  to its factory, and its validation loop overwrites rather than accumulates
  batch losses. These affect execution and reliable validation metrics.
- First CUDA smoke pass: all 21 commands completed with finite validation
  metrics, CUDA allocations and checkpoints; no batch-size changes required.
- User decision: keep VGG/NiN/GoogLeNet batch size 2048 and run those alone,
  waiving normal 1GiB headroom for these three models only.
- Whole-branch read-only reviewer found signal cleanup, stale resume/profile
  reuse, and retry-metrics contamination. Fixed with signal handlers and
  bounded termination, orphan PID checks, argument/source validation and
  separate attempt directories. Also reserve running jobs' future growth
  against current free VRAM if desktop memory use increases.
- Regression tests: 12 passing; reproductions for the identified failures were
  observed before fixes. `git diff --check` passes.
- Second smoke pass started with final code fingerprints and automatic
  transition to full training. Outputs: training_runs/2026-10-28_cuda.
- Final smoke pass: 21/21 completed successfully with matching fingerprints.
- Full training started: two CUDA jobs concurrently, 19 pending; both initial
  jobs reached epoch 8 with online W&B metrics and saved checkpoints.
- Full training remains in progress under detached supervisor PID 198855.
  These checkboxes describe implementation and launch, not 21 completed full runs.
