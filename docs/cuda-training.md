# CUDA training queue

Run commands from `/home/link/link_dl` using the existing virtual environment.
The queue reads the `# python ...` comments under each training entry point in
chapters 13–21. There are 21 experiments in 12 files. Chapters 19 and 20 use
neither argparse nor W&B and are excluded. W&B flags and hyperparameters from
the documented commands are preserved in full runs.

```bash
.venv/bin/python scripts/train_parallel.py --output training_runs/2026-10-28_cuda --phase all
```

`smoke` runs each command for one epoch, at most two training/validation
batches, with W&B disabled. These runs are not counted as full training.
`full` requires a matching successful smoke profile for every experiment.
`all` runs both phases. Repeating a command resumes completed experiments only
when the effective arguments, code fingerprint and checkpoint files match.
Failed/interrupted experiments restart from scratch in a new attempt directory;
optimizer state is not restored. Earlier attempts are retained.

CUDA is mandatory. The child launcher fails before loading the training script
if CUDA is unavailable. The normal limit is two jobs, four compute threads and
two DataLoader workers per job, with 1024MiB VRAM headroom. Admission uses
measured allocator peaks plus a CUDA context allowance. A lock prevents two
supervisors from using the same output directory.

The user explicitly chose to keep batch size 2048 for VGG, NiN and GoogLeNet.
Those three run alone, without the 1024MiB headroom requirement. Admission still
checks current free VRAM against the measured allocated peak plus 512MiB.
No batch sizes are automatically reduced. Other jobs retain their documented
options and argparse defaults (the RNN defaults to batch size 32).

Outputs under `training_runs/2026-10-28_cuda/` (ignored by Git):

- `manifest.json`: all original commands and experiment IDs.
- `smoke_status.json`, `full_status.json`: pending/running/completed/failed/blocked jobs.
- `supervisor.pid`, `supervisor.log`: detached supervisor identity and events.
- `<phase>/<job>/result.json`: latest attempt's completion report.
- `<phase>/<job>/attempt-NNN/command.json`: exact command, PID and start time.
- `<phase>/<job>/attempt-NNN/train.log`: per-attempt stdout and stderr.
- `<phase>/<job>/attempt-NNN/metrics.json`: live validation metrics and W&B URL/ID.
- `<phase>/<job>/attempt-NNN/result.json`: exit code, CUDA peaks, duration, parsed arguments and checkpoint paths.
- `<phase>/<job>/attempt-NNN/checkpoints/`: independent best checkpoints.

Stop the supervisor with SIGTERM using the PID in `supervisor.pid`. It
terminates child process groups and escalates to SIGKILL after 15 seconds.
If a supervisor was killed forcibly, a new supervisor refuses to start over a
still-running child; inspect the persisted command/PID before intervening.

Run regression tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
