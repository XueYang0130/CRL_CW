# CRL-CW

This project is a modern continual manipulation RL workspace built on:

- `Meta-World v3`
- `Gymnasium`
- `PyTorch`

It keeps the fixed `CW10` task order while using the current `Meta-World`
environment stack and full observations.

## Benchmark

The benchmark task sequence is:

1. `hammer-v3`
2. `push-wall-v3`
3. `faucet-close-v3`
4. `push-back-v3`
5. `stick-pull-v3`
6. `handle-press-side-v3`
7. `push-v3`
8. `shelf-place-v3`
9. `window-close-v3`
10. `peg-unplug-side-v3`

Single-task runs use the raw environment observation.

Continual runs append a ten-way task one-hot vector to the observation.

## Defaults

The main experiment protocol follows the reference SAC continual benchmark
hyperparameter defaults where practical:

- `1,000,000` steps per task
- `20,000` evaluation interval
- `1,000,000` replay size
- `128` batch size
- `10,000` random start steps
- `1,000` update start
- `50` update frequency
- `1e-3` learning rate
- `0.99` discount
- `0.995` Polyak averaging
- `200` max episode length

This project matches the baseline training protocol defaults, not the old
MuJoCo 2.0 stack or the old 12-dimensional observation protocol.

## Layout

```text
src/
  agents/
  envs/
  evaluation/
  training/
  utils/

methods/
  fine_tuning.py
  task_conditioned.py
  packnet.py
  clonex_sac.py

scripts/
  run.py
```

`outputs/` stores experiment artifacts.

`references/` stores external papers and third-party code snapshots.

`methods/` contains the method registry entries, architecture choices,
agent factories, and method-specific defaults. The continual runner contains
only the shared task loop.

## Running

Run one single-task baseline:

```bash
python scripts/run.py \
  --mode single \
  --method single_task_baseline \
  --task hammer-v3 \
  --total-steps 1000000 \
  --eval-every 20000 \
  --device cpu
```

Run an independent CW10 single-task baseline batch:

```bash
python scripts/run.py \
  --mode single-batch \
  --method single_task_baseline \
  --steps-per-task 1000000 \
  --eval-every 20000 \
  --device cpu
```

This batch command also aggregates the completed single-task runs into:

- `aggregate/baseline_curves.json`
- `aggregate/summary.json`

Run continual finetuning on CW10:

```bash
python scripts/run.py \
  --mode continual \
  --method fine_tuning \
  --sequence-task-count 10 \
  --steps-per-task 1000000 \
  --eval-every 20000 \
  --device cpu
```

Run the final ClonEx-SAC method with its paper defaults:

```bash
python scripts/run.py \
  --mode continual \
  --method clonex_sac \
  --sequence-task-count 10 \
  --steps-per-task 1000000 \
  --eval-every 20000 \
  --device cpu
```

Run from a YAML config:

```bash
python scripts/run.py \
  --config configs/w10_pilot.yaml
```

If you already have aggregated single-task baseline curves, pass them to the
continual run so the final summary includes forward transfer:

```bash
python scripts/run.py \
  --mode continual \
  --method fine_tuning \
  --baseline-curves outputs/cw10_single_task_baselines/<run>/aggregate/baseline_curves.json
```

## Outputs

A single-task run writes:

- `config.json`
- `evaluations.csv`
- `summary.json`
- `checkpoints/`

A continual run writes:

- `config.json`
- `evaluations.csv`
- `task_summaries.csv`
- `summary.json`
- `checkpoints/`

The continual `summary.json` includes:

- `average_performance`
- `average_return`
- `average_forgetting`
- `forward_transfer`
- per-task final success
- per-task forgetting

## Notes

- Smoke tests were validated on CPU.
- The project disables Gymnasium's passive env checker for Meta-World env
  creation to avoid noisy observation-space warnings from the upstream env
  definitions.
