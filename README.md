# CRL-CW

PyTorch continual reinforcement-learning workspace for the Continual World
CW10 manipulation sequence on Meta-World v3.

The current project protocol is:

- environments: Meta-World v3 through Gymnasium;
- observations: full native v3 observations, with no 12-dimensional legacy cut;
- reward protocol: `cw10_v1`;
- `stick-pull-v3`: local `v1_compatible` reward alignment for the v3
  observation layout;
- pilot budget: `500,000` environment steps per task;
- evaluation: every `20,000` steps, five stochastic episodes, zero
  deterministic episodes.

`cw10_v1` maps ordinary tasks to Meta-World's v1-style reward. For
`stick-pull-v3`, the repo uses a wrapper instead of editing installed
Meta-World source, because the old reward indices do not match the v3
observation layout.

## CW10 Tasks

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

## Repository Layout

```text
configs/      YAML presets
docs/         research log, walkthrough, and analysis notes
methods/      method registry entries and method defaults
outputs/      generated experiment artifacts
prompts/      LLM controller prompts and task descriptions
references/   external papers and third-party code snapshots
scripts/      runnable training and diagnostic entry points
src/          agents, environments, training, evaluation, and utilities
tests/        automated tests
```

`references/` is not imported by the active training pipeline. `outputs/`
contains experiment evidence and can be large.

## Main Methods

- `single_task_baseline`: independent SAC for forward-transfer baselines.
- `fine_tuning`: sequential SAC with no retention mechanism.
- `clonex_sac`: ClonEx-style multi-head SAC with best-return exploration and
  Gaussian KL actor cloning.
- `full_bc`: complete successful old-task rollout memory with the same
  Gaussian KL actor cloning loss.
- `full_bc_norm_balanced`: Full BC with BC gradient magnitude capped relative
  to the current SAC shared-backbone gradient.
- `full_bc_pcgrad`: norm-balanced Full BC with asymmetric PCGrad that removes
  only BC components conflicting with the current SAC gradient.
- `semantic_local_bc`: static semantic memory using selected general segments.
- `semantic_hybrid_bc`: general, background, and task-specific semantic memory,
  with optional online LLM control of the general segment weights.

`gradient-aware BC` is an experimental diagnostic path in the stick-pull probe
script. It is default-off and is not part of the formal baseline set.

Full BC and ClonEx continual runs also collect read-only gradient-conflict
evidence under `<run_dir>/gradient_diagnostics/`. This logging is enabled by
their method presets, sampled sparsely, incrementally persisted, and does not
alter training gradients. Use `--no-gradient-diagnostics` for a control.

## Common Commands

Run one single-task baseline:

```bash
PYTHONPATH=src .venv/bin/python scripts/run.py \
  --mode single \
  --method single_task_baseline \
  --task hammer-v3 \
  --env-version v3 \
  --reward-function-version cw10_v1 \
  --total-steps 500000 \
  --eval-every 20000 \
  --stoch-eval-episodes 5 \
  --det-eval-episodes 0 \
  --device cpu
```

Run a CW10 single-task batch for one seed:

```bash
caffeinate -dimsu env PYTHONPATH=src .venv/bin/python scripts/run.py \
  --mode single-batch \
  --method single_task_baseline \
  --env-version v3 \
  --reward-function-version cw10_v1 \
  --num-tasks 10 \
  --steps-per-task 500000 \
  --eval-every 20000 \
  --stoch-eval-episodes 5 \
  --det-eval-episodes 0 \
  --seed 1 \
  --device cpu \
  --output-dir outputs/single_task_baselines \
  --run-name cw10_v3_v1_500k_seed1
```

Run a continual method:

```bash
caffeinate -dimsu env PYTHONPATH=src .venv/bin/python scripts/run.py \
  --mode continual \
  --method full_bc \
  --env-version v3 \
  --reward-function-version cw10_v1 \
  --steps-per-task 500000 \
  --eval-every 20000 \
  --stoch-eval-episodes 5 \
  --det-eval-episodes 0 \
  --baseline-curves outputs/single_task_baselines/cw10_v3_v1_500k_seeds1_2/aggregate/baseline_curves.json \
  --seed 0 \
  --device cpu \
  --run-name full_bc_cw10_v3_v1_500k_seed0
```

Formal forward-transfer reporting should use aggregated multi-seed
single-task curves, then compare each continual seed against the same aggregate
baseline. Aggregate FT excludes task 0 because no previous task can transfer to
the first task.

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
- `checkpoints/task_*.pt`
- `controller_prompts/` and `controller_decisions.json` for LLM runs

Main continual metrics include average performance, average forgetting, raw
forward transfer, normalized forward transfer, area forward transfer, and
per-task final success.

## Handover

Read these first:

1. `docs/research_log.md`
2. `docs/repo_walkthrough.md`
3. `scripts/run.py`
4. `methods/semantic_hybrid_bc.py`
5. `src/training/continual_experiment.py`

Run tests after changing environments, metrics, memory construction, semantic
segmentation, or LLM control:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```
