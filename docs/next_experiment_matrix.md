## Next Experiment Matrix

This note turns the recent discussion into a concrete command-level experiment queue.

The goal is to separate four questions:

1. whether continual forgetting is locally structured;
2. whether naive fixed-event local BC is sufficient;
3. whether preservation quality depends on coverage size rather than only on semantic locality;
4. whether single-task SAC performance reveals a new-task learning gap under continual transfer.

All commands below assume:

- working directory: `~/crl_cw`
- environment: `.venv`
- reward protocol: `cw10_v1`
- source continual run: `outputs/cw10_continual/fine_tuning_cw10_v3_v1_500k_seed0`

## Group A: Pairwise Coverage Ablation

### A1. `faucet-close-v3 -> push-back-v3`

This pair currently behaves like a hard case for local BC.

Run:

1. `no_bc`
2. `local_bc(contact_or_alignment + manipulation)`
3. `local_bc(approach + contact_or_alignment + manipulation)`
4. `full_bc_173`
5. `full_bc_798`
6. `full_bc_full`

These runs should already exist or be rerun only if needed:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_bc_pair.py \
  --run-dir outputs/cw10_continual/fine_tuning_cw10_v3_v1_500k_seed0 \
  --old-task-index 2 \
  --new-task-index 3 \
  --segments contact_or_alignment manipulation \
  --variant no_bc \
  --reference-episodes 20 \
  --reference-max-attempts 80 \
  --steps 100000 \
  --eval-every 20000 \
  --stoch-eval-episodes 5 \
  --batch-size 128 \
  --local-bc-batch-size 128 \
  --local-bc-coefficient 10.0 \
  --seed 0 \
  --device cpu \
  --run-name pair_faucet_pushback_no_bc
```

```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_bc_pair.py \
  --run-dir outputs/cw10_continual/fine_tuning_cw10_v3_v1_500k_seed0 \
  --old-task-index 2 \
  --new-task-index 3 \
  --segments contact_or_alignment manipulation \
  --variant local_bc \
  --reference-episodes 20 \
  --reference-max-attempts 80 \
  --steps 100000 \
  --eval-every 20000 \
  --stoch-eval-episodes 5 \
  --batch-size 128 \
  --local-bc-batch-size 128 \
  --local-bc-coefficient 10.0 \
  --seed 0 \
  --device cpu \
  --run-name pair_faucet_pushback_local_bc
```

```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_bc_pair.py \
  --run-dir outputs/cw10_continual/fine_tuning_cw10_v3_v1_500k_seed0 \
  --old-task-index 2 \
  --new-task-index 3 \
  --segments approach contact_or_alignment manipulation \
  --variant local_bc \
  --reference-episodes 20 \
  --reference-max-attempts 80 \
  --steps 100000 \
  --eval-every 20000 \
  --stoch-eval-episodes 5 \
  --batch-size 128 \
  --local-bc-batch-size 128 \
  --local-bc-coefficient 10.0 \
  --seed 0 \
  --device cpu \
  --run-name pair_faucet_pushback_local_bc_acm
```

```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_bc_pair.py \
  --run-dir outputs/cw10_continual/fine_tuning_cw10_v3_v1_500k_seed0 \
  --old-task-index 2 \
  --new-task-index 3 \
  --segments contact_or_alignment manipulation \
  --variant full_bc \
  --reference-episodes 20 \
  --reference-max-attempts 80 \
  --max-reference-states 173 \
  --steps 100000 \
  --eval-every 20000 \
  --stoch-eval-episodes 5 \
  --batch-size 128 \
  --local-bc-batch-size 128 \
  --local-bc-coefficient 10.0 \
  --seed 0 \
  --device cpu \
  --run-name pair_faucet_pushback_full_bc_173
```

```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_bc_pair.py \
  --run-dir outputs/cw10_continual/fine_tuning_cw10_v3_v1_500k_seed0 \
  --old-task-index 2 \
  --new-task-index 3 \
  --segments approach contact_or_alignment manipulation \
  --variant full_bc \
  --reference-episodes 20 \
  --reference-max-attempts 80 \
  --max-reference-states 798 \
  --steps 100000 \
  --eval-every 20000 \
  --stoch-eval-episodes 5 \
  --batch-size 128 \
  --local-bc-batch-size 128 \
  --local-bc-coefficient 10.0 \
  --seed 0 \
  --device cpu \
  --run-name pair_faucet_pushback_full_bc_798
```

```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_bc_pair.py \
  --run-dir outputs/cw10_continual/fine_tuning_cw10_v3_v1_500k_seed0 \
  --old-task-index 2 \
  --new-task-index 3 \
  --segments contact_or_alignment manipulation \
  --variant full_bc \
  --reference-episodes 20 \
  --reference-max-attempts 80 \
  --steps 100000 \
  --eval-every 20000 \
  --stoch-eval-episodes 5 \
  --batch-size 128 \
  --local-bc-batch-size 128 \
  --local-bc-coefficient 10.0 \
  --seed 0 \
  --device cpu \
  --run-name pair_faucet_pushback_full_bc
```

### A2. `faucet-close-v3 -> handle-press-side-v3`

This pair tests whether a late-focused segment is sufficient.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_bc_pair.py \
  --run-dir outputs/cw10_continual/fine_tuning_cw10_v3_v1_500k_seed0 \
  --old-task-index 2 \
  --new-task-index 5 \
  --segments finish_or_stabilize \
  --variant no_bc \
  --reference-episodes 20 \
  --reference-max-attempts 80 \
  --steps 100000 \
  --eval-every 20000 \
  --stoch-eval-episodes 5 \
  --batch-size 128 \
  --local-bc-batch-size 128 \
  --local-bc-coefficient 10.0 \
  --seed 0 \
  --device cpu \
  --run-name pair_faucet_handlepress_no_bc
```

```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_bc_pair.py \
  --run-dir outputs/cw10_continual/fine_tuning_cw10_v3_v1_500k_seed0 \
  --old-task-index 2 \
  --new-task-index 5 \
  --segments finish_or_stabilize \
  --variant local_bc \
  --reference-episodes 20 \
  --reference-max-attempts 80 \
  --steps 100000 \
  --eval-every 20000 \
  --stoch-eval-episodes 5 \
  --batch-size 128 \
  --local-bc-batch-size 128 \
  --local-bc-coefficient 10.0 \
  --seed 0 \
  --device cpu \
  --run-name pair_faucet_handlepress_local_bc_finish
```

```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_bc_pair.py \
  --run-dir outputs/cw10_continual/fine_tuning_cw10_v3_v1_500k_seed0 \
  --old-task-index 2 \
  --new-task-index 5 \
  --segments finish_or_stabilize \
  --variant full_bc \
  --reference-episodes 20 \
  --reference-max-attempts 80 \
  --steps 100000 \
  --eval-every 20000 \
  --stoch-eval-episodes 5 \
  --batch-size 128 \
  --local-bc-batch-size 128 \
  --local-bc-coefficient 10.0 \
  --seed 0 \
  --device cpu \
  --run-name pair_faucet_handlepress_full_bc
```

## Group B: Single-Task Reference Runs

These runs measure whether the new task itself is easy under the same SAC protocol outside the continual setting.

### B1. `push-back-v3` single-task reference

```bash
PYTHONPATH=src .venv/bin/python scripts/run.py \
  --mode single \
  --method single_task_baseline \
  --task push-back-v3 \
  --env-version v3 \
  --reward-function-version cw10_v1 \
  --total-steps 100000 \
  --eval-every 20000 \
  --stoch-eval-episodes 5 \
  --det-eval-episodes 0 \
  --seed 0 \
  --device cpu \
  --run-name single_push_back_v3_100k_seed0
```

### B2. `handle-press-side-v3` single-task reference

```bash
PYTHONPATH=src .venv/bin/python scripts/run.py \
  --mode single \
  --method single_task_baseline \
  --task handle-press-side-v3 \
  --env-version v3 \
  --reward-function-version cw10_v1 \
  --total-steps 100000 \
  --eval-every 20000 \
  --stoch-eval-episodes 5 \
  --det-eval-episodes 0 \
  --seed 0 \
  --device cpu \
  --run-name single_handle_press_side_v3_100k_seed0
```

## Group C: Event-Segment Manifest Construction

This group turns the diagnostic event summaries into a reusable segment-selection manifest.

### C1. Example manifest from the faucet smoke-event output

```bash
PYTHONPATH=src .venv/bin/python scripts/build_selective_memory_manifest.py \
  --event-summaries \
    outputs/diagnostics/smoke_event_segments_faucet/task_2_faucet-close-v3/event_segment_summary.csv \
  --score-mode drift_mass \
  --top-k 2 \
  --run-name smoke_segment_manifest
```

### C2. Manifest from a stronger diagnostic set

If event-segment outputs are available for the target pairs, combine them:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_selective_memory_manifest.py \
  --event-summaries \
    outputs/diagnostics/smoke_event_segments_faucet/task_2_faucet-close-v3/event_segment_summary.csv \
  --score-mode drift_mass \
  --top-k 3 \
  --run-name faucet_pair_segment_manifest
```

## Group D: Interpretation Checklist

For each pairwise run, record:

- old final success
- new final success
- reference state count
- whether the new task is below its single-task reference
- whether the old task is above the no-BC baseline

Interpretation logic:

- if `single-task >> pairwise`, then the pair is showing a real learning-gap / interference effect
- if `full_bc` preserves old performance but suppresses new-task learning, then BC is trading plasticity for stability
- if `local_bc` preserves old performance with less new-task damage than equally sized `full_bc`, then the local selection mechanism is promising
- if `local_bc` fails while local drift clearly exists, the next method should improve **segment selection**, not simply add more event-bin combinations
