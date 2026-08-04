# Research Log and Handover

For code-level orientation, read [repo_walkthrough.md](repo_walkthrough.md)
after this document.

## 1. Research Question

This project studies continual reinforcement learning on the CW10 Meta-World
manipulation sequence. The working hypothesis is that transfer and forgetting
are not uniform over a whole policy or a whole trajectory. Old and new tasks
contain both cooperative behavioral structure and conflicting behavioral
structure.

The proposed direction is to preserve old knowledge selectively:

1. collect successful old-policy rollouts;
2. segment those rollouts into meaningful behavioral stages;
3. apply Gaussian KL behavior cloning only to selected memory;
4. use broad, task-specific, and possibly LLM-controlled memory channels so
   retention is not reduced to a single fixed trajectory filter.

Full BC is not the intended contribution. It is the current empirical oracle.
The research target is a structured semantic memory method that approaches or
exceeds Full BC's retention and transfer while using less memory, producing a
more interpretable mechanism, and exposing where old experience helps or
interferes.

## 2. Current Protocol

All currently valid pilot results use:

- benchmark: CW10 task order;
- environment stack: native Meta-World v3 with Gymnasium;
- observations: full v3 observations, no legacy 12-dimensional truncation;
- reward protocol: `cw10_v1`;
- `stick-pull-v3`: project-local `v1_compatible` reward wrapper;
- stick-pull alignment: `stick=obs[4:7]`, `handle=obs[11:14]`,
  `container=handle+[0.05,0,0]`;
- installed Meta-World source: not modified;
- horizon: 200 steps;
- pilot budget: 500,000 environment steps per task;
- evaluation interval: 20,000 steps;
- evaluation: five stochastic episodes and zero deterministic episodes;
- stochastic success: episode-level success if `info["success"]` is true at
  any step.

The current FT reference is the checkpoint-wise mean of complete pure
single-task seed 1 and seed 2 runs:
`outputs/single_task_baselines/cw10_v3_v1_500k_seeds1_2/aggregate/baseline_curves.json`.
It should be rebuilt with the same aggregation script when more seeds finish.

## 3. CW10 Task Order

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

## 4. Semantic Representation

The active segmenter is `task_aware_v3`. It uses task-aware geometry, motion,
grasp/contact, progress, success, and a post-success stabilization window.

### General labels

- `approach`
- `contact_or_alignment`
- `manipulation`
- `finish_or_stabilize`

### Task-specific labels

The segmenter also emits 22 task-specific refinement labels:

```text
align_hammer_to_nail
align_object_to_shelf
align_tcp_for_push
align_to_side_handle
align_tool_to_handle
approach_object
complete_and_stabilize
engage_faucet_handle
engage_window_handle
establish_contact
establish_hammer_contact
extract_peg_from_socket
grasp_and_pull_peg
grasp_or_engage_object
lift_and_transport_object
lift_and_transport_tool
move_engaged_object
press_handle_down
push_object_toward_target
rotate_faucet_closed
slide_window_closed
transport_hammer
```

| Task | Possible task-specific labels |
|---|---|
| `hammer-v3` | `approach_object`, `establish_hammer_contact`, `grasp_or_engage_object`, `transport_hammer`, `align_hammer_to_nail`, `move_engaged_object`, `complete_and_stabilize` |
| `push-wall-v3` | `approach_object`, `align_tcp_for_push`, `push_object_toward_target`, `complete_and_stabilize` |
| `faucet-close-v3` | `approach_object`, `engage_faucet_handle`, `rotate_faucet_closed`, `complete_and_stabilize` |
| `push-back-v3` | `approach_object`, `align_tcp_for_push`, `push_object_toward_target`, `complete_and_stabilize` |
| `stick-pull-v3` | `approach_object`, `establish_contact`, `grasp_or_engage_object`, `lift_and_transport_tool`, `align_tool_to_handle`, `move_engaged_object`, `complete_and_stabilize` |
| `handle-press-side-v3` | `approach_object`, `align_to_side_handle`, `press_handle_down`, `complete_and_stabilize` |
| `push-v3` | `approach_object`, `align_tcp_for_push`, `push_object_toward_target`, `complete_and_stabilize` |
| `shelf-place-v3` | `approach_object`, `establish_contact`, `grasp_or_engage_object`, `lift_and_transport_object`, `align_object_to_shelf`, `move_engaged_object`, `complete_and_stabilize` |
| `window-close-v3` | `approach_object`, `engage_window_handle`, `slide_window_closed`, `complete_and_stabilize` |
| `peg-unplug-side-v3` | `approach_object`, `establish_contact`, `grasp_or_engage_object`, `grasp_and_pull_peg`, `extract_peg_from_socket`, `move_engaged_object`, `complete_and_stabilize` |

## 5. Completed Pilot Runs

These are seed-0 pilot results preserved for reporting continuity. They are
useful for direction setting, but they are not final statistical claims.

The FT columns below are pilot values. Older runs were computed before the
current formal FT rule was locked to exclude task 0 and before the current
multi-seed single-task aggregate was finalized. Keep them in the running log
for progress reporting, then recompute the formal FT table after the new
single-task aggregate and the new continual multi-seed runs finish.

| Method | Average Performance | Average Forgetting | Forward Transfer | Area Forward Transfer | Average Return | Status |
|---|---:|---:|---:|---:|---:|---|
| Fine-tuning | 0.112 | 0.536 | -0.685 | -0.822 | 31,402.638 | valid lower-bound pilot |
| ClonEx-SAC | 0.732 | 0.028 | -0.267 | -0.381 | 231,822.926 | valid literature-style baseline pilot |
| Full BC | 0.904 | 0.048 | 0.239 | 0.210 | 255,417.002 | valid empirical oracle pilot |
| Semantic Local BC v3 | 0.684 | 0.224 | pending recomputation | pending recomputation | 129,207.431 | valid static semantic pilot |
| Semantic Hybrid LLM, three-channel | 0.820 | -0.032 | pending recomputation | pending recomputation | 186,332.000 | valid seed-0 proposed-method pilot; needs multi-seed confirmation |

Current interpretation:

- Fine-tuning shows severe catastrophic forgetting.
- ClonEx-SAC gives strong retention but did not dominate transfer in seed 0.
- Full BC is the strongest completed seed-0 method and shows that successful
  old-task trajectories contain useful retention and transfer information.
- Static Semantic Local BC is better than fine-tuning but loses too much old
  knowledge.
- The three-channel LLM semantic method improves over static Semantic Local BC
  in average performance and forgetting, but it has not yet matched Full BC.
- `stick-pull-v3` remains high variance and should be analyzed across seeds
  before making a strong method-level claim.

## 6. Method Definitions

### Fine-tuning

Sequential SAC with no explicit retention, no semantic memory, and no BC.

### ClonEx-SAC

Multi-head SAC with best-return historical-head exploration and Gaussian KL
actor cloning. The reimplementation follows the central method components but
uses PyTorch and the current Meta-World v3 protocol.

### Full BC

The same multi-head continual scaffold and Gaussian KL actor cloning loss, but
memory is made from complete successful old-task rollout states. It does not
semantic-filter those states. Full BC does not use old policies to collect new
task data by itself; best-return exploration is part of the shared continual
scaffold when enabled by the method configuration.

### Semantic Local BC

Static semantic memory over selected general labels. The current valid version
uses `task_aware_v3` and ClonEx-style Gaussian KL actor cloning, not MSE.

### Semantic Hybrid BC

Three-channel semantic memory:

- general channel: selected general labels and normalized weights;
- background channel: unselected general labels, ratio `0.2G`;
- task-specific channel: fine-grained labels from completed tasks, ratio
  `0.2G`.

With `llm_online`, GPT-5-mini controls only the general-channel label weights.
Background and task-specific channels are fixed safeguards. Future-task
trajectories and future-task task-specific memory are not available during
training task `k`.

### Gradient-aware BC

Experimental diagnostic only. It projects and caps BC gradients relative to SAC
gradients in `probe_stickpull_with_memory.py`. It is default-off and should not
be treated as a completed baseline.

## 7. Forward Transfer Rule

Formal aggregate forward transfer excludes task 0. The first task has no
previously learned task, so it cannot measure forward transfer.

The reporting plan is:

1. run independent single-task SAC for multiple seeds;
2. aggregate checkpoint-wise single-task curves into one mean curve per task;
3. compare each continual seed against the same aggregated single-task curves;
4. compute each continual seed's FT over tasks 1-9;
5. report mean and standard deviation over continual seeds.

Report raw FT, normalized FT, and area FT together. Normalized FT can be
unstable when the single-task baseline curve is near 1.0, so raw and area
curves should remain visible.

## 8. Immediate Formal Plan

First run three seeds, then decide whether to extend to five seeds:

- seeds: 0, 1, 2 initially;
- methods: `clonex_sac`, `full_bc`, `semantic_local_bc`,
  `semantic_hybrid_bc`;
- optional lower-bound: `fine_tuning`;
- budget: 500,000 steps per task;
- evaluation: every 20,000 steps, five stochastic episodes;
- protocol: `v3` + `cw10_v1`;
- metrics: AP, forgetting, raw FT, normalized FT, area FT, per-task tail-5
  success, and stick-pull successful-seed count.

The single-task baseline is currently a two-seed aggregate. Add more seeds
before the final statistical table because all formal FT claims depend on it.

## 9. Current Single-Task Commands

Seed 1:

```bash
cd /Users/xueyang/crl_cw

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

Seed 2:

```bash
cd /Users/xueyang/crl_cw

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
  --seed 2 \
  --device cpu \
  --output-dir outputs/single_task_baselines \
  --run-name cw10_v3_v1_500k_seed2
```

Aggregate command used after both runs completed:

```bash
PYTHONPATH=src .venv/bin/python scripts/aggregate_single_task_seeds.py \
  --batch-directories \
    outputs/single_task_baselines/cw10_v3_v1_500k_seed1 \
    outputs/single_task_baselines/cw10_v3_v1_500k_seed2 \
  --output-directory \
    outputs/single_task_baselines/cw10_v3_v1_500k_seeds1_2/aggregate \
  --tail-size 5
```

## 10. LLM Audit Requirements

For LLM runs:

- `.env` must contain `OPENAI_API_KEY`;
- `OPENAI_MODEL` or `--llm-controller-model` should be `gpt-5-mini`;
- prompt templates live in `prompts/llm_controller/`;
- exact prompts are stored in `<run_dir>/controller_prompts/`;
- decisions are stored in `<run_dir>/controller_decisions.json`;
- invalid JSON, invalid segments, empty memory, API errors, or missing fields
  fall back to the previous valid general-channel decision;
- logging failure terminates the run because unauditable LLM runs should not be
  used as evidence;
- source-task assertions must remain enabled.

## 11. Handover Checklist

Before launching formal runs:

- verify `.env` is present locally and not committed;
- verify run names are unique;
- keep `cw10_v1` and the stick-pull compatibility wrapper unchanged;
- use the same single-task aggregate baseline for all continual seeds;
- do not mix runs from different segmenter versions in one table;
- keep experimental gradient-aware results separate from formal baseline
  results.

After every completed run:

- preserve `config.json`, `evaluations.csv`, `task_summaries.csv`,
  `summary.json`, checkpoints, and LLM audit files;
- record reference-state counts and fallback rates for semantic methods;
- inspect stick-pull separately before interpreting aggregate AP;
- recompute the main table under the current FT rule.

## Gradient-Conflict Evidence Collection

Formal `full_bc` and `clonex_sac` runs now enable read-only gradient
diagnostics by default. This does not project, rescale, or otherwise modify
the gradients used for training. It samples one diagnostic point every 500 BC
updates and performs source-task decomposition only at existing evaluation
boundaries.

Each run writes incrementally to `gradient_diagnostics/`:

- `gradient_windows.csv`: SAC versus BC loss scale, shared-backbone and
  full-actor gradient norms, cosine, conflict mass, gradient decomposition,
  and pre-clip norm/clip scale;
- `gradient_layers.csv`: the same geometry for each actor-backbone layer;
- `gradient_task_pairs.csv`: source-old-task versus current-task conflict at
  evaluation boundaries;
- `summary.json`: run-wide means, extrema, conflict rate, and sample counts.

Rows include task names, global and task-local steps, evaluation index, raw
and weighted BC losses, and clipping settings. Memory source IDs are recovered
from task one-hot vectors; training aborts if current or future task data is
found in BC memory. A private diagnostic RNG and trajectory-equivalence test
ensure that logging does not alter parameter updates.

### Gradient-Handling Implementation Order

1. **Asymmetric PCGrad**: project only the harmful BC component against SAC.
   Test plasticity-priority and stability-priority variants first.
2. **CAGrad**: optimize a shared direction with explicit conflict aversion;
   this is the strongest candidate for improving the AP/forgetting/FT trade-off.
3. **MGDA**: add a Pareto-balanced convex-combination baseline. It may be
   conservative when gradients are noisy or badly scale-mismatched.
4. **ConFIG or a comparable modern optimizer**: add only after the first three
   establish that gradient geometry is the limiting factor.

Expected behavior: plasticity-priority PCGrad should improve new-task learning
and FT but may increase forgetting; stability-priority projection should do
the reverse. MGDA should be stable but slower. CAGrad has the best chance of
moving beyond Full BC's empirical Pareto point, but this requires multi-seed
evidence rather than a single favorable run.

### Implemented Gradient Methods

`full_bc_norm_balanced` uses the Full BC memory, exploration, KL target, and
training protocol without change. For shared-backbone gradients `g_sac` and
`g_bc`, it computes `s = min(1, r*||g_sac||/(||g_bc||+eps))`, with `r=1` by
default, then applies `s` to all BC actor gradients. The shared norm defines
cross-task interaction, while the common scale prevents large old-head
gradients from dominating global clipping.

`full_bc_pcgrad` first checks the shared-backbone dot product. If negative, it
replaces the shared BC component with
`g_bc - <g_bc,g_sac>/||g_sac||^2 * g_sac`. Task-specific heads are never
projected. It then applies the same norm-cap formula to the projected shared BC
gradient. Comparing these two methods therefore isolates conflict-direction
removal from gradient-scale control.

Both preserve the final Full BC update `(g_sac + g_bc_applied)/2` and existing
actor clipping. Diagnostics record the selected strategy, projection events,
raw and applied BC norms/cosines, and the applied scale.

## References

1. Wolczyk, M., et al. "Disentangling Transfer in Continual Reinforcement
   Learning." NeurIPS, 2022.
2. Wolczyk, M., et al. "Continual World: A Robotic Benchmark for Continual
   Reinforcement Learning." NeurIPS, 2021.
