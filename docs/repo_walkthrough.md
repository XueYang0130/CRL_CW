# CRL-CW Repository Walkthrough

This document explains the repository architecture, experiment lifecycle, method implementations, output artifacts, and handover workflow. Read `docs/research_log.md` first for the research hypothesis, current valid pilot results, and immediate multi-seed experiment plan.

## 1. What This Repository Does

`CRL-CW` is a PyTorch continual reinforcement-learning workspace for the Continual World CW10 task sequence on Meta-World v3. It supports:

1. independent single-task SAC baselines;
2. sequential continual-learning baselines;
3. ClonEx-style and full-trajectory behavior cloning;
4. semantic segmentation of successful old-policy rollouts;
5. rule-based and LLM-controlled semantic memory selection.

The current research path is:

```text
Meta-World v3 CW10
  -> multi-head SAC
  -> successful old-task rollouts
  -> task-aware semantic labels
  -> general + background + task-specific memory
  -> online LLM control of general-memory weights
  -> continual metrics and per-task analysis
```

The current protocol is Meta-World v3 with `--reward-function-version cw10_v1`.
For `stick-pull-v3`, `cw10_v1` uses the project-local `v1_compatible` reward
wrapper so the old reward structure is aligned to the v3 observation layout.

## 2. Top-Level Structure

```text
crl_cw/
|-- configs/       YAML presets and earlier selection manifests
|-- docs/          research log, handover notes, and analyses
|-- methods/       method registry entries and method defaults
|-- outputs/       generated experiment artifacts
|-- prompts/       versioned LLM prompts and task descriptions
|-- references/    external papers and third-party source snapshots
|-- scripts/       executable training and diagnostic entry points
|-- src/           reusable implementation code
|-- tests/         automated tests
|-- .env           local secrets; never commit or share
|-- .env.example   safe environment-variable template
|-- README.md
`-- requirements.txt
```

The key separation is:

- `methods/` describes what changes between algorithms;
- `src/` contains agents, environments, training, and evaluation;
- `scripts/` contains commands researchers run;
- `outputs/` contains generated evidence, not source code;
- `references/` is not imported by the active training pipeline.

`.venv/`, `.pytest_cache/`, `__pycache__/`, and `.DS_Store` are local generated artifacts.

## 3. Main Entry Point

The main executable is `scripts/run.py`. It supports:

| Mode | Meaning |
|---|---|
| `single` | Train SAC independently on one task. |
| `single-batch` | Run independent SAC on multiple CW tasks and aggregate baseline curves. |
| `continual` | Train one method through a sequential task stream. |

Execution follows:

```text
parse CLI and optional YAML
  -> load method preset from methods/
  -> validate arguments and mode compatibility
  -> create a unique run directory
  -> dispatch to single-task, batch, or continual runner
```

Configuration precedence is:

```text
explicit CLI > YAML config > method preset > parser default
```

`--llm-controller-model` overrides `OPENAI_MODEL` from the environment or `.env`. Run directories use `exist_ok=False`; an existing `--run-name` raises `FileExistsError`.

## 4. Environment Layer

The central file is `src/envs/cw_env.py`. `make_cw_env()` is the standard environment factory.

### CW10 order

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

`--sequence-task-count` truncates this sequence for pilots.

### Wrappers

- `RelaxedObservationSpace` preserves the full observation but replaces unreliable upstream bounds with an unbounded float32 Box.
- `TaskOneHotObservation` appends a task one-hot when requested by a method.
- `StickPullV1CompatibleReward` adapts the old v1 reward structure to the v3 observation layout without editing installed Meta-World code.

The active protocol is:

```text
--env-version v3
--reward-function-version cw10_v1
```

For ordinary tasks, `cw10_v1` resolves to v3 `reward_function_version="v1"`. For `stick-pull-v3`, it resolves to the project-local compatibility wrapper using:

```text
stick     = observation[4:7]
handle    = observation[11:14]
container = handle + [0.05, 0.0, 0.0]
```

Do not remove this special case when reproducing current results.

### Two meanings of task one-hot

- `task_conditioned` uses one shared output head and feeds the one-hot into the backbone.
- Multi-head methods append the one-hot but set `hide_task_id=True`; the backbone sees only physical observations and the one-hot selects the actor/critic head.

## 5. Method Registry

`methods/__init__.py` registers every method. Each method file exports a `MethodSpec` describing:

- valid modes;
- task-vector behavior;
- single-head or multi-head architecture;
- default hyperparameters;
- agent factory;
- reference-policy exploration;
- optional WSRL/JSRL guide mode.

| Method | Structure | Role |
|---|---|---|
| `single_task_baseline` | Independent single-head SAC | FT baseline curves. |
| `fine_tuning` | Single-head SAC, no retention | Lower continual baseline. |
| `task_conditioned` | Single head with task vector input | Conditioning ablation. |
| `packnet` | PackNet SAC agent | Auxiliary baseline. |
| `ssde` | Sparse descriptor-gated actor with SSDE task-transition mechanisms | Adapted SSDE reproduction. |
| `ssde_approx` | Importance ownership plus activation reset | Lightweight diagnostic, not formal SSDE. |
| `clonex_sac` | Multi-head, best-return exploration, KL cloning | Literature baseline. |
| `full_bc` | Complete successful-trajectory KL cloning | Current empirical oracle. |
| `semantic_local_bc` | Selected general semantic segments | Static semantic baseline. |
| `semantic_hybrid_bc` | General, background, and task-specific memory | Current proposed scaffold. |
| `adaptive_semantic_bc` | Pair/manifest adaptive selection | Earlier adaptive variant. |
| `general_task_specific_bc` | General plus task-specific side channel | Earlier two-channel variant. |
| `stage_aware_semantic_bc` | Rule-based stage schedule | Dynamic diagnostic baseline. |
| `wsrl_continual` | Fixed old-policy warm start | Exploratory transfer baseline. |
| `jsrl_continual` | Curriculum over guide horizon | Exploratory transfer baseline. |

Only methods listed as completed and valid in `research_log.md` should support current main-table claims.

To add a method, create `methods/<name>.py`, define `METHOD = MethodSpec(...)`, register it in `methods/__init__.py`, add an agent factory if needed, and write tests. Keep common task sequencing in the shared runner.

## 6. SAC Core

Core files:

```text
src/agents/sac_agent.py
src/agents/networks.py
src/agents/sac_losses.py
src/agents/replay_buffer.py
src/training/sac_trainer.py
```

### Network architecture

The default actor and critic use:

- four hidden layers of 256 units;
- LayerNorm plus `tanh` after the first layer;
- LeakyReLU with slope 0.2 after later layers;
- Xavier-uniform linear initialization;
- squashed Gaussian actor;
- twin Q critics;
- task-specific output heads for multi-head methods.

Actor log standard deviation is clamped to `[-20, 2]`. Actions are sampled with reparameterization, squashed by `tanh`, and scaled to the environment action bounds.

### Default training schedule

```text
replay capacity     1,000,000
batch size          128
random actions      first 10,000 environment steps
updates begin       after step 1,000
update schedule     50 updates every 50 environment steps
learning rate       1e-3
gamma               0.99
Polyak               0.995
target output std   0.089
episode horizon     200
```

Updates therefore begin from random replay at step 1,000 while random action collection continues until step 10,000.

### Online replay buffer

`ReplayBuffer` stores observation, action, reward, next observation, and true `terminated`. Time-limit `truncated` is not stored as terminal, preserving critic bootstrapping across horizon truncation.

Online replay is different from episodic or semantic BC memory. Continual methods may clear online replay at task changes without deleting BC memory.

## 7. Single-Task Training

`src/training/single_task_experiment.py` performs:

```text
create train/eval environments
  -> build single-task SAC
  -> allocate online replay
  -> train with SACTrainer
  -> evaluate periodically
  -> save best-success, best-return, and final checkpoints
  -> write evaluations and summary
```

Outputs:

```text
outputs/single_task/<run_name>/
|-- config.json
|-- evaluations.csv
|-- summary.json
`-- checkpoints/
    |-- best_success.pt
    |-- best_return.pt
    `-- final.pt
```

`single-batch` launches independent single-task subprocesses and aggregates their stochastic success curves into `aggregate/baseline_curves.json`. Continual runs need these curves to calculate forward transfer.

For formal reporting, run multiple pure single-task seeds, aggregate the
checkpoint-wise mean curve per task, and compare every continual seed against
the same aggregated baseline. The current two-seed reference is
`outputs/single_task_baselines/cw10_v3_v1_500k_seeds1_2/aggregate/baseline_curves.json`.
Rebuild it with `scripts/aggregate_single_task_seeds.py` when more seeds finish.

## 8. Continual Training

The shared runner is `src/training/continual_experiment.py`.

```text
seed RNGs and select task sequence
  -> build method-specific agent
  -> allocate online replay
  -> for each task:
       activate task/head
       optionally save old replay memory
       clear replay and reset optimizer when configured
       select old-policy exploration or guide
       install BC reference memory
       train and evaluate the active task
       collect successful reference rollouts
       update method memory
       save task checkpoint and summary
  -> evaluate all tasks at sequence end
  -> compute continual metrics
  -> write summary.json
```

ClonEx-style methods normally preserve shared network parameters, use a task-specific output head, clear online replay, rebuild optimizer state, and keep BC memory separately.

### Best-return exploration

`ExplorationHeadSelector` gives every prior actor head an episode, accumulates return/success statistics, and subsequently chooses the historical head with the highest mean return episode by episode during the initial exploration window.

The old head only generates actions. Replay observations carry the current task ID, so SAC updates the current task head.

## 9. Behavior-Cloning Methods

### ClonEx-SAC

Files:

```text
methods/clonex_sac.py
src/agents/clonex_sac_agent.py
```

Before old online replay is cleared, `on_task_start()` samples episodic observations and stores their old actor means and log standard deviations. Later actor updates combine SAC gradients with Gaussian KL cloning gradients. Critics are not cloned.

### Full BC

Files:

```text
methods/full_bc.py
src/agents/full_bc_agent.py
```

After each task, Full BC collects complete successful deterministic trajectories and freezes old-policy distribution targets on those states. New-task actor updates use the same Gaussian KL mechanism.

The main memory-source distinction is:

```text
ClonEx-SAC = sampled states from online replay
Full BC    = complete successful old-task rollout states
```

Both can use best-return historical-head exploration through the shared continual scaffold.

`FullBehaviorCloningSACAgent` also contains a default-off gradient-aware BC
diagnostic path used by `scripts/probe_stickpull_with_memory.py`. Keep that
separate from formal Full BC, ClonEx-SAC, and semantic baseline claims.

## 10. Semantic Segmentation

`src/training/semantic_segments.py` implements the active `task_aware_v3` scheme.

`extract_features()` derives TCP-object distance, object-target distance, normalized progress, object displacement/motion, gripper signals, grasp/in-place information, and success.

`TaskAwareV3Segmenter` returns both:

```text
SemanticLabel(general=<four-stage label>, task_specific=<fine event label>)
```

General labels are `approach`, `contact_or_alignment`, `manipulation`, and `finish_or_stabilize`. The complete 22-label task-specific vocabulary is documented in `research_log.md`.

Forward stage transitions can occur immediately; regressions require repeated evidence. This asymmetric debounce reduces rapid boundary oscillation. Main reference memory uses successful episodes only.

Segmentation diagnostics:

- `scripts/benchmark_segmentation_validity.py`
- `scripts/analyze_segmentation_boundaries.py`
- `scripts/diagnose_event_segments.py`

## 11. Three-Channel Memory

`build_hybrid_memory_from_store()` in the continual runner constructs reference memory.

### General channel

Selected general labels are resampled using explicit normalized weights. General budget `G` equals the total available states in selected general labels.

### Background channel

Background comes from unselected general labels. `--background-segment-ratio 0.2` contributes `0.2G` states.

### Task-specific channel

Successful states are independently indexed by their fine-grained labels. In `llm_online`, `--task-specific-segment-ratio 0.2` samples `0.2G` states from completed-task fine-grained memory.

Approximate final composition is:

```text
general        71.4%
background     14.3%
task-specific  14.3%
```

The current complete LLM method does not require the older task-specific general-segment manifest.

## 12. LLM Controller

Files:

```text
src/training/llm_controller.py
prompts/llm_controller/online_semantic_gate.txt
prompts/llm_controller/protocol.json
prompts/llm_controller/cw10_v3_task_descriptions.json
```

At each configured update, the prompt includes protocol, previous/current task profiles, current training stage, recent success/return curves, current general selection, general memory counts, and completed-task-specific counts.

GPT-5-mini selects and weights only the general channel. Background and task-specific channels are fixed safeguards.

Expected output:

```json
{
  "selected_segments": ["contact_or_alignment", "manipulation"],
  "priority": ["manipulation", "contact_or_alignment"],
  "weights": {
    "contact_or_alignment": 0.4,
    "manipulation": 0.6
  },
  "reason": "focus current bottleneck"
}
```

The code validates JSON shape, segment names, non-empty memory, priority membership, and finite positive weights. Weights are normalized to sum exactly to 1. Invalid output or API errors preserve the previous valid general configuration.

### Leakage protection

Task `k` is added to memory only after its training and reference collection finish. During task `k`, memory may contain only tasks `0...k-1`. Runtime assertions check source indices before every task and every LLM call.

Only previous/current task profiles are serialized. Future task descriptions and trajectories are not sent to the LLM.

### Audit trail

```text
<run_dir>/controller_prompts/       exact prompt files
<run_dir>/controller_decisions.json raw output, normalized decision, fallback, counts, sources
```

Logging failure terminates the run. `OPENAI_API_KEY` is loaded from `.env` and is not logged.

## 13. Evaluation and Metrics

Files:

```text
src/evaluation/evaluator.py
src/evaluation/continual_metrics.py
src/evaluation/summary.py
src/evaluation/single_task_baselines.py
```

An episode succeeds if `info["success"]` is true at any step.

- Average performance: mean final success over tasks.
- Forgetting: end-of-task success minus final success.
- Raw FT: mean continual active-task curve minus mean single-task curve.
- Normalized FT: raw FT divided by baseline remaining headroom.
- Area FT: normalized trapezoidal curve-area difference.

Aggregate FT excludes task 0 because no earlier task can transfer to the first
task. Task-0 FT values may still appear as per-task diagnostics, but CW10
main-table FT should average tasks 1-9.

Report raw and normalized FT together because normalized FT can be extreme when baseline headroom is small. FT is unavailable without protocol-matched `--baseline-curves`.

## 14. Output Artifacts

Standard continual output:

```text
outputs/cw10_continual/<run_name>/
|-- config.json
|-- evaluations.csv
|-- task_summaries.csv
|-- summary.json
|-- checkpoints/task_0.pt ... task_9.pt
|-- controller_prompts/          # LLM runs
`-- controller_decisions.json    # LLM runs
```

- `evaluations.csv`: active-task curves and final evaluations.
- `task_summaries.csv`: exploration source, updates, memory counts, semantic selection, and task statistics.
- `summary.json`: aggregate and per-task continual metrics.
- checkpoints: actor, critics, target critics, entropy parameters, optimizer
  state, step, and metadata.

Checkpoints intentionally exclude online replay and method-specific memory,
including Full BC episodic memory. They are evaluation and diagnostic snapshots,
not lossless continual-training resume points.

## 15. Diagnostic Scripts

| Script | Purpose |
|---|---|
| `diagnose_phase1.py` | Compare task-end and later checkpoints to localize forgetting. |
| `run_local_bc_pair.py` | Controlled old/new task BC pair experiments. |
| `benchmark_segmentation_validity.py` | Validate segment boundaries on successful rollouts. |
| `analyze_segmentation_boundaries.py` | Analyze stage order and switching. |
| `diagnose_event_segments.py` | Produce event-level summaries. |
| `build_selective_memory_manifest.py` | Build earlier selection manifests. |
| `build_single_task_matrix_manifest.py` | Prepare broad pairwise diagnostics. |
| `build_pre_stickpull_memory.py` | Construct pre-stick-pull memory. |
| `probe_stickpull_with_memory.py` | Probe memory sources for stick-pull. |
| `build_llm_segment_selection_package.py` | Package offline evidence for LLM review. |

These are diagnostic tools, not replacements for a complete CW10 run.

## 16. Tests

`tests/` covers environments, reward wrappers, task order, networks, replay semantics, SAC losses and updates, trainer schedules, checkpoints, metrics, semantic labels, hybrid sampling, CLI precedence, and LLM parsing/fallback behavior.

Run:

```bash
cd /Users/xueyang/crl_cw
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Recent targeted verification after the FT aggregation change:

```text
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_continual_metrics.py \
  tests/test_continual_run_validation.py -q

18 tests passed
```

Earlier full-suite verification reached 133 passing tests before later
iteration. Before committing or launching expensive formal runs after code
changes, run the complete test suite again.

## 17. Current Formal Experiments

The immediate plan is to build multi-seed single-task curves, run three
continual seeds first, compare ClonEx-SAC, Full BC, Semantic Local BC, and
Semantic Hybrid BC, then decide whether to extend to five seeds.

The proposed semantic method structure is:

```text
semantic_hybrid_bc
  + task_aware_v3 segmentation
  + GPT-5-mini general controller
  + background 0.2G
  + task-specific 0.2G
  + best-return exploration
  + Gaussian KL actor cloning
```

Do not add the older task-specific manifest to this experiment.

## 18. Handover Reading Order

1. `docs/research_log.md`
2. `docs/repo_walkthrough.md`
3. `scripts/run.py`
4. `methods/semantic_hybrid_bc.py`
5. `src/training/continual_experiment.py`
6. `src/training/semantic_segments.py`
7. `src/training/llm_controller.py` and `prompts/llm_controller/`
8. `src/agents/full_bc_agent.py`
9. `src/agents/sac_agent.py`, `networks.py`, and `sac_trainer.py`
10. `src/evaluation/continual_metrics.py` and relevant tests

## 19. Common Failure Modes

### Existing run directory

Use a unique run name. Do not casually delete old evidence.

### FT fields are null

Pass protocol-matched single-task baseline curves.

### Missing API key

Set `OPENAI_API_KEY` in `.env`; never paste it into commands or logs.

### Frequent fallback

Inspect `controller_decisions.json`, prompt files, available counts, and empty selections. Do not manually alter an active run.

### Empty semantic memory

Inspect `reference_success_episodes`, `collected_segment_states`, and memory counts before changing the algorithm.

### Leakage assertion

Never disable the source-task assertion to continue an experiment. Investigate and discard the affected run.

### Incompatible comparisons

Do not combine different environment versions, reward protocols, budgets, evaluation intervals, seed sets, or segmenter versions without explicit qualification.

## 20. Source-Control and Artifact Rules

- Never commit `.env` or API credentials.
- Keep external snapshots under `references/` unchanged unless intentionally updating them.
- Treat completed `outputs/` as experiment evidence.
- Do not commit `.venv/`, caches, or `__pycache__/`.
- Run tests and `git diff --check` before committing code.
- Record protocol-changing edits in `research_log.md` before comparing new and old results.

## 21. Gradient Diagnostics

`src/agents/gradient_diagnostics.py` is a read-only instrumentation layer used
by `FullBehaviorCloningSACAgent` and `ClonExSACAgent`. Both methods now share
the same Gaussian-KL cloning and actor-gradient combination implementation,
preventing baseline-specific diagnostic drift.

The instrumentation observes already-computed gradients and does not call an
optimizer, alter replay sampling, or consume the training RNG. Sparse global
and layer measurements use `--gradient-diagnostics-interval` (default 500).
At existing evaluation boundaries, a private RNG samples each old task to
measure its contribution against the current SAC gradient.

Formal Full BC and ClonEx presets enable diagnostics. Use
`--no-gradient-diagnostics` for a control and
`--gradient-diagnostics-source-batch-size` to change attribution batch size.
Files are appended at every evaluation boundary, so an interrupted run keeps
all completed windows. The three CSV files support time-series plots, layer
heatmaps, and old-task-by-current-task conflict matrices.

The first intervention methods are `full_bc_norm_balanced` and
`full_bc_pcgrad`. Both retain Full BC's successful-rollout memory and
best-return exploration. The former changes only BC gradient magnitude. The
latter additionally projects a conflicting BC shared-backbone gradient away
from the current SAC gradient and applies the cap to that projected gradient.
Because both use the same cap rule, their comparison tests gradient direction
without retaining an unnecessary pre-projection scale penalty.
