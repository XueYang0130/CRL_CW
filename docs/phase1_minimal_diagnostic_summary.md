# Phase-1 Minimal Diagnostic Summary

## Scope

This note summarizes the current phase-1 diagnostic study for the `fine_tuning` continual SAC baseline on CW10 under the current project protocol:

- Meta-World `v3`
- `cw10_v1` reward protocol
- `500k` steps per task
- `25` stochastic evaluation episodes for checkpoint comparisons
- deterministic action comparison on fixed reference states

The goal of this diagnostic is not to propose a new algorithm yet. It is to test whether continual forgetting exhibits **within-trajectory heterogeneity**, i.e. whether different parts of a previously learned task drift differently after subsequent task training.

## Diagnostic Setup

For one previously learned task `T_i`, we:

1. load the checkpoint saved immediately after finishing `T_i`;
2. evaluate that checkpoint on `T_i` to confirm the task was actually learned;
3. collect successful reference trajectories in `T_i` using the task-end policy;
4. replay the saved observations through a later policy checkpoint;
5. compute action drift on the **same old states**:

`drift_t = || a_old(s_t) - a_cmp(s_t) ||_2`

6. split each successful episode into 5 equal temporal phases;
7. aggregate per-phase mean action drift.

We ran two complementary variants:

- `task-end vs final checkpoint`
- `task-end vs subsequent checkpoints`

The first tests whether local drift exists at all. The second tests when forgetting starts to emerge and whether different later tasks induce different local drift patterns.

## Main Findings From `task-end vs final`

### 1. Strong task-level forgetting exists

Several tasks were learned well at task-end and degraded heavily by the end of the full continual sequence:

- `hammer-v3`
- `push-wall-v3`
- `faucet-close-v3`
- `push-back-v3`
- `handle-press-side-v3`
- `window-close-v3`

This establishes a clean setting for studying local forgetting.

### 2. Phase-level drift is not uniform

Across multiple tasks, the mean action drift differs substantially across temporal phases. The strongest and most stable examples are:

- `window-close-v3`
- `push-wall-v3`
- `faucet-close-v3`
- `handle-press-side-v3`

This provides direct evidence that forgetting is not well described as a uniform task-level degradation.

### 3. The phenomenon is reasonably stable across seeds

Across `seed0`, `seed1`, and `seed21`, the tasks above repeatedly show non-trivial phase gaps. In contrast:

- `peg-unplug-side-v3` remains learned at the end of training and shows near-zero drift

This makes `peg-unplug-side-v3` a useful negative control.

### 4. Not every task is suitable for this analysis

Some tasks do not currently provide reliable evidence for forgetting because they were not stably learned at task-end:

- `stick-pull-v3`
- `shelf-place-v3`
- `push-v3` is borderline and should be treated as a secondary case rather than a core one

## Main Findings From `task-end vs subsequent checkpoints`

The subsequent-checkpoint analysis is more informative than the final-only comparison, because it reveals **when** forgetting emerges.

### 1. Forgetting often appears immediately after the next task

For several core tasks, performance drops sharply after only one subsequent task checkpoint. This suggests forgetting is often not a late cumulative-only effect; it can be introduced immediately after a task switch.

### 2. Different later tasks induce different local drift patterns

This is the strongest motivation for a local interference perspective.

Examples:

- `faucet-close-v3 -> push-back-v3`
  - strongest drift occurs early
  - interpretation: the later task appears to rewrite early approach / positioning / initial interaction behavior

- `faucet-close-v3 -> handle-press-side-v3`
  - drift is much smaller in the first phases and much larger in later phases
  - interpretation: early approach behavior is relatively preserved, while later execution behavior is heavily altered

- `push-wall-v3 -> push-v3`
  - early drift is smaller, middle/later drift is larger
  - interpretation: some early pushing-related approach behavior may still transfer, while later task-specific manipulation geometry diverges

These cases indicate that the **same old task** can be damaged in **different local regions** depending on which later task is learned.

## Current Interpretation

At this point, the diagnostics already support the following claim:

> Continual forgetting in CW10 is not only task-level. It often has a structured local form, and the location of strongest drift depends on the interaction between the old task and the later task.

This does **not** yet prove that the current 5-way temporal segmentation is semantically correct. It only shows that even a coarse temporal partition already exposes non-uniform forgetting patterns.

## Important Limitations

### 1. The current phases are temporally defined, not semantically defined

The current `phase_0 ... phase_4` segmentation is based on equal episode partitions:

- first 20%
- second 20%
- ...
- final 20%

Therefore, the current study does **not** yet establish true skill boundaries.

### 2. Action drift is not yet the same as harmful drift

We currently know that actions changed, but not yet whether every local change is performance-harmful according to the old task value function or other grounded proxies.

### 3. The study currently isolates existence and timing, not full causal mechanism

We are not yet using gradient conflict. This is intentional: the current phase focuses on establishing that local interference structure exists before moving to stronger mechanism claims.

## Recommended Immediate Next Step

The most rational next step is **not** to jump directly to LLM segmentation or gradient conflict.

Instead:

1. keep the current phase-1 results as the empirical motivation;
2. design a lightweight **event-based segmentation** scheme for a few core tasks;
3. check whether event-based segments explain the high-drift regions more naturally than equal temporal bins.

Recommended core tasks for this step:

- `window-close-v3`
- `push-wall-v3`
- `faucet-close-v3`
- `handle-press-side-v3`

Recommended illustrative task pairs:

- `faucet-close-v3 -> push-back-v3`
- `faucet-close-v3 -> handle-press-side-v3`
- `push-wall-v3 -> push-v3`

## Decision Point After Event-Based Segmentation

If event-based segmentation produces cleaner and more interpretable local forgetting patterns, the next methodological decision should be:

### First test local behavior cloning without LLM

This is the most rational path.

Reason:

- it isolates the value of **local selection itself**
- it avoids adding a second new variable too early
- it answers a stronger methodological question:

> Is segment-level selective preservation already useful before semantic labeling is introduced?

Only after this is answered should the project decide whether LLM-based segment labeling adds meaningful value beyond event-based local replay / cloning.

## Recommended Research Order

1. phase-1 temporal diagnostic: completed
2. event-based segmentation diagnostic: next
3. local behavior cloning with non-LLM segments: should come before LLM segmentation
4. LLM-assisted segment labeling: only if step 3 shows local preservation is effective and semantics appears to provide additional structure

