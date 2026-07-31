# Research Log

## Current Research Goal

The current project studies continual reinforcement learning on CW10 [2] under a unified Meta-World v3 protocol. The main goal is to develop a semantic local interference-aware continual RL method that can preserve useful old-task knowledge while maintaining strong new-task learning and forward transfer.

At this stage, the research focus is:

1. establish strong and fair continual learning baselines under the current protocol,
2. identify whether full trajectory preservation or selective semantic preservation is more effective,
3. use the baseline comparison to guide the design of the next semantic method.

Current protocol:

- Meta-World v3
- reward protocol: `cw10_v1`
- `500k` steps per task
- evaluation every `20k` steps
- `5` stochastic evaluation episodes

## Core Hypothesis

The central hypothesis of this project is that forgetting, forward transfer, and final performance degradation in continual reinforcement learning do not arise uniformly throughout the full training process of a new task. Instead, they are likely concentrated on specific behavioral substructures.

We assume that old and new tasks contain both:

- cooperative structure, where previous knowledge can accelerate new-task learning,
- conflicting structure, where previous knowledge interferes with adaptation and should instead be selectively stabilized.

Under this view, old and new knowledge exist in a mixed cooperative-competitive relationship rather than a purely supportive or purely harmful one.

This motivates a more selective alternative to coarse behavior-cloning-based preservation. Existing BC-style continual RL methods, such as ClonEx-SAC [1], preserve past knowledge at a relatively coarse granularity. While this can improve retention, it does not explicitly distinguish between transferable and interfering components of old behavior. As a result, it may preserve segments that are unnecessary on shared structure and overly constrain learning on conflicting structure.

Our main idea is therefore to separate cooperative and conflicting parts of cross-task experience. To do this, we plan to use an LLM in two roles:

1. semantic labeling:
   annotate old-task rollouts into interpretable segments such as `approach`, `contact`, `alignment`, `manipulation`, or `stabilization`, using task descriptions and rollout summaries;
2. adaptive gating:
   select which segment types should be preserved through behavior cloning when a new task arrives.

We further hypothesize that this selection should not be fixed. The old experience segments that are useful for stabilizing old knowledge may not be the same as the segments that are useful for helping new-task learning, and the relevant subset may change during the course of training. Therefore, segment preservation may need to be dynamic both across tasks and within a single task's training process.

## Completed Baselines

### 1. Fine-tuning

Fine-tuning is the naïve continual SAC baseline. A single continual learner is trained across the task sequence without any explicit mechanism for retention or transfer.

Role in this project:

- lower-bound continual baseline
- reference point for catastrophic forgetting

### 2. ClonEx-SAC Reimplementation

This baseline is a PyTorch reimplementation of the core ClonEx-SAC mechanism. It uses:

- multi-head SAC
- best-return exploration across previous heads
- episodic actor distillation through KL-based cloning

Role in this project:

- primary continual RL baseline from the ClonEx line
- reference method for transfer-oriented continual learning

### 3. Full-trajectory Behavior Cloning (`full_bc`)

This baseline uses the same general continual scaffold as the ClonEx-style setup, but stores successful full-trajectory reference states after each task and applies actor distillation on this full reference memory.

Role in this project:

- strong memory-preservation baseline
- bridge between ClonEx-style distillation and future semantic selection

### 4. Semantic Local Behavior Cloning (`semantic_local_bc`)

This method keeps the same continual SAC and actor-distillation scaffold as `full_bc`, but replaces full successful trajectory preservation with selective preservation of semantically labeled trajectory segments. In the current completed version, the preserved segments are fixed and include `contact_or_alignment` and `manipulation`. No LLM is involved yet in this version.

Role in this project:

- first completed selective semantic memory baseline
- proof-of-concept for local rather than full behavior preservation

### 4a. Semantic Local Behavior Cloning with Segment Fix (`semantic_local_bc_segfix`)

After the first `semantic_local_bc` run, we discovered that the formal segmentation logic for `window-close-v3` was flawed: the task was incorrectly mapped to a saturated progress state at reference-memory construction time, which led to missing semantic reference states for that task. We then repaired the task-aware semantic feature extraction and reran the same CW10 protocol with the same fixed segment set.

Role in this project:

- corrected semantic local BC baseline
- cleaner reference point for subsequent semantic-memory ablations

### 5. Adaptive Semantic Behavior Cloning (`adaptive_semantic_bc`)

This is the new task-adaptive scaffold built on top of `semantic_local_bc`. It keeps the same continual SAC and actor-distillation backbone, but replaces the fixed segment set with a manifest-driven pairwise selection mechanism. For each old-task to new-task transition, the method can now load a different semantic segment subset from a JSON manifest.

Current status:

- implementation completed,
- unit tests passed,
- continual smoke test passed,
- current selection is manifest-driven placeholder logic rather than online LLM selection.

## Main Results

| Method | Average Performance | Average Forgetting | Forward Transfer | Area Forward Transfer | Average Return |
|---|---:|---:|---:|---:|---:|
| fine_tuning | 0.112 | 0.536 | -0.685 | -0.822 | 31402.638 |
| clonex_sac | 0.732 | 0.028 | -0.267 | -0.381 | 231822.926 |
| full_bc | 0.904 | 0.048 | 0.239 | 0.210 | 255417.002 |
| semantic_local_bc | 0.516 | 0.300 | -0.078 | -0.176 | 69627.303 |
| semantic_local_bc_segfix | 0.680 | 0.260 | -0.526 | -0.805 | 122173.466 |

## Segmentation Fix Impact

The segmentation repair produced a meaningful improvement over the original `semantic_local_bc` run:

| Metric | semantic_local_bc | semantic_local_bc_segfix |
|---|---:|---:|
| Average Performance | 0.516 | 0.680 |
| Average Forgetting | 0.300 | 0.260 |
| Forward Transfer | -0.078 | -0.526 |
| Raw Forward Transfer | 0.028 | 0.077 |
| Area Forward Transfer | -0.176 | -0.805 |
| Average Return | 69627.303 | 122173.466 |

The most important qualitative change is that `window-close-v3`, which previously contributed zero semantic reference states, now contributes nonzero selected semantic memory under the repaired task-aware segmentation logic.

## Current Conclusion

The current results support the following conclusions.

1. Fine-tuning is clearly insufficient under the current protocol. Forgetting is severe and forward transfer is strongly negative.

2. The ClonEx-SAC reimplementation remains a meaningful continual baseline. It substantially improves retention over fine-tuning, but its forward transfer advantage is not very strong under the current protocol.

3. `full_bc` is currently the strongest completed baseline. It achieves the best overall performance and the best transfer behavior among the three completed runs.

4. Full memory preservation is currently stronger than the present ClonEx-style episodic replay baseline under this protocol.

5. `semantic_local_bc` clearly outperforms naïve fine-tuning, which supports the basic feasibility of selective semantic preservation.

6. However, the current fixed-segment version of `semantic_local_bc` does not yet match either `clonex_sac` or `full_bc` in final overall performance. Its forward transfer is less negative than `clonex_sac`, but its retention remains too weak on several later tasks.

## Current Interpretation

The completed `semantic_local_bc` run suggests that selective semantic preservation is a meaningful direction, but fixed segment selection is too rigid.

The main interpretation at this stage is:

- selective preservation is viable,
- fixed segment types can support strong learning on many tasks,
- but the same fixed segment set is insufficient for robust retention across the full sequence,
- especially for harder or more structurally distinct tasks.

## Next Steps

The next work items are:

1. run the first full CW10 experiment for `adaptive_semantic_bc`,
2. replace placeholder manifest choices with LLM-based task-adaptive segment selection,
3. explore stage-adaptive gating so that the preserved segment types can change during the course of new-task training,
4. test whether adaptive semantic selection can close the gap to `full_bc` while keeping the efficiency and interpretability advantages of selective memory.

[1]Wolczyk, Maciej, et al. "Disentangling transfer in continual reinforcement learning." Advances in Neural Information Processing Systems 35 (2022): 6304-6317.
[2]Wolczyk, Maciej, et al. "Continual world: A robotic benchmark for continual reinforcement learning." Advances in Neural Information Processing Systems. 2021.
