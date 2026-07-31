## Local-BC Ablation Summary

### Context

This note summarizes the first event-based local behavior-cloning (local BC) ablations conducted after the phase-1 temporal diagnostic and the event-based drift analysis.

The working hypothesis was:

> If forgetting is locally structured, then selectively preserving only the most interference-prone trajectory segments may provide a better stability-plasticity tradeoff than full-trajectory behavior cloning.

The experiments used:

- CW10 continual fine-tuning checkpoint source:
  - `outputs/cw10_continual/fine_tuning_cw10_v3_v1_500k_seed0`
- pairwise continual transfer setup:
  - initialize from an old task-end checkpoint
  - continue training on one later task
  - compare:
    - `no_bc`
    - `full_bc`
    - `local_bc`

All pairwise ablations used:

- `100k` further training steps on the later task
- `5` stochastic evaluation episodes every `20k` steps
- local BC coefficient `10.0`

## Pair 1: `faucet-close-v3 -> push-back-v3`

### Event-diagnostic motivation

The event-based diagnostic suggested that this pair shows strong early-to-middle drift in:

- `approach`
- `contact_or_alignment`
- `manipulation`

### Results

#### Baseline comparisons

- `no_bc`
  - old final success: `0.0`
  - new final success: `0.5`

- `full_bc` with `4000` reference states
  - old final success: `1.0`
  - new final success: `1.0`

#### First local selection attempt

- `local_bc` on `contact_or_alignment + manipulation`
  - reference states: `173`
  - old final success: `0.0`
  - new final success: `0.7`

This showed that the initial local subset was too narrow to preserve the old task.

#### Sample-count control

- `full_bc` with `173` reference states
  - old final success: `1.0`
  - new final success: `0.1`

This established that the previous comparison was confounded by reference-memory size:

- `full_bc_173` strongly favored stability
- `local_bc_173` favored plasticity

#### Expanded local selection

- `local_bc` on `approach + contact_or_alignment + manipulation`
  - reference states: `798`
  - old final success: `1.0`
  - new final success: `0.3`

This showed that including `approach` is necessary for preserving the old task in this pair.

#### Fairer comparison against full BC

- `full_bc` with `798` reference states
  - old final success: `1.0`
  - new final success: `0.8`

### Interpretation

This pair provides three important conclusions:

1. forgetting is indeed locally structured;
2. the first local subset was too narrow;
3. even after expanding the local subset to the full early-to-middle interaction region, equally sized `full_bc` still outperformed the current event-selected `local_bc`.

Therefore:

> For `faucet-close-v3 -> push-back-v3`, the current event-based local BC does not outperform equally sized full-trajectory BC.

This pair should currently be treated as a **hard case / counterexample**, not as a positive result for local BC.

## Pair 2: `faucet-close-v3 -> handle-press-side-v3`

### Event-diagnostic motivation

The event-based diagnostic suggested that drift in this pair is relatively weak in the middle region and stronger near:

- `finish_or_stabilize`

### Results

- `no_bc`
  - old final success: `0.2`
  - new final success: `1.0`

- `local_bc` on `finish_or_stabilize`
  - old final success: `0.0`
  - new final success: `1.0`

- `full_bc`
  - old final success: `0.6`
  - new final success: `1.0`

### Interpretation

This pair shows that:

- preserving only the late event segment is insufficient;
- the old task appears to require a broader behavioral context than the final segment alone;
- again, `full_bc` is stronger than the current event-selected `local_bc`.

Therefore:

> For `faucet-close-v3 -> handle-press-side-v3`, the current late-segment local BC also fails to beat full-trajectory BC.

## Main Takeaway

The current evidence supports the following refined conclusion:

1. **Local forgetting structure is real.**
   - This was established by the temporal and event-based diagnostics.

2. **Naive event-based local BC is not yet sufficient.**
   - The existence of local drift does not automatically imply that a coarse event subset is the right preservation unit.

3. **The next method should not simply use fixed event labels as the preservation mask.**
   - A stronger selection mechanism is needed.

## Updated Research Direction

The most defensible next-step claim is now:

> Continual forgetting appears locally structured, but converting that structure into a better preservation mechanism requires a more selective and more semantically grounded segment-selection policy than the current hand-crafted event bins.

This motivates moving from:

- fixed event-segment local BC

to:

- **segment scoring**
- **selective memory construction**
- eventually **semantic / interference-aware segment labeling**

## Practical Recommendation

The current project should treat the completed local BC ablations as:

- a successful diagnostic stress test of the proposal;
- a negative result for naive fixed-event local BC;
- a justification for building the next prototype around **selective segment scoring**, not around more brute-force event-bin ablations.
