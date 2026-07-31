## Event-Segmentation Diagnostic Status

### What is already established

The temporal phase-1 diagnostic already showed that forgetting in CW10 is not uniform over the whole trajectory. For several well-learned tasks, action drift differs across trajectory phases, and the drift pattern depends on which later task was learned.

This supports the core proposal claim that continual interference may be locally structured rather than globally uniform.

### What the new event-based diagnostic adds

We now have a second diagnostic script:

- `scripts/diagnose_event_segments.py`

This script does not rely on equal 20% temporal bins. Instead, it re-rolls successful trajectories from an old task-end checkpoint and assigns each step to a lightweight event segment using task geometry:

- `approach`
- `contact_or_alignment`
- `manipulation`
- `finish_or_stabilize`

The current implementation supports four core tasks:

- `faucet-close-v3`
- `push-wall-v3`
- `window-close-v3`
- `handle-press-side-v3`

For each old task and each later checkpoint, the script:

1. verifies the old task-end checkpoint still solves the old task;
2. evaluates the later checkpoint on the same old task;
3. replays successful old-task trajectories under the old policy;
4. compares old and later actions on the same old states;
5. aggregates drift by event segment instead of only by time.

### Smoke-test result

A smoke test on `faucet-close-v3` from `seed0` comparing against:

- `push-back-v3`
- `handle-press-side-v3`

already shows interpretable structure.

Observed pattern:

- `faucet-close-v3 -> push-back-v3`
  - high drift already appears in `approach`, `contact_or_alignment`, and `manipulation`
  - this is consistent with early-stage rewriting of the old task behavior

- `faucet-close-v3 -> handle-press-side-v3`
  - drift is much lower in `manipulation` than in the late `finish_or_stabilize` phase
  - this is consistent with partial preservation of intermediate interaction but strong distortion near later execution / stabilization

This is exactly the kind of evidence we needed before moving beyond temporal bins.

## Recommended Immediate Plan

### Step 2A: finish the event-based diagnostic on a small core set

Run the event-segmentation diagnostic on the most informative task pairs:

- `faucet-close-v3 -> push-back-v3`
- `faucet-close-v3 -> handle-press-side-v3`
- `push-wall-v3 -> push-v3`
- `window-close-v3 -> peg-unplug-side-v3` as a lower-drift contrast if useful

Priority should be on `seed0` first, then one additional seed for stability.

Goal:

- confirm that event segments are more interpretable than equal temporal phases;
- identify which event regions are consistently fragile.

### Step 2B: convert the diagnosis into an intervention

If Step 2A confirms that the high-drift region is localized, the next rational move is:

- **local behavior cloning without LLM segmentation**

This should come before LLM-based semantic labeling.

Reason:

- it isolates whether **local preservation itself** helps;
- it avoids introducing two new variables at once;
- it gives a cleaner causal test of the proposal.

The minimal intervention should be:

1. store a small replay set from the old task-end checkpoint;
2. keep only steps from the selected high-value local segment;
3. apply a local BC loss while learning the later task;
4. compare against:
   - no BC
   - full-trajectory BC
   - local segment BC

### Step 2C: only after that, decide whether LLM segmentation is needed

If local BC helps but the hand-designed event segmentation is too crude, then it becomes rational to ask whether LLM-based segment labeling gives a better selection mechanism.

At that point the question becomes:

> Does semantic labeling improve segment selection beyond simple geometry-based events?

That is a much cleaner research question than introducing LLM labeling immediately.

## Operational Recommendation

The most rational near-term workflow is:

1. complete the event-based diagnostic on the core task pairs;
2. identify one strong positive pair and one contrast pair;
3. implement the smallest local-BC intervention on those pairs;
4. only then decide whether to expand into LLM-based semantic segmentation.

In short:

- temporal diagnostic: done
- event diagnostic: now ready
- local BC ablation: next
- LLM segmentation: later, only if justified
