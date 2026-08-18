# SSDE Reproduction Status

`ssde` is a mechanism-level PyTorch reproduction of the official SSDE code,
adapted to this repository's common CW10-v3 evaluation protocol. It is not a
claim of bitwise equivalence to the original JAX implementation.

## Reproduced mechanisms

- Official CW10 task-description strings encoded by
  `all-MiniLM-L12-v2` into 384-dimensional descriptors.
- Four 1024-unit actor layers with fixed and task-random Lasso dictionaries.
- Binary task gates formed by the union of fixed and random sparse codes.
- Frozen previously allocated actor connections and beta-scaled overlap reuse.
- Uniform-previous-policy exploration during the initial 10,000 task steps.
- Optional 20,000-update task-start distillation toward `N(0, I)` on the
  previous task replay buffer; enabled by default as in the official run.
- Fresh critic initialization and optimizer reset at every task boundary.
- Input-sensitivity checks every 80,000 environment steps and restoration of
  dormant available units to their post-distillation task-start values.
- Per-task logging of gate density, overlap ratio, beta, distillation updates,
  and reactivated neurons in `task_summaries.csv`.

## Deliberate protocol adaptations

- Meta-World v3 environments with the repository's `cw10_v1` reward mapping,
  including the existing v3/v1-compatible observation realignment.
- 500,000 environment steps per task for comparison with local baselines,
  instead of the original 1,000,000-step v1 protocol.
- The repository's task-conditioned multi-head actor/critic interface and
  common evaluation/checkpoint format. The official launch uses a shared
  output policy by default even though its code contains a multi-head path.
- The repository's common SAC critic implementation and reward handling.

These adaptations keep environment budget, metrics, and task interfaces
comparable to the existing baselines. Results should be described as an
adapted reproduction, not as direct reproduction of the paper's reported
numbers.

The older `ssde_approx` method omits descriptor sparse coding and task-start
distillation. It is intentionally retained under a distinct name and must not
be reported as the SSDE baseline.
