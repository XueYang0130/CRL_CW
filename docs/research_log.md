# Research Journal

**Working title:** *Remembering Without Restraining: Gradient-Aware Successful Replay for Continual Reinforcement Learning*

**Current reporting scope:** completed CW10 runs for seeds 1–5.

For a code-level guide, see [repo_walkthrough.md](repo_walkthrough.md).

## 1. Research Objective

This project studies the stability-plasticity problem in continual reinforcement
learning (CRL). When a new task arrives, the actor must remain plastic enough to
acquire new behavior while retaining policies learned for previous tasks.
Behavior cloning (BC) is effective for retention, but applying an unrestricted
BC gradient can dominate or oppose the current-task SAC gradient and delay new
skill acquisition.

The central hypothesis is that old-policy preservation and current-task learning
should be treated as two interacting optimization objectives rather than as a
single fixed weighted loss. The proposed approach therefore combines:

1. successful-state replay to retain behaviorally meaningful old-task states;
2. best-teacher relabelling to give every stored state a consistent target from
   the strongest saved actor snapshot for that task;
3. asymmetric gradient projection to protect the current SAC direction from a
   conflicting BC direction;
4. adaptive BC norm control to prevent either objective from dominating solely
   because of gradient scale;
5. an optional semantic critic route for tasks whose value structure is unlikely
   to transfer safely.

The main method is **Success Replay Best Adaptive PCGrad**. The critic-routing
variant, **Semantic-Routed Frozen Transfer PCGrad**, is currently treated as a
structured extension rather than a replacement for the main method.

## 2. Experimental Protocol

All results in this journal use the same protocol:

- benchmark: CW10;
- environment API: Meta-World v3 with Gymnasium;
- reward protocol: `cw10_v1`;
- observation protocol: full v3 observations with task identity appended for
  the multi-head architecture and hidden from the shared representation;
- `stick-pull-v3`: project-local v1-compatible reward alignment using
  `stick=obs[4:7]`, `handle=obs[11:14]`, and
  `container=handle+[0.05,0,0]`;
- horizon: 200 environment steps;
- training budget: 500,000 environment steps per task;
- evaluation frequency: every 20,000 steps;
- online evaluation: five stochastic episodes and zero deterministic episodes;
- final metric window: mean of the last five final evaluation points;
- replay batch size: 128;
- successful reference-memory capacity: 10,000 states per completed task;
- forward-transfer reference: checkpoint-wise aggregate of the completed
  independent SAC curves at
  `outputs/single_task_baselines/cw10_v3_v1_500k_seeds1_2/aggregate/baseline_curves.json`.

The CW10 sequence is:

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

## 3. Metrics

### Average performance

For each task, final performance is the mean success over the last five final
evaluation points. Average performance is the mean of those ten task values:

\[
\mathrm{AP}=\frac{1}{10}\sum_{i=1}^{10}p_i^{\mathrm{final}}.
\]

Higher is better. Task 0 is included because AP measures final competence over
the complete sequence.

### Average forgetting

For task \(i\), forgetting is its tail performance when task \(i\) finished
minus its final tail performance after CW10:

\[
F_i=p_i^{\mathrm{end}}-p_i^{\mathrm{final}}, \qquad
F=\frac{1}{10}\sum_i F_i.
\]

Lower is better. Positive values indicate forgetting; negative values indicate
that later training improved the task. Task 0 is included because it can be
forgotten after the first task transition.

### Raw forward transfer

For each task after the first, raw forward transfer is the difference between
the mean active-task learning curve in continual training and the corresponding
independent SAC learning curve. The reported aggregate averages tasks 1–9:

\[
\mathrm{FT}_{\mathrm{raw}}
=\frac{1}{9}\sum_{i=2}^{10}
\left(\overline{p_i^{\mathrm{CRL}}}-\overline{p_i^{\mathrm{single}}}\right).
\]

Higher is better. Task 0 is excluded because no previous task exists to provide
forward transfer.

### Area forward transfer

Area forward transfer compares the complete continual and single-task learning
curves using a normalized trapezoidal area. It is retained as a diagnostic,
while raw FT is used in the primary robust comparison because it is easier to
interpret and less sensitive to near-saturated baseline denominators.

## 4. Methods

### CloneX-SAC

CloneX-SAC is the primary baseline. It uses the same task-conditioned,
multi-head SAC scaffold and best-return historical-head exploration used by the
proposed method. At a task transition it samples broad states from the previous
task replay buffer, evaluates the old actor distribution on those states, and
adds the resulting `(state, mean, log_std, task_id)` items to cumulative
reference memory. Gaussian KL behavior cloning regularizes the actor throughout
subsequent tasks. SAC and BC use the standard joint actor update without PCGrad
or adaptive norm balancing.

### Success Replay Best Adaptive PCGrad

The main method changes both the memory distribution and the actor-gradient
integration.

**Successful-state reservoir.** During each task, successful episodes contribute
states to a reservoir with a capacity of 10,000 states. The states may come from
different stages of training, but all come from episodes that eventually
succeeded.

**Best-teacher relabelling.** Actor snapshots are evaluated during training.
Snapshots are ranked by evaluation success and then return. At task end, the
best snapshot is temporarily loaded and every reservoir state is relabelled with
that teacher's Gaussian output:

\[
(s,\mu_{\mathrm{best}}(s),\log\sigma_{\mathrm{best}}(s),\mathrm{task\_id}).
\]

This avoids imitating sampled action noise or mixing targets from many historical
versions of the actor.

**Asymmetric PCGrad.** Let \(g_{\mathrm{SAC}}\) and \(g_{\mathrm{BC}}\) be
the shared actor-backbone gradients. When their dot product is negative, only
the BC gradient is projected:

\[
g'_{\mathrm{BC}}
=g_{\mathrm{BC}}
-\frac{g_{\mathrm{BC}}^\top g_{\mathrm{SAC}}}
{\lVert g_{\mathrm{SAC}}\rVert^2+\epsilon}g_{\mathrm{SAC}}.
\]

The SAC direction is not projected because current-task acquisition has
priority.

**Adaptive combination.** After projection and the global BC norm cap, the
applied BC gradient is rescaled relative to the SAC gradient. Its target
shared-backbone norm is `0.2` times the SAC norm when the raw objectives are
compatible and `0.05` times the SAC norm when they conflict. The final shared
actor direction is:

\[
g_{\mathrm{actor}}=g_{\mathrm{SAC}}+\alpha_t g'_{\mathrm{BC}},
\]

where \(\alpha_t\) is computed online from the two gradient norms and is capped
so that the BC gradient is never amplified beyond its available magnitude.
The raw cloning coefficient remains `100.0`, but it does not directly determine
the final BC/SAC ratio after projection and adaptive scaling.

The critic remains the standard transferred multi-head SAC critic. BC updates
the shared actor backbone and the old task-specific actor heads; it does not
directly update the critic.

### Semantic-Routed Frozen Transfer PCGrad

This variant keeps the entire actor-side method above and changes only critic
handling. A frozen offline semantic route marks `stick-pull-v3` as the first
novel tool-mediated task; all other tasks use critic transfer.

For `stick-pull-v3`, the lifelong transfer critic is snapshotted and frozen. A
temporary freshly initialized critic supplies the SAC value gradients during
that task. At task end, the temporary critic is discarded and the unchanged
transfer critic is restored for later tasks. This tests whether a structurally
novel value problem should be prevented from overwriting the critic
representation carried across the sequence.

The routing file is
`configs/critic_routes/cw10_v3_schema_routes.json`. Routing is decided before
training from task semantics; it does not inspect evaluation outcomes from the
run and therefore does not use future performance as an oracle.

### Earlier memory-selection branches

**Full BC** uses complete successful old-task rollouts as reference memory and
applies the same Gaussian KL actor objective without semantic filtering. It
served as the successful-experience starting point for the later methods.

**Semantic Local BC** assigns each trajectory state one of four shared labels:
`approach`, `contact_or_alignment`, `manipulation`, or
`finish_or_stabilize`. A fixed subset of labels is retained for BC. The labels
have common names across tasks, but their geometric predicates are task aware.

**Semantic Hybrid BC** augments the four general labels with task-specific
labels and optional background memory. The `llm_online` branch uses the LLM as
a controller for the weights of available general segments; it does not create
future-task memory or expose future trajectories. The active task-aware semantic
ontology contains these 22 task-specific labels:

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

These semantic branches established the memory-selection problem, but they are
not included in the primary five-seed result table because the current main
method is the gradient-aware successful-replay method.

## 5. Five-Seed Results

### Robust aggregate comparison

The table reports the interquartile mean (IQM) and stratified-bootstrap 95%
confidence interval over seeds 1–5. The implementation follows the `rliable`
statistical workflow with 50,000 bootstrap resamples. For forgetting, lower is
better.

| Method | Average Performance IQM (95% CI) | Raw FT IQM (95% CI) | Forgetting IQM (95% CI) |
|---|---:|---:|---:|
| CloneX-SAC | 0.837 [0.684, 0.897] | 0.135 [0.029, 0.228] | 0.057 [0.023, 0.073] |
| **Adaptive PCGrad** | **0.879 [0.809, 0.904]** | **0.191 [0.121, 0.231]** | 0.021 [-0.033, 0.061] |
| Frozen Transfer PCGrad | 0.841 [0.780, 0.889] | 0.169 [0.108, 0.240] | **-0.001 [-0.027, 0.033]** |

Relative to CloneX-SAC, Adaptive PCGrad improves the IQM by `+0.041` in average
performance and `+0.055` in raw forward transfer, while reducing forgetting by
`0.036`. Frozen Transfer PCGrad provides the strongest retention result, but it
does not improve average performance over Adaptive PCGrad.

### Probability of improvement over CloneX-SAC

| Method | Average Performance | Raw FT | Lower Forgetting |
|---|---:|---:|---:|
| Adaptive PCGrad | 0.640 [0.240, 1.000] | 0.700 [0.320, 1.000] | 0.760 [0.400, 1.000] |
| Frozen Transfer PCGrad | 0.540 [0.160, 0.920] | 0.600 [0.200, 0.960] | **0.920 [0.680, 1.000]** |

The direction of the five-seed evidence favors Adaptive PCGrad on all three
metrics. However, the AP and raw-FT confidence intervals remain wide and overlap
the baseline; they should be presented as promising evidence rather than a
claim of definitive statistical superiority. The clearest current signal is
the retention advantage of the frozen-transfer variant.

### Per-seed main metrics

| Method | Seed | Average Performance | Raw FT | Forgetting | Area FT |
|---|---:|---:|---:|---:|---:|
| CloneX-SAC | 1 | 0.884 | 0.260 | 0.068 | 0.603 |
| CloneX-SAC | 2 | 0.824 | 0.083 | 0.044 | 0.171 |
| CloneX-SAC | 3 | 0.804 | 0.165 | 0.060 | 0.462 |
| CloneX-SAC | 4 | 0.904 | 0.157 | 0.012 | 0.364 |
| CloneX-SAC | 5 | 0.624 | 0.002 | 0.076 | -0.491 |
| Adaptive PCGrad | 1 | 0.784 | 0.100 | 0.028 | 0.011 |
| Adaptive PCGrad | 2 | 0.880 | 0.210 | -0.048 | 0.514 |
| Adaptive PCGrad | 3 | 0.860 | 0.165 | -0.004 | -0.851 |
| Adaptive PCGrad | 4 | 0.908 | 0.241 | 0.072 | 0.046 |
| Adaptive PCGrad | 5 | 0.896 | 0.196 | 0.040 | 0.521 |
| Frozen Transfer PCGrad | 1 | 0.828 | 0.164 | 0.020 | 0.326 |
| Frozen Transfer PCGrad | 2 | 0.764 | 0.094 | -0.032 | -0.329 |
| Frozen Transfer PCGrad | 3 | 0.812 | 0.135 | -0.008 | -0.649 |
| Frozen Transfer PCGrad | 4 | 0.892 | 0.207 | -0.016 | 0.308 |
| Frozen Transfer PCGrad | 5 | 0.884 | 0.256 | 0.040 | 0.172 |

### Final per-task success

Each value below is first computed as the final five-evaluation mean for one
run, then averaged over seeds 1–5.

| Task | CloneX-SAC | Adaptive PCGrad | Frozen Transfer PCGrad |
|---|---:|---:|---:|
| `hammer-v3` | 0.760 | 0.944 | **0.952** |
| `push-wall-v3` | 0.504 | **0.736** | 0.712 |
| `faucet-close-v3` | **1.000** | 0.992 | 0.976 |
| `push-back-v3` | **0.952** | 0.920 | 0.800 |
| `stick-pull-v3` | 0.280 | **0.488** | 0.400 |
| `handle-press-side-v3` | **1.000** | 0.992 | **1.000** |
| `push-v3` | 0.840 | 0.896 | **0.944** |
| `shelf-place-v3` | **0.744** | 0.688 | 0.576 |
| `window-close-v3` | **1.000** | **1.000** | **1.000** |
| `peg-unplug-side-v3` | **1.000** | **1.000** | **1.000** |

Adaptive PCGrad's aggregate gain is not produced by a single task. It improves
`hammer-v3`, `push-wall-v3`, `stick-pull-v3`, and `push-v3` relative to CloneX,
while remaining weaker on `push-back-v3` and `shelf-place-v3`. The frozen critic
route improves retention but does not consistently solve acquisition on
`stick-pull-v3` or `shelf-place-v3`; value routing alone is therefore not a
complete explanation of the remaining variance.

## 6. Mechanistic Findings

### BC dominance is dynamic rather than source-task specific

Gradient diagnostics show that no old source task is consistently harmful.
The same source can have compatible BC minibatches and conflicting BC
minibatches at different stages and in different seeds. A fixed rule such as
permanently removing one old task is therefore not supported.

The more repeatable failure pattern is a **BC-dominant regime**:

1. the current-task SAC backbone gradient becomes weak;
2. the raw BC/SAC norm ratio becomes very large;
3. old-policy regularization restricts shared-backbone movement;
4. exploration and critic learning improve slowly;
5. the SAC signal weakens further relative to BC.

Across existing diagnostics, the raw BC/SAC ratio is strongly associated with
the inverse SAC gradient norm. This means a very large ratio does not simply
mean that BC became stronger; it often means that the current task stopped
producing a useful actor gradient. Adaptive scaling prevents the raw cloning
coefficient from directly controlling the final update, but it cannot by itself
guarantee that SAC enters a productive behavioral branch.

### PCGrad primarily protects plasticity while replay protects retention

Successful reference memory supplies the retention target. PCGrad does not
create that target; it changes how the target is allowed to alter the shared
actor backbone. The asymmetric projection is most useful when BC and SAC
conflict, because it removes only the BC component that opposes current-task
improvement. In compatible regions, the method retains a larger BC contribution.

This division explains the current empirical pattern: Adaptive PCGrad improves
average performance and raw transfer while retaining a lower forgetting IQM
than CloneX. The result supports gradient-aware integration, but does not prove
that successful-only memory is universally superior to broad memory.

### Memory concentration remains a limitation

Successful-only states are informative but can concentrate the teacher target
around a narrow old-task behavior. This is especially risky when the new task
requires a new action phase not represented in old successful trajectories.
Broader memory can act as a less concentrated regularizer, even when many of its
states are not individually task-critical. Existing diagnostics motivate this
interpretation, but the five-seed primary table does not yet isolate memory
distribution from gradient integration.

### Critic transfer is task dependent

The frozen-transfer variant shows that preserving a long-term critic can reduce
forgetting. Its lower average performance shows that semantic novelty is not
sufficient to choose the best critic route. A reset critic may improve one task's
local value learning while discarding useful cross-task value structure, and a
frozen transfer critic can become mismatched to the actor after the routed task.
The current route should therefore be presented as a critic-side ablation, not
as the final routing solution.

## 7. Current Interpretation

The five-seed evidence supports the following conclusions:

1. Successful replay with best-teacher relabelling and adaptive asymmetric
   PCGrad is competitive with and currently stronger in IQM than CloneX-SAC on
   average performance, raw forward transfer, and forgetting.
2. The strongest gain is not merely retention. Adaptive PCGrad also improves
   the mean active-task learning curve relative to independent SAC.
3. CloneX-SAC remains a strong and variable baseline. Five seeds are enough to
   establish the current trend, but not enough for a definitive superiority
   claim on AP or FT.
4. Frozen semantic critic routing provides the best forgetting IQM, but trades
   away some plasticity and does not replace the actor-side main method.
5. `stick-pull-v3` and `shelf-place-v3` remain the most informative stress tests.
   They expose acquisition failures that cannot be explained by forgetting
   alone.

The paper's defensible contribution is therefore not "PCGrad always beats
CloneX." It is a structured account of how successful replay, teacher
consistency, and gradient-scale-aware conflict handling jointly improve the
stability-plasticity trade-off, supported by a competitive five-seed result and
mechanistic gradient diagnostics.

## 8. Method Evolution

The project progressed through the following stages:

1. **Fine-tuning and CloneX baselines:** established catastrophic forgetting and
   the strength of actor behavior cloning.
2. **Full successful-trajectory BC:** showed that old successful behavior can be
   retained without using CloneX's broad replay-state distribution.
3. **Semantic Local and Hybrid BC:** tested fixed general segments, task-specific
   labels, background memory, and online LLM weighting. These experiments showed
   that semantic selection alone can under-cover retention states and is highly
   sensitive to segmentation quality.
4. **Successful replay and best-teacher relabelling:** replaced post-task rollout
   collection with a successful-state reservoir and consistent teacher targets.
5. **Adaptive PCGrad:** added SAC-priority projection and norm-relative BC
   integration, producing the current main result.
6. **Critic routing:** tested reset, dual-critic, and frozen-transfer mechanisms.
   The frozen route improved retention but did not dominate the actor-side main
   method.
7. **Current diagnostics:** investigate memory coverage, BC cadence, layerwise
   conflict, hard conflict gates, guide policies, and optimistic n-step replay.
   These remain analyses or candidate ablations and are not part of the primary
   five-seed claim.

The semantic work remains relevant as a possible memory-value estimator, but
the present evidence favors using semantics to propose or interpret decisions
rather than allowing an LLM to directly control optimization without empirical
feedback.

## 9. Next Steps

### Required for the paper

1. Complete the statistical table with the same protocol and preserve all run
   directories, summaries, evaluation curves, and gradient diagnostics.
2. Run matched ablations that separately remove best-teacher relabelling,
   PCGrad projection, and adaptive norm scaling.
3. Compare successful-only and matched-capacity broad memory under the same
   integration rule to isolate memory distribution from optimization.
4. Report per-task acquisition curves for `stick-pull-v3` and
   `shelf-place-v3`, not only final success.
5. Add compute, memory-state count, and wall-clock comparisons with CloneX-SAC.

### Mechanistic analysis

1. Relate task-stage BC/SAC norm ratio and cosine conflict to acquisition delay.
2. Measure old-policy KL drift on fixed reference states across task
   checkpoints.
3. Test whether gradient concentration or memory feature rank predicts when
   successful-only replay becomes restrictive.
4. Distinguish causality from correlation with small checkpoint-based swaps of
   memory distribution and gradient integration, rather than repeatedly running
   complete CW10 variants.

### Reporting discipline

- Use the last-five-evaluations definition consistently for per-task final
  success.
- Include task 0 in average performance and forgetting.
- Exclude task 0 from aggregate forward transfer.
- Report IQM, 95% bootstrap confidence intervals, and probability of
  improvement alongside ordinary means.
- Do not describe overlapping five-seed AP or FT intervals as conclusive
  statistical superiority.
- Keep exploratory methods separate from the primary method table until their
  protocol is complete.

## 10. Reproducibility Pointers

- Main entry point: `scripts/run.py`
- Continual loop: `src/training/continual_experiment.py`
- Main method registration: `methods/success_replay_best_adaptive_pcgrad.py`
- Frozen critic route: `methods/semantic_routed_frozen_transfer_pcgrad.py`
- Actor BC and gradient integration: `src/agents/full_bc_agent.py`
- Frozen-transfer critic implementation:
  `src/agents/semantic_routed_dual_critic_agent.py`
- Metric definitions: `src/evaluation/continual_metrics.py`
- Robust statistics: `scripts/analyze_rliable_cw10.py`
- Five-seed analysis artifacts:
  `outputs/analysis/rliable_cw10_seeds1_5/`

The current primary experimental statement is: under the fixed CW10 v3/v1,
500k-per-task protocol, seeds 1–5 show that successful replay with consistent
best-teacher targets and adaptive SAC-priority PCGrad improves the IQM
stability-plasticity trade-off relative to CloneX-SAC, while semantic frozen
critic transfer further reduces forgetting but does not improve overall
performance.
