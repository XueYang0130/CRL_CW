# Windows Experiment Commands

This document uses Windows PowerShell. Run every command from the repository
root. The formal protocol is Meta-World v3 with the `cw10_v1` reward mapping,
500,000 steps per task, and five stochastic evaluation episodes every 20,000
steps. The repository-local `cw10_v1` wrapper automatically applies the
v1-compatible observation alignment required by `stick-pull-v3`.

## 1. Environment setup

```powershell
cd C:\path\to\crl_cw

py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

$env:PYTHONPATH = "src"
```

Keep the computer awake through Windows Settings > System > Power & battery >
Screen and sleep while long experiments are running. PowerShell does not use
the macOS `caffeinate` command.

## 2. Shared CW10 protocol

Define the common arguments once in each new PowerShell session:

```powershell
$Python = ".\.venv\Scripts\python.exe"
$BaselineCurves = "outputs/single_task_baselines/cw10_v3_v1_500k_seeds1_2/aggregate/baseline_curves.json"
$Seed = 0

$CW10Args = @(
    "--mode", "continual",
    "--env-version", "v3",
    "--reward-function-version", "cw10_v1",
    "--steps-per-task", "500000",
    "--eval-every", "20000",
    "--stoch-eval-episodes", "5",
    "--det-eval-episodes", "0",
    "--episodic-memory-per-task", "10000",
    "--baseline-curves", $BaselineCurves,
    "--seed", "$Seed",
    "--device", "cpu"
)
```

The commands print evaluation results directly in the terminal. Each run also
writes checkpoints, evaluation curves, task summaries, gradient diagnostics,
and `summary.json` under `outputs/cw10_continual/<run-name>/`.

## 3. Single-task baseline

Run all ten tasks independently for one seed:

```powershell
& $Python scripts/run.py `
    --mode single-batch `
    --method single_task_baseline `
    --env-version v3 `
    --reward-function-version cw10_v1 `
    --num-tasks 10 `
    --steps-per-task 500000 `
    --eval-every 20000 `
    --stoch-eval-episodes 5 `
    --det-eval-episodes 0 `
    --seed $Seed `
    --device cpu `
    --output-dir outputs/single_task_baselines `
    --run-name "cw10_v3_v1_500k_seed$Seed"
```

The aggregated baseline file referenced by continual runs must already exist.
Do not substitute a single continual run for the single-task curves when
computing forward transfer.

## 4. Formal continual baselines

### Naive fine-tuning

```powershell
& $Python scripts/run.py @CW10Args `
    --method fine_tuning `
    --run-name "fine_tuning_cw10_v3_v1_500k_seed$Seed"
```

### CloneX-SAC

```powershell
& $Python scripts/run.py @CW10Args `
    --method clonex_sac `
    --run-name "clonex_sac_cw10_v3_v1_500k_seed$Seed"
```

### Full behavior cloning

```powershell
& $Python scripts/run.py @CW10Args `
    --method full_bc `
    --run-name "full_bc_cw10_v3_v1_500k_seed$Seed"
```

### Full BC with PCGrad

```powershell
& $Python scripts/run.py @CW10Args `
    --method full_bc_pcgrad `
    --run-name "full_bc_pcgrad_cw10_v3_v1_500k_seed$Seed"
```

## 5. Success-replay methods

### Best-teacher adaptive PCGrad

```powershell
& $Python scripts/run.py @CW10Args `
    --method success_replay_best_adaptive_pcgrad `
    --run-name "success_replay_best_adaptive_pcgrad_cw10_v3_v1_500k_seed$Seed"
```

### Original semantic-routed dual critic

This version updates a background transfer critic during reset-routed tasks.
It is retained as an ablation.

```powershell
& $Python scripts/run.py @CW10Args `
    --method semantic_routed_dual_critic_pcgrad `
    --run-name "semantic_routed_dual_critic_pcgrad_cw10_v3_v1_500k_seed$Seed"
```

### Semantic-routed frozen transfer critic

This is the current critic-routing method. The route manifest resets the
active critic for `stick-pull-v3`, freezes the persistent transfer critic
during that task, and restores it for later tasks. Adam state is reset at task
boundaries, consistently with CloneX and the success-replay protocol.

```powershell
& $Python scripts/run.py @CW10Args `
    --method semantic_routed_frozen_transfer_pcgrad `
    --run-name "semantic_routed_frozen_transfer_pcgrad_cw10_v3_v1_500k_seed$Seed"
```

The stick-pull task summary should report `critic_route=reset` and
`background_critic_updates=0`.

## 6. Semantic and schedule ablations

### Static semantic local BC

```powershell
& $Python scripts/run.py @CW10Args `
    --method semantic_local_bc `
    --run-name "semantic_local_bc_cw10_v3_v1_500k_seed$Seed"
```

### Semantic hybrid BC

```powershell
& $Python scripts/run.py @CW10Args `
    --method semantic_hybrid_bc `
    --run-name "semantic_hybrid_bc_cw10_v3_v1_500k_seed$Seed"
```

### Progress-gated adaptive PCGrad

```powershell
& $Python scripts/run.py @CW10Args `
    --method success_replay_best_progress_pcgrad `
    --run-name "success_replay_best_progress_pcgrad_cw10_v3_v1_500k_seed$Seed"
```

### LLM-scheduled adaptive PCGrad

Configure the API key only for methods that call the online LLM controller:

```powershell
$env:OPENAI_API_KEY = "YOUR_API_KEY"
```

Do not commit the API key or a populated `.env` file.

```powershell
& $Python scripts/run.py @CW10Args `
    --method success_replay_best_llm_schedule_pcgrad `
    --run-name "success_replay_best_llm_schedule_pcgrad_cw10_v3_v1_500k_seed$Seed"
```

## 7. Running another seed

Open a new PowerShell window, redefine `$Python`, `$BaselineCurves`, `$Seed`,
and `$CW10Args`, then run the required method. Use a unique run name because
the runner deliberately refuses to overwrite an existing result directory.

For example:

```powershell
$Seed = 1
$SeedIndex = [Array]::IndexOf($CW10Args, "--seed")
$CW10Args[$SeedIndex + 1] = "$Seed"

& $Python scripts/run.py @CW10Args `
    --method semantic_routed_frozen_transfer_pcgrad `
    --run-name "semantic_routed_frozen_transfer_pcgrad_cw10_v3_v1_500k_seed$Seed"
```

## 8. Monitoring generated results

List the most recently modified run directories:

```powershell
Get-ChildItem outputs\cw10_continual -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 10 Name, LastWriteTime
```

Inspect a completed summary:

```powershell
Get-Content outputs\cw10_continual\RUN_NAME\summary.json
```
