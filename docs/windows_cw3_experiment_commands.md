# Windows CW3 Experiment Commands

This document defines the Windows run for the eight official three-task CW3
sequences used by the RECALL comparison. It runs seeds `4` and `5` for:

- `clonex_sac`
- `success_replay_best_adaptive_pcgrad`
- `semantic_routed_frozen_transfer_pcgrad`
- `recall`

The protocol is the project protocol: Meta-World `v3`, project reward
`cw10_v1` (including the repository's `stick-pull-v3` v1-compatible reward
alignment), 500,000 environment steps per task, stochastic evaluation every
20,000 steps with 5 episodes, CPU execution, and gradient diagnostics.

## 1. Setup

Run PowerShell from the repository root after pulling `main`:

```powershell
cd C:\path\to\crl_cw
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"
$Python = ".\.venv\Scripts\python.exe"
```

## 2. Single-task SAC reference curves

These curves are needed for normalized/raw forward transfer. They are not
needed to compute final per-task success or forgetting. Run one matched
single-task batch for each seed before the continual grid:

```powershell
foreach ($Seed in 4, 5) {
    & $Python -u scripts\run.py `
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
        --output-dir outputs\single_task_baselines `
        --run-name "cw10_v3_v1_500k_seed$Seed"
    if ($LASTEXITCODE -ne 0) { throw "single-task baseline failed for seed $Seed" }
}
```

Each baseline file is then used only with the matching continual seed:

```powershell
$BaselineCurves = "outputs\single_task_baselines\cw10_v3_v1_500k_seed$Seed\aggregate\baseline_curves.json"
```

The ten-task batch is sufficient because the union of the eight CW3 sequences
contains all ten CW10 environments. The loader reorders curves by task name,
so repeated tasks in `cw3_5` are handled as repeated sequence positions.

## 3. Full CW3 grid

The following block runs seed 4 completely, then seed 5 completely. Each seed
contains 8 sequences x 4 methods = 32 runs. Existing completed runs are
skipped only when `summary.json` exists; an incomplete directory is treated as
an error rather than overwritten.

```powershell
$Sequences = @("cw3_0", "cw3_1", "cw3_2", "cw3_3", "cw3_4", "cw3_5", "cw3_6", "cw3_7")
$Methods = @(
    "clonex_sac",
    "success_replay_best_adaptive_pcgrad",
    "semantic_routed_frozen_transfer_pcgrad",
    "recall"
)

foreach ($Seed in 4, 5) {
    $BaselineCurves = "outputs\single_task_baselines\cw10_v3_v1_500k_seed$Seed\aggregate\baseline_curves.json"
    if (-not (Test-Path $BaselineCurves)) {
        throw "Missing matched SAC baseline: $BaselineCurves"
    }

    foreach ($Sequence in $Sequences) {
        foreach ($Method in $Methods) {
            $RunName = "${Method}_${Sequence}_v3_v1_500k_seed${Seed}"
            $RunDir = Join-Path "outputs\cw10_continual" $RunName
            if (Test-Path (Join-Path $RunDir "summary.json")) {
                Write-Host "[skip] $RunName"
                continue
            }
            if (Test-Path $RunDir) {
                throw "Incomplete output directory exists: $RunDir"
            }

            $Common = @(
                "--mode", "continual",
                "--method", $Method,
                "--task-sequence", $Sequence,
                "--env-version", "v3",
                "--reward-function-version", "cw10_v1",
                "--steps-per-task", "500000",
                "--eval-every", "20000",
                "--stoch-eval-episodes", "5",
                "--det-eval-episodes", "0",
                "--best-return-eval-episodes", "10",
                "--episodic-memory-per-task", "10000",
                "--episodic-batch-size", "128",
                "--gradient-diagnostics",
                "--gradient-diagnostics-interval", "500",
                "--gradient-diagnostics-source-batch-size", "128",
                "--reset-buffer-on-task-change",
                "--reset-optimizer-on-task-change",
                "--baseline-curves", $BaselineCurves,
                "--seed", "$Seed",
                "--device", "cpu",
                "--output-dir", "outputs\cw10_continual",
                "--run-name", $RunName
            )

            if ($Method -eq "recall") {
                $Common += @(
                    "--actor-cloning-coefficient", "10.0",
                    "--recall-value-reg-coef", "1.0",
                    "--gradient-clip-norm", "0.1"
                )
            }
            if ($Method -eq "semantic_routed_frozen_transfer_pcgrad") {
                $RouteManifest = "configs\critic_routes\cw3_v3_${Sequence}_routes.json"
                $Common += @("--critic-route-manifest", $RouteManifest)
            }

            Write-Host "[start] $RunName"
            & $Python -u scripts\run.py @Common
            if ($LASTEXITCODE -ne 0) { throw "continual run failed: $RunName" }
        }
    }
}
```

The CW3 route manifests are sequence-specific. In particular, `cw3_5` uses
transfer for its first `stick-pull-v3` position and reset for the repeated
third position; this respects the implementation rule that the first task
must establish the persistent transfer critic.

## 4. Inspecting results

```powershell
Get-ChildItem outputs\cw10_continual -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 20 Name, LastWriteTime

Get-Content outputs\cw10_continual\recall_cw3_0_v3_v1_500k_seed4\summary.json
```

For the eight-sequence aggregate, use the existing
`scripts\analyze_recall_cw3.py` for RECALL and an equivalent seed/sequence
filter for the other methods. The script uses 10,000 bootstrap repetitions and
95% intervals by default.

## 5. Expected cost

The continual grid is 64 runs x 3 tasks x 500,000 steps = 96 million
environment steps, plus two 10-task single-task batches (10 million steps per
seed). Running sequentially is safer for CPU memory; parallel processes can
exhaust the machine's CPU and RAM.
