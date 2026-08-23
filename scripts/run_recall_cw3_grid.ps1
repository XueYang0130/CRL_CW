$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"
$python = ".\.venv\Scripts\python.exe"
$sequences = 0..7 | ForEach-Object { "cw3_$_" }
$seeds = 1..5
$common = @(
    "--mode", "continual",
    "--method", "recall",
    "--env-version", "v3",
    "--reward-function-version", "cw10_v1",
    "--steps-per-task", "500000",
    "--eval-every", "20000",
    "--stoch-eval-episodes", "5",
    "--det-eval-episodes", "0",
    "--best-return-eval-episodes", "10",
    "--batch-size", "128",
    "--episodic-memory-per-task", "10000",
    "--episodic-batch-size", "128",
    "--actor-cloning-coefficient", "10.0",
    "--gradient-clip-norm", "0.1",
    "--reset-buffer-on-task-change",
    "--reset-optimizer-on-task-change",
    "--gradient-diagnostics",
    "--gradient-diagnostics-interval", "500",
    "--gradient-diagnostics-source-batch-size", "128",
    "--baseline-curves", "outputs/single_task_baselines/cw10_v3_v1_500k_seeds1_2/aggregate/baseline_curves.json",
    "--device", "cpu",
    "--output-dir", "outputs/cw10_continual"
)

foreach ($sequence in $sequences) {
    foreach ($seed in $seeds) {
        $runName = "recall_${sequence}_v3_v1_500k_seed${seed}"
        $runDir = Join-Path "outputs\cw10_continual" $runName
        if (Test-Path (Join-Path $runDir "summary.json")) {
            Write-Host "[skip] completed $runName"
            continue
        }
        if (Test-Path $runDir) {
            throw "Incomplete directory exists: $runDir"
        }
        Write-Host "[start] $runName"
        & $python -u scripts\run.py @common `
            --task-sequence $sequence `
            --seed $seed `
            --run-name $runName
        if ($LASTEXITCODE -ne 0) {
            throw "Run failed with exit code $LASTEXITCODE: $runName"
        }
    }
}
