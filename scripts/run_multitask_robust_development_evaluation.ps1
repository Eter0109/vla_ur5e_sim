param(
    [string]$Python = "C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe",
    [string]$Checkpoint = "outputs\multitask_robust\smolvla_30k_lazy_bs2_final\seed1000\checkpoints\030000\pretrained_model",
    [string]$OutputRoot = "outputs\multitask_robust\evaluation_lazy",
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$CheckpointFull = [System.IO.Path]::GetFullPath((Join-Path $Root $Checkpoint))
$OutputFull = [System.IO.Path]::GetFullPath((Join-Path $Root $OutputRoot))
$PushRoot = Join-Path $Root "data\lerobot\multitask_robust_push_1500"
$PickRoot = Join-Path $Root "data\lerobot\multitask_robust_pick_place_1500"

while (-not (Test-Path $CheckpointFull)) {
    Start-Sleep -Seconds $PollSeconds
}
New-Item -ItemType Directory -Force -Path $OutputFull | Out-Null

function Invoke-DevelopmentEval([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Development evaluation failed with exit code $LASTEXITCODE"
    }
}

Invoke-DevelopmentEval @(
    (Join-Path $PSScriptRoot "run_push_vla_only_benchmark.py"),
    "--checkpoint", $CheckpointFull, "--dataset-root", $PushRoot,
    "--repo-id", "local/multitask_robust_push_1500",
    "--manifest", (Join-Path $Root "configs\benchmarks\push_robust_development_nominal_v1.json"),
    "--episodes", "50", "--replan-steps", "4", "--temporal-decay", "0.75",
    "--policy-seed", "1000", "--samples-per-plan", "1",
    "--output", (Join-Path $OutputFull "push_nominal.json"),
    "--overwrite-development"
)
Invoke-DevelopmentEval @(
    (Join-Path $PSScriptRoot "run_push_vla_only_benchmark.py"),
    "--checkpoint", $CheckpointFull, "--dataset-root", $PushRoot,
    "--repo-id", "local/multitask_robust_push_1500",
    "--manifest", (Join-Path $Root "configs\benchmarks\push_robust_development_randomized_v1.json"),
    "--episodes", "50", "--replan-steps", "4", "--temporal-decay", "0.75",
    "--policy-seed", "1000", "--samples-per-plan", "1",
    "--output", (Join-Path $OutputFull "push_randomized.json"),
    "--overwrite-development"
)
Invoke-DevelopmentEval @(
    (Join-Path $PSScriptRoot "run_pick_place_vla_only.py"),
    "--checkpoint", $CheckpointFull, "--dataset-root", $PickRoot,
    "--repo-id", "local/multitask_robust_pick_place_1500",
    "--manifest", (Join-Path $Root "configs\benchmarks\pick_place_robust_development_nominal_v1.json"),
    "--episodes", "50", "--temporal-ensemble-decay", "0.75", "--replan-steps", "4",
    "--samples-per-plan", "2", "--control-mode", "vla_action_calibrated",
    "--closed-negative-y-gain", "1.8", "--policy-seed", "1000",
    "--output", (Join-Path $OutputFull "pick_place_nominal.json"),
    "--overwrite-development"
)
Invoke-DevelopmentEval @(
    (Join-Path $PSScriptRoot "run_pick_place_vla_only.py"),
    "--checkpoint", $CheckpointFull, "--dataset-root", $PickRoot,
    "--repo-id", "local/multitask_robust_pick_place_1500",
    "--manifest", (Join-Path $Root "configs\benchmarks\pick_place_robust_development_randomized_v1.json"),
    "--episodes", "50", "--temporal-ensemble-decay", "0.75", "--replan-steps", "4",
    "--samples-per-plan", "2", "--control-mode", "vla_action_calibrated",
    "--closed-negative-y-gain", "1.8", "--policy-seed", "1000",
    "--output", (Join-Path $OutputFull "pick_place_randomized.json"),
    "--overwrite-development"
)

function Get-SuccessRate([string]$Path) {
    $Result = Get-Content $Path -Raw | ConvertFrom-Json
    if ($null -ne $Result.summary -and $null -ne $Result.summary.success_rate) {
        return [double]$Result.summary.success_rate
    }
    $Rows = $Result
    $Successes = 0
    $Episodes = 0
    foreach ($Row in $Rows) {
        $Episodes++
        if ($Row.success) { $Successes++ }
    }
    return ($Successes / $Episodes)
}

$Summary = [ordered]@{
    checkpoint = $CheckpointFull
    push_nominal = Get-SuccessRate (Join-Path $OutputFull "push_nominal.json")
    push_randomized = Get-SuccessRate (Join-Path $OutputFull "push_randomized.json")
    pick_place_nominal = Get-SuccessRate (Join-Path $OutputFull "pick_place_nominal.json")
    pick_place_randomized = Get-SuccessRate (Join-Path $OutputFull "pick_place_randomized.json")
}
$Summary | ConvertTo-Json | Set-Content (Join-Path $OutputFull "development_summary.json")

& $Python (Join-Path $PSScriptRoot "verify_target80_development.py") `
    --checkpoint $CheckpointFull `
    --push-nominal (Join-Path $OutputFull "push_nominal.json") `
    --push-randomized (Join-Path $OutputFull "push_randomized.json") `
    --pick-nominal (Join-Path $OutputFull "pick_place_nominal.json") `
    --pick-randomized (Join-Path $OutputFull "pick_place_randomized.json") `
    --output (Join-Path $OutputFull "target80_gate.json")
exit $LASTEXITCODE
