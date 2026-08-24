param(
    [string]$Python = "C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe",
    [int]$Seed = 1000
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$PushRoot = Join-Path $Root "data\lerobot\multitask_robust_push_1500"
$PickRoot = Join-Path $Root "data\lerobot\multitask_robust_pick_place_1500"
$Output = Join-Path $Root "outputs\multitask_robust\smolvla_30k\seed$Seed"
$Report = Join-Path $Root "outputs\multitask_robust\pipeline.log"

function Write-PipelineLog([string]$Message) {
    $Line = "$(Get-Date -Format o) $Message"
    $Line | Tee-Object -FilePath $Report -Append
}

Write-PipelineLog "waiting_for_pick_place_collection"
while (-not (Test-Path (Join-Path $PickRoot "collection.complete"))) {
    Start-Sleep -Seconds 60
}

Write-PipelineLog "auditing_multitask_datasets"
& $Python (Join-Path $PSScriptRoot "audit_multitask_datasets.py") `
    --push-root $PushRoot --push-repo-id "local/multitask_robust_push_1500" `
    --pick-place-root $PickRoot --pick-place-repo-id "local/multitask_robust_pick_place_1500"
if ($LASTEXITCODE -ne 0) { throw "Multitask dataset audit failed" }

Write-PipelineLog "starting_training"
& (Join-Path $PSScriptRoot "train_multitask_robust_smolvla.ps1") -Python $Python -Output $Output -Seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Unified training failed" }
$Checkpoint = Join-Path $Output "checkpoints\030000\pretrained_model"
if (-not (Test-Path $Checkpoint)) { throw "Expected training checkpoint is missing: $Checkpoint" }

$EvalRoot = Join-Path $Root "outputs\multitask_robust\evaluation"
New-Item -ItemType Directory -Force -Path $EvalRoot | Out-Null
Write-PipelineLog "starting_development_evaluation"
& $Python (Join-Path $PSScriptRoot "run_push_vla_only_benchmark.py") `
    --checkpoint $Checkpoint --dataset-root $PushRoot --repo-id "local/multitask_robust_push_1500" `
    --manifest (Join-Path $Root "configs\benchmarks\push_robust_development_nominal_v1.json") `
    --episodes 50 --output (Join-Path $EvalRoot "push_nominal.json") --overwrite-development
& $Python (Join-Path $PSScriptRoot "run_push_vla_only_benchmark.py") `
    --checkpoint $Checkpoint --dataset-root $PushRoot --repo-id "local/multitask_robust_push_1500" `
    --manifest (Join-Path $Root "configs\benchmarks\push_robust_development_randomized_v1.json") `
    --episodes 50 --output (Join-Path $EvalRoot "push_randomized.json") --overwrite-development
& $Python (Join-Path $PSScriptRoot "run_pick_place_vla_only.py") `
    --checkpoint $Checkpoint --dataset-root $PickRoot --repo-id "local/multitask_robust_pick_place_1500" `
    --manifest (Join-Path $Root "configs\benchmarks\pick_place_robust_development_nominal_v1.json") `
    --episodes 50 --output (Join-Path $EvalRoot "pick_place_nominal.json") --overwrite-development
& $Python (Join-Path $PSScriptRoot "run_pick_place_vla_only.py") `
    --checkpoint $Checkpoint --dataset-root $PickRoot --repo-id "local/multitask_robust_pick_place_1500" `
    --manifest (Join-Path $Root "configs\benchmarks\pick_place_robust_development_randomized_v1.json") `
    --episodes 50 --output (Join-Path $EvalRoot "pick_place_randomized.json") --overwrite-development
if ($LASTEXITCODE -ne 0) { throw "Development evaluation failed" }

function Get-SuccessRate([string]$Path) {
    $Rows = Get-Content $Path -Raw | ConvertFrom-Json
    return (@($Rows | Where-Object { $_.success }).Count / @($Rows).Count)
}
$Gates = @{
    push_nominal = Get-SuccessRate (Join-Path $EvalRoot "push_nominal.json")
    push_randomized = Get-SuccessRate (Join-Path $EvalRoot "push_randomized.json")
    pick_place_nominal = Get-SuccessRate (Join-Path $EvalRoot "pick_place_nominal.json")
    pick_place_randomized = Get-SuccessRate (Join-Path $EvalRoot "pick_place_randomized.json")
}
$Gates | ConvertTo-Json | Set-Content (Join-Path $EvalRoot "development_gate.json")
if ($Gates.push_nominal -lt 0.80 -or $Gates.pick_place_nominal -lt 0.80 -or $Gates.push_randomized -lt 0.75 -or $Gates.pick_place_randomized -lt 0.75) {
    throw "Development gates failed: $($Gates | ConvertTo-Json -Compress)"
}

Write-PipelineLog "starting_frozen_blind_evaluation"
& $Python (Join-Path $PSScriptRoot "run_push_vla_only_benchmark.py") `
    --checkpoint $Checkpoint --dataset-root $PushRoot --repo-id "local/multitask_robust_push_1500" `
    --manifest (Join-Path $Root "configs\benchmarks\push_robust_blind_v1.json") `
    --episodes 100 --output (Join-Path $EvalRoot "push_blind.json")
& $Python (Join-Path $PSScriptRoot "run_pick_place_vla_only.py") `
    --checkpoint $Checkpoint --dataset-root $PickRoot --repo-id "local/multitask_robust_pick_place_1500" `
    --manifest (Join-Path $Root "configs\benchmarks\pick_place_robust_blind_v1.json") `
    --episodes 100 --output (Join-Path $EvalRoot "pick_place_blind.json")
if ($LASTEXITCODE -ne 0) { throw "Blind evaluation failed" }

$Blind = @{
    push = Get-SuccessRate (Join-Path $EvalRoot "push_blind.json")
    pick_place = Get-SuccessRate (Join-Path $EvalRoot "pick_place_blind.json")
}
$Blind | ConvertTo-Json | Set-Content (Join-Path $EvalRoot "blind_summary.json")
if ($Blind.push -lt 0.70 -or $Blind.pick_place -lt 0.70) {
    throw "Blind target failed: $($Blind | ConvertTo-Json -Compress)"
}
Write-PipelineLog "goal_achieved push=$($Blind.push) pick_place=$($Blind.pick_place)"
