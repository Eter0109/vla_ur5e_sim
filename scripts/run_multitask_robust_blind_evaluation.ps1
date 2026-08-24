param(
    [string]$Python = "C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe",
    [string]$Checkpoint = "outputs\multitask_robust\smolvla_30k_lazy_bs2_final\seed1000\checkpoints\030000\pretrained_model",
    [string]$DevelopmentRoot = "outputs\multitask_robust\evaluation_lazy_bs2_final",
    [string]$OutputRoot = "outputs\multitask_robust\blind_evaluation_lazy_bs2_final",
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$CheckpointFull = [System.IO.Path]::GetFullPath((Join-Path $Root $Checkpoint))
$DevelopmentFull = [System.IO.Path]::GetFullPath((Join-Path $Root $DevelopmentRoot))
$OutputFull = [System.IO.Path]::GetFullPath((Join-Path $Root $OutputRoot))
$GatePath = Join-Path $DevelopmentFull "target80_gate.json"
$PushRoot = Join-Path $Root "data\lerobot\multitask_robust_push_1500"
$PickRoot = Join-Path $Root "data\lerobot\multitask_robust_pick_place_1500"

while (-not (Test-Path (Join-Path $DevelopmentFull "pick_place_randomized.json"))) {
    Start-Sleep -Seconds $PollSeconds
}
& $Python (Join-Path $PSScriptRoot "verify_target80_development.py") `
    --checkpoint $CheckpointFull `
    --push-nominal (Join-Path $DevelopmentFull "push_nominal.json") `
    --push-randomized (Join-Path $DevelopmentFull "push_randomized.json") `
    --pick-nominal (Join-Path $DevelopmentFull "pick_place_nominal.json") `
    --pick-randomized (Join-Path $DevelopmentFull "pick_place_randomized.json") `
    --output $GatePath
if ($LASTEXITCODE -ne 0) {
    throw "Strict target-80 development gate failed; blind evaluation remains locked"
}
New-Item -ItemType Directory -Force -Path $OutputFull | Out-Null

& $Python (Join-Path $PSScriptRoot "run_push_vla_only_benchmark.py") `
    --checkpoint $CheckpointFull --dataset-root $PushRoot --repo-id "local/multitask_robust_push_1500" `
    --manifest (Join-Path $Root "configs\benchmarks\push_robust_blind_v1.json") `
    --episodes 100 --replan-steps 4 --temporal-decay 0.75 --policy-seed 1000 `
    --samples-per-plan 1 --output (Join-Path $OutputFull "push_blind.json")
if ($LASTEXITCODE -ne 0) { throw "Push blind evaluation failed" }

& $Python (Join-Path $PSScriptRoot "run_pick_place_vla_only.py") `
    --checkpoint $CheckpointFull --dataset-root $PickRoot --repo-id "local/multitask_robust_pick_place_1500" `
    --manifest (Join-Path $Root "configs\benchmarks\pick_place_robust_blind_v1.json") `
    --episodes 100 --temporal-ensemble-decay 0.75 --replan-steps 4 `
    --samples-per-plan 2 --control-mode vla_action_calibrated `
    --closed-negative-y-gain 1.8 --policy-seed 1000 `
    --output (Join-Path $OutputFull "pick_place_blind.json")
if ($LASTEXITCODE -ne 0) { throw "PickPlace blind evaluation failed" }

function Get-SuccessRate([string]$Path) {
    $Result = Get-Content $Path -Raw | ConvertFrom-Json
    if ($null -ne $Result.summary -and $null -ne $Result.summary.success_rate) {
        return [double]$Result.summary.success_rate
    }
    $Rows = @($Result)
    return (@($Rows | Where-Object { $_.success }).Count / $Rows.Count)
}

$Summary = [ordered]@{
    checkpoint = $CheckpointFull
    push = Get-SuccessRate (Join-Path $OutputFull "push_blind.json")
    pick_place = Get-SuccessRate (Join-Path $OutputFull "pick_place_blind.json")
    target_met = $false
}
$Summary.target_met = $Summary.push -ge 0.80 -and $Summary.pick_place -ge 0.80
$Summary | ConvertTo-Json | Set-Content (Join-Path $OutputFull "blind_summary.json")
if (-not $Summary.target_met) {
    throw "Blind target failed: $($Summary | ConvertTo-Json -Compress)"
}
