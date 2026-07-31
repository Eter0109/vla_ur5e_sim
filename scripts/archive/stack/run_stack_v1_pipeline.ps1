# Archived Stack v1 pipeline; paths reflect the historical repository layout.
param(
    [string]$Python = "python",
    [string]$Dataset = "data\lerobot\stack_v1_3000",
    [string]$Manifest = "configs\benchmarks\stack_collect_v1.json",
    [string]$Tokenizer = "outputs\smolvla_ablation_c_15k_seed1000\checkpoints\015000\pretrained_model",
    [string]$OutputRoot = "outputs\stack_v1",
    [ValidateSet(1000, 1001, 1002)]
    [int]$Seed = 1000,
    [int]$WaitForCollectorPid = 0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ($WaitForCollectorPid -gt 0) {
    if (Get-Process -Id $WaitForCollectorPid -ErrorAction SilentlyContinue) {
        Write-Host "Waiting for collection process $WaitForCollectorPid..."
        Wait-Process -Id $WaitForCollectorPid -ErrorAction Stop
    }
    else {
        Write-Host "Collection process $WaitForCollectorPid already exited; auditing now."
    }
}

& $Python scripts/audit_stack_dataset.py `
    --root $Dataset `
    --repo-id local/ur5e_stack_v1 `
    --manifest $Manifest `
    --episodes 3000 `
    --tokenizer $Tokenizer
if ($LASTEXITCODE -ne 0) {
    throw "Stack v1 dataset audit failed; refusing to train."
}

& (Join-Path $PSScriptRoot "train_stack_v1.ps1") `
    -Python $Python `
    -Dataset $Dataset `
    -OutputRoot $OutputRoot `
    -Seed $Seed
exit $LASTEXITCODE
