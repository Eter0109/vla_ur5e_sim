# Archived Stack v1 launcher; paths reflect the historical repository layout.
param(
    [string]$Python = "python",
    [string]$OutputRoot = "outputs\stack_v1",
    [int]$WaitForPipelinePid = 0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ($WaitForPipelinePid -gt 0) {
    if (Get-Process -Id $WaitForPipelinePid -ErrorAction SilentlyContinue) {
        Write-Host "Waiting for seed 1000 pipeline process $WaitForPipelinePid..."
        Wait-Process -Id $WaitForPipelinePid -ErrorAction Stop
    }
    else {
        Write-Host "Seed 1000 pipeline $WaitForPipelinePid already exited; checking checkpoint."
    }
}

$Seed1000Final = Join-Path $OutputRoot "seed1000\checkpoints\040000\pretrained_model"
if (-not (Test-Path (Join-Path $Seed1000Final "config.json"))) {
    throw "Seed 1000 did not produce its final checkpoint; refusing to start later seeds."
}

foreach ($Seed in 1001, 1002) {
    & (Join-Path $PSScriptRoot "train_stack_v1.ps1") `
        -Python $Python `
        -OutputRoot $OutputRoot `
        -Seed $Seed
    if ($LASTEXITCODE -ne 0) {
        throw "Seed $Seed training failed."
    }
    $FinalCheckpoint = Join-Path $OutputRoot "seed$Seed\checkpoints\040000\pretrained_model\config.json"
    if (-not (Test-Path $FinalCheckpoint)) {
        throw "Seed $Seed exited without its final checkpoint."
    }
}

Write-Host "All three Stack v1 training seeds completed."
