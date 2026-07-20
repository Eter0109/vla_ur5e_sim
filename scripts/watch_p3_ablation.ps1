<# Wait for the externally started A run, then continue the P3 matrix safely. #>
param(
    [int]$PollSeconds = 300
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$AOutput = Join-Path $Root "outputs\smolvla_ablation_a_15k_seed1000"
$Required = @(
    (Join-Path $AOutput "checkpoints\015000\pretrained_model"),
    (Join-Path $AOutput "train.log"),
    (Join-Path $AOutput "run_manifest.json"),
    (Join-Path $AOutput "source.patch")
)

while (@($Required | Where-Object { -not (Test-Path $_) }).Count -gt 0) {
    Write-Output "$(Get-Date -Format o) waiting for completed A provenance bundle"
    Start-Sleep -Seconds $PollSeconds
}

Write-Output "$(Get-Date -Format o) A is complete; continuing P3 matrix"
& (Join-Path $PSScriptRoot "run_p3_ablation.ps1") -RunTraining -ResumeEvaluation
exit $LASTEXITCODE
