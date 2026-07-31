# Archived Stack v1 training launcher; paths reflect the historical repository layout.
param(
    [string]$Python = "python",
    [string]$Dataset = "data\lerobot\stack_v1_3000",
    [string]$OutputRoot = "outputs\stack_v1",
    [ValidateSet(1000, 1001, 1002)]
    [int]$Seed = 1000
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$WarmStart = Join-Path $Root `
    "outputs\smolvla_ablation_c_15k_seed1000\checkpoints\015000\pretrained_model"
$Output = Join-Path $OutputRoot "seed$Seed"

# PyAV decoding with Windows worker processes aborts in the native runtime.
# Keep the stable setting explicit in each run manifest.
& (Join-Path $PSScriptRoot "train_smolvla.ps1") `
    -Python $Python `
    -Model $WarmStart `
    -Dataset $Dataset `
    -RepoId "local/ur5e_stack_v1" `
    -Output $Output `
    -Steps 40000 `
    -BatchSize 8 `
    -NumWorkers 0 `
    -LearningRate 0.00002 `
    -WarmupSteps 500 `
    -DecaySteps 40000 `
    -DecayLR 0.0000025 `
    -ChunkSize 16 `
    -ActionSteps 8 `
    -SaveFreq 2000 `
    -Seed $Seed `
    -RotationLossWeight 0 `
    -GripperLossWeight 0 `
    -PhaseBalanced `
    -FullExpert

exit $LASTEXITCODE
