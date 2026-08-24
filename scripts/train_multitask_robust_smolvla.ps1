param(
    [string]$Python = "python",
    [string]$Dataset = "data\lerobot\multitask_robust_3000",
    [string]$RepoId = "local/multitask_robust_3000",
    [string]$Output = "outputs\multitask_robust\smolvla_30k\seed1000",
    [int]$Steps = 30000,
    [int]$Seed = 1000,
    [int]$BatchSize = 2
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$WarmStart = Join-Path $Root "outputs\smolvla_ablation_c_15k_seed1000\checkpoints\015000\pretrained_model"
$env:VLA_FAST_PARQUET_LOADER = "0"
$env:VLA_LAZY_PARQUET_LOADER = "1"
$env:VLA_LAZY_PARQUET_CACHE_FILES = "2"

& (Join-Path $PSScriptRoot "train_smolvla.ps1") `
    -Python $Python -Model $WarmStart `
    -Dataset $Dataset -RepoId $RepoId `
    -Output $Output -Steps $Steps -BatchSize $BatchSize -NumWorkers 0 `
    -LearningRate 0.000012 -WarmupSteps 1000 -DecaySteps $Steps -DecayLR 0.0000015 `
    -ChunkSize 16 -ActionSteps 8 -SaveFreq 2000 -Seed $Seed `
    -XYZLossWeight 2.0 -RotationLossWeight 0 -GripperLossWeight 2.5 `
    -WandbMode disabled -FullExpert
exit $LASTEXITCODE
