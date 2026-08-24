param(
    [string]$Python = "C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe",
    [string]$Dataset = "data\lerobot\multitask_robust_3000",
    [string]$RepoId = "local/multitask_robust_3000",
    [string]$CorrectedPushDataset = "data\lerobot\multitask_robust_push_recovery_300",
    [string]$CorrectedPushRepoId = "local/multitask_robust_push_recovery_300",
    [string]$Output = "outputs\multitask_robust\smolvla_finetune_5k\seed1000",
    [int]$Steps = 5000,
    [int]$SaveFreq = 1000,
    [int]$Seed = 1000,
    [int]$BatchSize = 2
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$WarmStart = Join-Path $Root "outputs\multitask_robust\smolvla_full_retrain_30k_corrected_push\seed1000\checkpoints\030000\pretrained_model"

if (-not (Test-Path $WarmStart)) {
    throw "The warm start model is missing: $WarmStart"
}

$env:VLA_FAST_PARQUET_LOADER = "0"
$env:VLA_LAZY_PARQUET_LOADER = "1"
$env:VLA_LAZY_PARQUET_CACHE_FILES = "2"

& (Join-Path $PSScriptRoot "train_smolvla.ps1") `
    -Python $Python -Model $WarmStart `
    -Dataset $Dataset -RepoId $RepoId `
    -AuxiliaryDataset $CorrectedPushDataset -AuxiliaryRepoId $CorrectedPushRepoId `
    -AuxiliarySampleWeight 1.0 `
    -Output $Output -Steps $Steps -BatchSize $BatchSize -NumWorkers 0 `
    -LearningRate 0.000005 -WarmupSteps 500 -DecaySteps $Steps -DecayLR 0.000001 `
    -ChunkSize 16 -ActionSteps 8 -SaveFreq $SaveFreq -Seed $Seed `
    -XYZLossWeight 2.0 -RotationLossWeight 0 -GripperLossWeight 2.5 `
    -WandbMode disabled -FullExpert
exit $LASTEXITCODE
