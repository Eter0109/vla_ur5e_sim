param(
    [string]$Python = "C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe",
    [string]$BaseDataset = "data\lerobot\multitask_robust_3000",
    [string]$BaseRepoId = "local/multitask_robust_3000",
    [string]$RecoveryDataset = "data\lerobot\multitask_robust_push_recovery_300",
    [string]$RecoveryRepoId = "local/multitask_robust_push_recovery_300",
    [string]$Output = "outputs\multitask_robust\smolvla_recovery_12k_aux4\seed1000",
    [int]$Steps = 12000,
    [int]$Seed = 1000,
    [int]$BatchSize = 2
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$WarmStart = Join-Path $Root "outputs\multitask_robust\smolvla_30k_lazy_bs2_final\seed1000\checkpoints\030000\pretrained_model"

# Keep embedded-image parquet files memory-mapped and add the corrected Push
# trajectories at four times their natural replay frequency. The base dataset
# retains both original tasks, preventing correction-only fine-tuning from
# forgetting PickPlace.
$env:VLA_FAST_PARQUET_LOADER = "0"
$env:VLA_LAZY_PARQUET_LOADER = "1"
$env:VLA_LAZY_PARQUET_CACHE_FILES = "2"

& (Join-Path $PSScriptRoot "train_smolvla.ps1") `
    -Python $Python -Model $WarmStart `
    -Dataset $BaseDataset -RepoId $BaseRepoId `
    -AuxiliaryDataset $RecoveryDataset -AuxiliaryRepoId $RecoveryRepoId `
    -AuxiliarySampleWeight 4.0 `
    -Output $Output -Steps $Steps -BatchSize $BatchSize -NumWorkers 0 `
    -LearningRate 0.000006 -WarmupSteps 500 -DecaySteps $Steps -DecayLR 0.0000015 `
    -ChunkSize 16 -ActionSteps 8 -SaveFreq $Steps -Seed $Seed `
    -XYZLossWeight 2.0 -RotationLossWeight 0 -GripperLossWeight 2.5 `
    -WandbMode disabled -FullExpert
exit $LASTEXITCODE
