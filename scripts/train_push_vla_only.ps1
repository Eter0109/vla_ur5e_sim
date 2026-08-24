param(
    [string]$Python = "python",
    [string]$Dataset = "data\lerobot\ur5e_push_1000_v6",
    [string]$OutputRoot = "outputs\push\vla_only_forward_v6",
    [string]$RepoId = "local/ur5e_push_1000_v6",
    [ValidateSet(1000, 1001, 1002)] [int]$Seed = 1000,
    [int]$Steps = 20000,
    [int]$BatchSize = 8
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$WarmStart = Join-Path $Root "outputs\smolvla_ablation_c_15k_seed1000\checkpoints\015000\pretrained_model"
& (Join-Path $PSScriptRoot "train_smolvla.ps1") `
    -Python $Python -Model $WarmStart -Dataset $Dataset `
    -RepoId $RepoId `
    -Output (Join-Path $OutputRoot "seed$Seed") -Steps $Steps `
    -BatchSize $BatchSize -NumWorkers 0 `
    -LearningRate 0.000012 -WarmupSteps 1000 -DecaySteps $Steps -DecayLR 0.0000015 `
    -ChunkSize 16 -ActionSteps 8 -SaveFreq 2000 -Seed $Seed `
    -XYZLossWeight 2.0 -RotationLossWeight 0 -GripperLossWeight 2.5 `
    -GlobalTaskPrompt "push the block into the red target circle" `
    -WandbMode disabled -FullExpert
exit $LASTEXITCODE
