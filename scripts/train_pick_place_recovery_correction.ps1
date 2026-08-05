param(
    [string]$Python = "python",
    [string]$BaseDataset = "data\lerobot\pick_place_v2_native_bin_1000",
    [string]$NegativeDataset = "data\lerobot\pick_place_correction_negative_y_v1_400",
    [string]$PositiveDataset = "data\lerobot\pick_place_correction_positive_y_v1_400",
    [string]$GraspRecoveryDataset = "data\lerobot\pick_place_grasp_recovery_positive_y_v1_400",
    [string]$Model = "outputs\pick_place_v2_native_bin\vla_only_global_20k\seed1000\checkpoints\020000\pretrained_model",
    [string]$Output = "outputs\pick_place_v2_native_bin\recovery_correction_3k\seed1000",
    [int]$Steps = 3000,
    [int]$BatchSize = 8,
    [int]$Seed = 1000
)

$ErrorActionPreference = "Stop"
foreach ($dataset in @($NegativeDataset, $PositiveDataset, $GraspRecoveryDataset)) {
    if (-not (Test-Path (Join-Path $dataset "collection.complete"))) {
        throw "Correction dataset is not finalized: $dataset"
    }
}

# The three correction sources occupy about 23% of frame samples: sufficient to
# teach recovery without replacing the original deployment normalizer/domain.
& (Join-Path $PSScriptRoot "train_smolvla.ps1") `
    -Python $Python -Model $Model -Dataset $BaseDataset `
    -RepoId "local/ur5e_pick_place_v2_native_bin" `
    -AuxiliaryDataset "$NegativeDataset;$PositiveDataset;$GraspRecoveryDataset" `
    -AuxiliaryRepoId "local/ur5e_pick_place_correction_negative_y_v1_400;local/ur5e_pick_place_correction_positive_y_v1_400;local/ur5e_pick_place_grasp_recovery_positive_y_v1_400" `
    -AuxiliarySampleWeight "0.20;0.20;0.35" `
    -Output $Output -Steps $Steps -BatchSize $BatchSize -NumWorkers 0 `
    -LearningRate 0.0000015 -WarmupSteps 200 -DecaySteps $Steps -DecayLR 0.0000004 `
    -ChunkSize 16 -ActionSteps 8 -SaveFreq 500 -Seed $Seed `
    -XYZLossWeight 2.0 -RotationLossWeight 0 -GripperLossWeight 2.5 `
    -ApproachWeight 0.18 -GraspWeight 0.28 -LiftWeight 0.20 -TransportWeight 0.20 -PlaceReleaseWeight 0.14 `
    -GlobalTaskPrompt "place the red cube in the blue storage bin" `
    -PhaseBalanced -WandbMode disabled -FullExpert
exit $LASTEXITCODE
