param(
    [string]$Python = "python",
    [string]$BaseDataset = "data\lerobot\pick_place_v2_native_bin_1000",
    [string]$TeacherDataset = "data\lerobot\pick_place_calibrated_teacher_v4_400",
    [string]$Model = "outputs\pick_place_v2_native_bin\vla_only_global_20k\seed1000\checkpoints\020000\pretrained_model",
    [string]$Output = "outputs\pick_place_v2_native_bin\visual_lora_transport_v4_2k\seed1000",
    [int]$Steps = 2000,
    [int]$BatchSize = 8,
    [int]$Rank = 16,
    [int]$Seed = 1000
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path (Join-Path $TeacherDataset "collection.complete"))) {
    throw "Teacher dataset is not finalized: $TeacherDataset"
}

# LoRA exposes visual/cross-attention adaptation that action-expert-only runs
# cannot learn, while keeping the 500M SmolVLA deployment model unchanged.
& (Join-Path $PSScriptRoot "train_smolvla.ps1") `
    -Python $Python -Model $Model -Dataset $BaseDataset `
    -RepoId "local/ur5e_pick_place_v2_native_bin" `
    -AuxiliaryDataset $TeacherDataset `
    -AuxiliaryRepoId "local/ur5e_pick_place_calibrated_teacher_v4_400" `
    -AuxiliarySampleWeight "1.0" -AuxiliaryPhaseGroups "transport" `
    -Output $Output -Steps $Steps -BatchSize $BatchSize -NumWorkers 0 -Rank $Rank `
    -LearningRate 0.000002 -WarmupSteps 150 -DecaySteps $Steps -DecayLR 0.0000004 `
    -ChunkSize 16 -ActionSteps 8 -SaveFreq 500 -Seed $Seed `
    -XYZLossWeight 2.0 -RotationLossWeight 0 -GripperLossWeight 2.5 `
    -ApproachWeight 0.18 -GraspWeight 0.20 -LiftWeight 0.20 -TransportWeight 0.27 -PlaceReleaseWeight 0.15 `
    -GlobalTaskPrompt "place the red cube in the blue storage bin" `
    -PhaseBalanced -WandbMode disabled
exit $LASTEXITCODE
