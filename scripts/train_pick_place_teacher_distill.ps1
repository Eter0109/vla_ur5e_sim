param(
    [string]$Python = "python",
    [string]$BaseDataset = "data\lerobot\pick_place_v2_native_bin_1000",
    [string]$TeacherDataset = "data\lerobot\pick_place_calibrated_teacher_v4_400",
    [string]$Model = "outputs\pick_place_v2_native_bin\vla_only_global_20k\seed1000\checkpoints\020000\pretrained_model",
    [string]$Output = "outputs\pick_place_v2_native_bin\teacher_distill_transport_v4_3k\seed1000",
    [int]$Steps = 3000,
    [int]$BatchSize = 8,
    [int]$Seed = 1000
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path (Join-Path $TeacherDataset "collection.complete"))) {
    throw "Teacher dataset is not finalized: $TeacherDataset"
}

# Replay corrected teacher actions only during transport. Other phases stay on
# the broad base distribution so the proven grasp behaviour is not overwritten.
& (Join-Path $PSScriptRoot "train_smolvla.ps1") `
    -Python $Python -Model $Model -Dataset $BaseDataset `
    -RepoId "local/ur5e_pick_place_v2_native_bin" `
    -AuxiliaryDataset $TeacherDataset `
    -AuxiliaryRepoId "local/ur5e_pick_place_calibrated_teacher_v4_400" `
    -AuxiliarySampleWeight "1.0" -AuxiliaryPhaseGroups "transport" `
    -Output $Output -Steps $Steps -BatchSize $BatchSize -NumWorkers 0 `
    -LearningRate 0.0000015 -WarmupSteps 200 -DecaySteps $Steps -DecayLR 0.0000004 `
    -ChunkSize 16 -ActionSteps 8 -SaveFreq 500 -Seed $Seed `
    -XYZLossWeight 2.0 -RotationLossWeight 0 -GripperLossWeight 2.5 `
    -ApproachWeight 0.18 -GraspWeight 0.20 -LiftWeight 0.20 -TransportWeight 0.27 -PlaceReleaseWeight 0.15 `
    -GlobalTaskPrompt "place the red cube in the blue storage bin" `
    -PhaseBalanced -WandbMode disabled -FullExpert
exit $LASTEXITCODE
