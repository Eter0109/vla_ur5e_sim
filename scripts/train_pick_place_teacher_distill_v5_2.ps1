param(
    [string]$Python = "C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe",
    [string]$BaseDataset = "data\lerobot\pick_place_v2_native_bin_1000",
    [string]$TeacherDataset = "data\lerobot\pick_place_calibrated_teacher_v4_400",
    [string]$Model = "outputs\pick_place_v2_native_bin\vla_only_global_20k\seed1000\checkpoints\020000\pretrained_model",
    [string]$Output = "outputs\pick_place_v2_native_bin\teacher_distill_transport_v5_2_800\seed1000",
    [int]$Steps = 800,
    [int]$BatchSize = 8,
    [int]$Seed = 1000
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path (Join-Path $TeacherDataset "collection.complete"))) {
    throw "Teacher dataset is not finalized: $TeacherDataset"
}

# v5.2: AuxiliarySampleWeight = 0.50, LR = 6e-7, Steps = 800
& (Join-Path $PSScriptRoot "train_smolvla.ps1") `
    -Python $Python -Model $Model -Dataset $BaseDataset `
    -RepoId "local/ur5e_pick_place_v2_native_bin" `
    -AuxiliaryDataset $TeacherDataset `
    -AuxiliaryRepoId "local/ur5e_pick_place_calibrated_teacher_v4_400" `
    -AuxiliarySampleWeight "0.50" -AuxiliaryPhaseGroups "transport" `
    -Output $Output -Steps $Steps -BatchSize $BatchSize -NumWorkers 0 `
    -LearningRate 0.0000006 -WarmupSteps 100 -DecaySteps $Steps -DecayLR 0.0000002 `
    -ChunkSize 16 -ActionSteps 8 -SaveFreq 200 -Seed $Seed `
    -XYZLossWeight 2.0 -RotationLossWeight 0 -GripperLossWeight 2.5 `
    -ApproachWeight 0.18 -GraspWeight 0.20 -LiftWeight 0.20 -TransportWeight 0.27 -PlaceReleaseWeight 0.15 `
    -GlobalTaskPrompt "place the red cube in the blue storage bin" `
    -PhaseBalanced -WandbMode disabled -FullExpert
exit $LASTEXITCODE
