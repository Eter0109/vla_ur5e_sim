param(
    [string]$Python = "C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe",
    [string]$BaseDataset = "data\lerobot\multitask_robust_3000",
    [string]$PushDataset = "data\lerobot\push_robust_targeted_recovery_v2_500",
    [string]$PickNegativeDataset = "data\lerobot\pick_place_correction_negative_y_v1_400",
    [string]$PickPositiveDataset = "data\lerobot\pick_place_correction_positive_y_v1_400",
    [string]$PickGraspDataset = "data\lerobot\pick_place_grasp_recovery_positive_y_v1_400",
    [string]$TargetedAudit = "outputs\multitask_robust\audit_targeted_push_recovery_v2_500.json",
    [string]$Output = "outputs\multitask_robust\smolvla_target80_lora_recovery_3k\seed1000",
    [int]$Steps = 3000,
    [int]$SaveFreq = 500,
    [int]$Seed = 1000,
    [int]$BatchSize = 2,
    [int]$Rank = 8,
    [string]$AuxiliaryWeights = "0.75;0.10;0.10;0.15",
    [switch]$PhaseBalanced
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$WarmStart = Join-Path $Root "outputs\multitask_robust\smolvla_30k_lazy_bs2_final\seed1000\checkpoints\030000\pretrained_model"
$ExpectedWarmStartSha256 = "DFFBFCD07911EBCC0658B853ED5031855741DAE26C3241B9AA67AB56C76DD7B7"
$TargetedManifest = Join-Path $Root "configs\benchmarks\push_robust_targeted_recovery_collection_v2.json"

if (-not (Test-Path $WarmStart)) { throw "Warm start is missing: $WarmStart" }
$WarmStartSha256 = (Get-FileHash (Join-Path $WarmStart "model.safetensors") -Algorithm SHA256).Hash
if ($WarmStartSha256 -ne $ExpectedWarmStartSha256) {
    throw "The target-80 warm start hash is wrong: $WarmStartSha256"
}

$BaseInfoPath = Join-Path $BaseDataset "meta\info.json"
$BaseProvenancePath = Join-Path $BaseDataset "meta\build_provenance.json"
if (-not (Test-Path $BaseInfoPath) -or -not (Test-Path $BaseProvenancePath)) {
    throw "The merged base dataset metadata is incomplete: $BaseDataset"
}
$BaseInfo = Get-Content $BaseInfoPath -Raw | ConvertFrom-Json
$BaseProvenance = Get-Content $BaseProvenancePath -Raw | ConvertFrom-Json
if (
    [int]$BaseInfo.total_episodes -ne 3000 -or
    [int]$BaseInfo.total_frames -ne 216932 -or
    [int]$BaseInfo.total_tasks -ne 2 -or
    [string]$BaseProvenance.push_prompt -ne "push the block into the red target circle" -or
    [string]$BaseProvenance.pick_place_prompt -ne "place the red cube in the blue storage bin"
) { throw "The merged base dataset contract is wrong: $BaseDataset" }
foreach ($Source in @($BaseProvenance.push_root, $BaseProvenance.pick_place_root)) {
    $SourceFull = if ([IO.Path]::IsPathRooted($Source)) { $Source } else { Join-Path $Root $Source }
    if (-not (Test-Path (Join-Path $SourceFull "collection.complete"))) {
        throw "The merged base source dataset is not finalized: $SourceFull"
    }
}
foreach ($Dataset in @($PushDataset, $PickNegativeDataset, $PickPositiveDataset, $PickGraspDataset)) {
    if (-not (Test-Path (Join-Path $Dataset "collection.complete"))) {
        throw "Training auxiliary dataset is not finalized: $Dataset"
    }
}

$TargetedAuditFull = if ([IO.Path]::IsPathRooted($TargetedAudit)) {
    [IO.Path]::GetFullPath($TargetedAudit)
} else { [IO.Path]::GetFullPath((Join-Path $Root $TargetedAudit)) }
if (-not (Test-Path $TargetedAuditFull)) { throw "Targeted Push audit is missing" }
$Audit = Get-Content $TargetedAuditFull -Raw | ConvertFrom-Json
$ExpectedManifestSha256 = (Get-FileHash $TargetedManifest -Algorithm SHA256).Hash.ToLowerInvariant()
if (
    $Audit.status -ne "ok" -or
    [int]$Audit.episodes -ne 500 -or
    [string]$Audit.manifest_sha256 -ne $ExpectedManifestSha256
) { throw "Targeted Push audit does not authorize LoRA training" }

$env:VLA_FAST_PARQUET_LOADER = "0"
$env:VLA_LAZY_PARQUET_LOADER = "1"
$env:VLA_LAZY_PARQUET_CACHE_FILES = "2"
$PushPrompt = "push the block into the red target circle"
$PickPrompt = "place the red cube in the blue storage bin"

# LoRA adapts visual/cross-modal linear layers while retaining the base weights.
# A phase-balanced run can protect grasp and release behavior while correcting
# visual Push failures; the historical recovery remains the default settings.
$PhaseBalancedArgs = @()
if ($PhaseBalanced) { $PhaseBalancedArgs = @("-PhaseBalanced") }

& (Join-Path $PSScriptRoot "train_smolvla.ps1") `
    -Python $Python -Model $WarmStart `
    -Dataset $BaseDataset -RepoId "local/multitask_robust_3000" `
    -AuxiliaryDataset "$PushDataset;$PickNegativeDataset;$PickPositiveDataset;$PickGraspDataset" `
    -AuxiliaryRepoId "local/push_robust_targeted_recovery_v2_500;local/ur5e_pick_place_correction_negative_y_v1_400;local/ur5e_pick_place_correction_positive_y_v1_400;local/ur5e_pick_place_grasp_recovery_positive_y_v1_400" `
    -AuxiliarySampleWeight $AuxiliaryWeights `
    -AuxiliaryTaskPrompts "$PushPrompt;$PickPrompt;$PickPrompt;$PickPrompt" `
    -Output $Output -Steps $Steps -BatchSize $BatchSize -NumWorkers 0 -Rank $Rank `
    -LearningRate 0.000002 -WarmupSteps 200 -DecaySteps $Steps -DecayLR 0.0000004 `
    -ChunkSize 16 -ActionSteps 8 -SaveFreq $SaveFreq -Seed $Seed `
    -XYZLossWeight 2.0 -RotationLossWeight 0 -GripperLossWeight 2.5 `
    -WandbMode disabled @PhaseBalancedArgs
exit $LASTEXITCODE
