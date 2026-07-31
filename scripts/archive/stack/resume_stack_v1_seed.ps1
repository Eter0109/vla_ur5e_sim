# Archived Stack v1 launcher; paths reflect the historical repository layout.
param(
    [string]$Python = "python",
    [string]$Dataset = "data\lerobot\stack_v1_3000",
    [string]$OutputRoot = "outputs\stack_v1_final_workers0",
    [ValidateSet(1000, 1001, 1002)]
    [int]$Seed = 1000,
    [string]$CheckpointPath = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Output = Join-Path $OutputRoot "seed$Seed"
$ManifestPath = Join-Path $Output "run_manifest.json"
$PendingManifestPath = "$Output.run_manifest.json"

if (-not (Test-Path $ManifestPath)) {
    if (-not (Test-Path $PendingManifestPath)) {
        throw "Cannot resume without run provenance: $ManifestPath"
    }
    # Fresh runs keep this temporary manifest beside the output directory until
    # LeRobot exits. Promote it only when recovery needs a provenance record.
    Copy-Item -Force $PendingManifestPath $ManifestPath
}

if (-not $CheckpointPath) {
    $CheckpointRoot = Join-Path $Output "checkpoints"
    $CheckpointPath = Get-ChildItem -LiteralPath $CheckpointRoot -Directory |
        Sort-Object Name -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $CheckpointPath -or -not (Test-Path (Join-Path $CheckpointPath "pretrained_model\train_config.json"))) {
    throw "Resume checkpoint is missing train_config.json: $CheckpointPath"
}

$Manifest = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
if ([int]$Manifest.seed -ne $Seed -or [string]$Manifest.dataset -ne $Dataset) {
    throw "Resume request does not match the recorded seed or dataset provenance."
}

& (Join-Path $PSScriptRoot "train_smolvla.ps1") `
    -Python $Python `
    -Model $Manifest.model `
    -Dataset $Dataset `
    -RepoId $Manifest.repo_id `
    -Output $Output `
    -Steps ([int]$Manifest.steps) `
    -BatchSize ([int]$Manifest.batch_size) `
    -NumWorkers ([int]$Manifest.num_workers) `
    -LearningRate ([double]$Manifest.learning_rate) `
    -WarmupSteps ([int]$Manifest.warmup_steps) `
    -DecaySteps ([int]$Manifest.decay_steps) `
    -DecayLR ([double]$Manifest.decay_lr) `
    -ChunkSize ([int]$Manifest.chunk_size) `
    -ActionSteps ([int]$Manifest.action_steps) `
    -SaveFreq 2000 `
    -Seed $Seed `
    -RotationLossWeight ([double]$Manifest.rotation_loss_weight) `
    -GripperLossWeight ([double]$Manifest.gripper_loss_weight) `
    -PhaseBalanced:([bool]$Manifest.phase_balanced) `
    -WandbMode disabled `
    -Resume `
    -CheckpointPath $CheckpointPath `
    -FullExpert:([bool]$Manifest.full_expert)

exit $LASTEXITCODE
