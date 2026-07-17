param(
    [string]$Python = "python",
    [string]$Model = ".runtime\models\smolvla_base",
    [string]$Dataset = "data\lerobot\expert_gate10",
    [string]$RepoId = "local/ur5e_custom_lift",
    [string]$Output = "outputs\smolvla_lora_smoke",
    [string]$CheckpointPath = "",
    [int]$Steps = 20,
    [int]$Rank = 4,
    [int]$ImageSize = 256,
    [int]$ChunkSize = 16,
    [int]$ActionSteps = 8,
    [int]$BatchSize = 1,
    [double]$LearningRate = 0.0001,
    [int]$WarmupSteps = 1000,
    [int]$DecaySteps = 0,
    [int]$LogFreq = 20,
    [int]$SaveFreq = 0,
    [int]$Seed = 1000,
    [double]$RotationLossWeight = 1.0,
    [double]$GripperLossWeight = 1.0,
    [double]$TransitionOversampleFactor = 1.0,
    [int]$TransitionOversampleWindow = 0,
    [ValidateSet("disabled", "offline", "online")]
    [string]$WandbMode = "offline",
    [switch]$Resume,
    [switch]$FullExpert
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = (Get-Command $Python -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$env:HF_HOME = Join-Path $Root ".runtime\hf"
$env:HF_DATASETS_CACHE = Join-Path $Root ".runtime\hf_datasets"
$SystemTemp = [System.IO.Path]::GetTempPath()
$env:NUMBA_CACHE_DIR = Join-Path $SystemTemp "vla_sim_numba"
New-Item -ItemType Directory -Force -Path $env:NUMBA_CACHE_DIR | Out-Null
$env:TOKENIZERS_PARALLELISM = "false"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:USE_TF = "0"
$env:TF_CPP_MIN_LOG_LEVEL = "3"
$env:VLA_ROTATION_LOSS_WEIGHT = [string]$RotationLossWeight
$env:VLA_GRIPPER_LOSS_WEIGHT = [string]$GripperLossWeight
$env:VLA_TRANSITION_OVERSAMPLE_FACTOR = [string]$TransitionOversampleFactor
$env:VLA_TRANSITION_OVERSAMPLE_WINDOW = [string]$TransitionOversampleWindow
$env:VLA_SAMPLING_SEED = [string]$Seed

& $Python -c "import torch, lerobot, peft; assert torch.cuda.is_available(), 'CUDA is unavailable'"
if ($LASTEXITCODE -ne 0) {
    throw "The active Python environment is not ready. Run: conda activate vla_sim_gpu"
}

if ($SaveFreq -le 0) {
    $SaveFreq = $Steps
}
if ($ActionSteps -lt 1 -or $ActionSteps -gt $ChunkSize) {
    throw "ActionSteps must be in [1, ChunkSize]."
}
if ($BatchSize -lt 1) {
    throw "BatchSize must be positive."
}
if (
    [double]::IsNaN($RotationLossWeight) -or
    [double]::IsInfinity($RotationLossWeight) -or
    [double]::IsNaN($GripperLossWeight) -or
    [double]::IsInfinity($GripperLossWeight) -or
    $RotationLossWeight -lt 0 -or
    $GripperLossWeight -lt 0 -or
    (3 + 3 * $RotationLossWeight + $GripperLossWeight) -le 0
) {
    throw "Loss weights must be finite, non-negative, and not all zero."
}
if (
    [double]::IsNaN($TransitionOversampleFactor) -or
    [double]::IsInfinity($TransitionOversampleFactor) -or
    $TransitionOversampleFactor -lt 1 -or
    $TransitionOversampleWindow -lt 0
) {
    throw "Transition oversampling factor must be finite and >= 1; window must be >= 0."
}

if ($Resume) {
    if (-not $CheckpointPath) {
        $CheckpointPath = Get-ChildItem (Join-Path $Output "checkpoints") -Directory |
            Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
    }
    $PreviousConfigPath = Join-Path $CheckpointPath "pretrained_model\train_config.json"
    if (-not (Test-Path $PreviousConfigPath)) {
        throw "Resume checkpoint is missing train_config.json: $PreviousConfigPath"
    }
    $PreviousConfig = Get-Content -Raw $PreviousConfigPath | ConvertFrom-Json
    $LearningRate = [double]$PreviousConfig.optimizer.lr
    $WarmupSteps = [int]$PreviousConfig.scheduler.num_warmup_steps
    $DecaySteps = [int]$PreviousConfig.scheduler.num_decay_steps
}
elseif ($DecaySteps -le 0) {
    $DecaySteps = $Steps
}

$TrainArgs = @(
    (Join-Path $PSScriptRoot "train_entrypoint.py")
    "--policy.path=$Model"
    "--policy.input_features=null"
    "--policy.device=cuda"
    "--policy.use_amp=true"
    "--policy.push_to_hub=false"
    # The local SmolVLA checkpoint below supplies all policy weights. Avoid a
    # second VLM initialization before LeRobot loads that checkpoint.
    "--policy.load_vlm_weights=false"
    "--policy.resize_imgs_with_padding=[$ImageSize,$ImageSize]"
    "--policy.chunk_size=$ChunkSize"
    "--policy.n_action_steps=$ActionSteps"
    "--policy.tokenizer_max_length=16"
    "--policy.num_steps=10"
    "--policy.optimizer_lr=$LearningRate"
    "--policy.scheduler_warmup_steps=$WarmupSteps"
    "--policy.scheduler_decay_steps=$DecaySteps"
    "--dataset.repo_id=$RepoId"
    "--dataset.root=$Dataset"
    "--dataset.use_imagenet_stats=false"
    "--batch_size=$BatchSize"
    "--num_workers=0"
    "--steps=$Steps"
    "--eval_freq=0"
    "--log_freq=$LogFreq"
    "--save_freq=$SaveFreq"
    "--output_dir=$Output"
    "--seed=$Seed"
    "--job_name=$([System.IO.Path]::GetFileName($Output))"
)

if ($WandbMode -eq "disabled") {
    $TrainArgs += "--wandb.enable=false"
}
else {
    $TrainArgs += "--wandb.enable=true"
    $TrainArgs += "--wandb.project=vla-ur5e-sim"
    $TrainArgs += "--wandb.mode=$WandbMode"
    $TrainArgs += "--wandb.disable_artifact=true"
}

if (-not $FullExpert) {
    $TrainArgs += "--peft.method_type=LORA"
    $TrainArgs += "--peft.r=$Rank"
}
if ($Resume) {
    $env:VLA_RESUME_CHECKPOINT = $CheckpointPath
    $TrainArgs += "--resume=true"
    $TrainArgs += "--optimizer.type=adamw"
    $TrainArgs += "--optimizer.lr=$LearningRate"
    $TrainArgs += "--optimizer.betas=[0.9,0.95]"
    $TrainArgs += "--optimizer.eps=1e-8"
    $TrainArgs += "--optimizer.weight_decay=1e-10"
    $TrainArgs += "--optimizer.grad_clip_norm=10"
    $TrainArgs += "--scheduler.type=cosine_decay_with_warmup"
    $TrainArgs += "--scheduler.num_warmup_steps=$WarmupSteps"
    $TrainArgs += "--scheduler.num_decay_steps=$DecaySteps"
    $TrainArgs += "--scheduler.peak_lr=$LearningRate"
    $TrainArgs += "--scheduler.decay_lr=0.0000025"
}

$OutputFull = if ([System.IO.Path]::IsPathRooted($Output)) {
    [System.IO.Path]::GetFullPath($Output)
}
else {
    [System.IO.Path]::GetFullPath((Join-Path $Root $Output))
}
$OutputParent = Split-Path -Parent $OutputFull
New-Item -ItemType Directory -Force -Path $OutputParent | Out-Null
$TemporaryLog = "$OutputFull.train.log"
$TemporaryManifest = "$OutputFull.run_manifest.json"
$TemporaryPatch = "$OutputFull.source.patch"

$GitCommit = $null
$GitBranch = $null
$GitDirty = $null
try {
    $GitCommit = (& git -c "safe.directory=$($Root.Replace('\', '/'))" rev-parse HEAD).Trim()
    $GitBranch = (& git -c "safe.directory=$($Root.Replace('\', '/'))" branch --show-current).Trim()
    $GitStatus = (& git -c "safe.directory=$($Root.Replace('\', '/'))" status --porcelain) -join "`n"
    $GitDirty = [bool]$GitStatus
    if ($GitDirty) {
        & git -c "safe.directory=$($Root.Replace('\', '/'))" diff --binary |
            Set-Content -Encoding utf8 $TemporaryPatch
    }
}
catch {
    Write-Warning "Could not capture Git provenance: $_"
}

$Manifest = [ordered]@{
    schema_version = 1
    created_at = [DateTimeOffset]::Now.ToString("o")
    python = $Python
    model = $Model
    dataset = $Dataset
    repo_id = $RepoId
    output = $OutputFull
    steps = $Steps
    seed = $Seed
    full_expert = [bool]$FullExpert
    rank = if ($FullExpert) { $null } else { $Rank }
    image_size = $ImageSize
    chunk_size = $ChunkSize
    action_steps = $ActionSteps
    batch_size = $BatchSize
    learning_rate = $LearningRate
    warmup_steps = $WarmupSteps
    decay_steps = $DecaySteps
    rotation_loss_weight = $RotationLossWeight
    gripper_loss_weight = $GripperLossWeight
    transition_oversample_factor = $TransitionOversampleFactor
    transition_oversample_window = $TransitionOversampleWindow
    resume = [bool]$Resume
    checkpoint_path = $CheckpointPath
    wandb_mode = $WandbMode
    git = [ordered]@{ commit = $GitCommit; branch = $GitBranch; dirty = $GitDirty }
}
$Manifest | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 $TemporaryManifest

Push-Location $Root
$ExitCode = 1
try {
    # Fresh LeRobot runs require a nonexistent output directory. Resume runs
    # require the existing checkpoint tree, so only precreate in that case.
    if ($Resume) {
        New-Item -ItemType Directory -Force -Path $Output | Out-Null
    }
    if ($Resume -and (Test-Path (Join-Path $OutputFull "train.log"))) {
        Copy-Item -Force (Join-Path $OutputFull "train.log") $TemporaryLog
    }
    $TeeArguments = @{ FilePath = $TemporaryLog }
    if ($Resume) {
        $TeeArguments.Append = $true
    }
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python $TrainArgs 2>&1 | Tee-Object @TeeArguments
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
}
finally {
    Pop-Location
    if (Test-Path $OutputFull) {
        if (Test-Path $TemporaryLog) {
            Move-Item -Force $TemporaryLog (Join-Path $OutputFull "train.log")
        }
        if (Test-Path $TemporaryManifest) {
            Move-Item -Force $TemporaryManifest (Join-Path $OutputFull "run_manifest.json")
        }
        if (Test-Path $TemporaryPatch) {
            Move-Item -Force $TemporaryPatch (Join-Path $OutputFull "source.patch")
        }
    }
}
exit $ExitCode
