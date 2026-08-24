param(
    [string]$Python = "python",
    [string]$Model = ".runtime\models\smolvla_base",
    [string]$Dataset = "data\lerobot\expert_gate10",
    [string]$RepoId = "local/ur5e_custom_lift",
    [string]$AuxiliaryDataset = "",
    [string]$AuxiliaryRepoId = "",
    [string]$AuxiliarySampleWeight = "1.0",
    [string]$AuxiliaryPhaseGroups = "",
    [string]$BaseTaskPrompt = "",
    [string]$AuxiliaryTaskPrompts = "",
    [string]$Output = "outputs\smolvla_lora_smoke",
    [string]$CheckpointPath = "",
    [int]$Steps = 20,
    [int]$Rank = 4,
    [int]$ImageSize = 256,
    [int]$ChunkSize = 16,
    [int]$ActionSteps = 8,
    [int]$BatchSize = 1,
    [int]$NumWorkers = 0,
    [double]$LearningRate = 0.0001,
    [int]$WarmupSteps = 1000,
    [int]$DecaySteps = 0,
    [double]$DecayLR = 0,
    [int]$LogFreq = 20,
    [int]$SaveFreq = 0,
    [int]$Seed = 1000,
    [double]$XYZLossWeight = 1.0,
    [double]$RotationLossWeight = 1.0,
    [double]$GripperLossWeight = 1.0,
    [double]$ApproachWeight = 0.25,
    [double]$GraspWeight = 0.20,
    [double]$LiftWeight = 0.25,
    [double]$TransportWeight = 0.20,
    [double]$PlaceReleaseWeight = 0.10,
    [double]$TransitionOversampleFactor = 1.0,
    [int]$TransitionOversampleWindow = 0,
    [string]$GlobalTaskPrompt = "",
    [switch]$PhaseBalanced,
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
$env:VLA_XYZ_LOSS_WEIGHT = [string]$XYZLossWeight
$env:VLA_ROTATION_LOSS_WEIGHT = [string]$RotationLossWeight
$env:VLA_GRIPPER_LOSS_WEIGHT = [string]$GripperLossWeight
$env:VLA_APPROACH_WEIGHT = [string]$ApproachWeight
$env:VLA_GRASP_WEIGHT = [string]$GraspWeight
$env:VLA_LIFT_WEIGHT = [string]$LiftWeight
$env:VLA_TRANSPORT_WEIGHT = [string]$TransportWeight
$env:VLA_PLACE_RELEASE_WEIGHT = [string]$PlaceReleaseWeight
$env:VLA_PHASE_CHUNK_SIZE = [string]$ChunkSize
$env:VLA_TRANSITION_OVERSAMPLE_FACTOR = [string]$TransitionOversampleFactor
$env:VLA_TRANSITION_OVERSAMPLE_WINDOW = [string]$TransitionOversampleWindow
$env:VLA_GLOBAL_TASK_PROMPT = $GlobalTaskPrompt
$env:VLA_SAMPLING_SEED = [string]$Seed
$env:VLA_PHASE_BALANCED = if ($PhaseBalanced) { "1" } else { "0" }
$env:VLA_AUXILIARY_DATASET = $AuxiliaryDataset
$env:VLA_AUXILIARY_REPO_ID = $AuxiliaryRepoId
$env:VLA_AUXILIARY_SAMPLE_WEIGHT = [string]$AuxiliarySampleWeight
$env:VLA_AUXILIARY_PHASE_GROUPS = $AuxiliaryPhaseGroups
$env:VLA_BASE_TASK_PROMPT = $BaseTaskPrompt
$env:VLA_AUXILIARY_TASK_PROMPTS = $AuxiliaryTaskPrompts

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
if ($NumWorkers -lt 0) {
    throw "NumWorkers must be non-negative."
}
if ($AuxiliaryDataset) {
    if (-not $AuxiliaryRepoId) {
        throw "AuxiliaryRepoId is required when AuxiliaryDataset is set."
    }
    $AuxiliaryRoots = @($AuxiliaryDataset -split ';' | Where-Object { $_ })
    $AuxiliaryRepoIds = @($AuxiliaryRepoId -split ';' | Where-Object { $_ })
    $AuxiliaryWeights = @($AuxiliarySampleWeight -split ';' | ForEach-Object { [double]$_ })
    if (
        $AuxiliaryRoots.Count -ne $AuxiliaryRepoIds.Count -or
        $AuxiliaryRoots.Count -ne $AuxiliaryWeights.Count -or
        @($AuxiliaryWeights | Where-Object {
            [double]::IsNaN($_) -or [double]::IsInfinity($_) -or $_ -le 0
        }).Count -gt 0
    ) {
        throw "AuxiliarySampleWeight must be finite and positive."
    }
}
if (
    [double]::IsNaN($XYZLossWeight) -or
    [double]::IsInfinity($XYZLossWeight) -or
    [double]::IsNaN($RotationLossWeight) -or
    [double]::IsInfinity($RotationLossWeight) -or
    [double]::IsNaN($GripperLossWeight) -or
    [double]::IsInfinity($GripperLossWeight) -or
    $XYZLossWeight -lt 0 -or
    $RotationLossWeight -lt 0 -or
    $GripperLossWeight -lt 0 -or
    (3 * $XYZLossWeight + 3 * $RotationLossWeight + $GripperLossWeight) -le 0
) {
    throw "Loss weights must be finite, non-negative, and not all zero."
}
if (
    [double]::IsNaN($ApproachWeight) -or
    [double]::IsInfinity($ApproachWeight) -or
    [double]::IsNaN($GraspWeight) -or
    [double]::IsInfinity($GraspWeight) -or
    [double]::IsNaN($LiftWeight) -or
    [double]::IsInfinity($LiftWeight) -or
    [double]::IsNaN($TransportWeight) -or
    [double]::IsInfinity($TransportWeight) -or
    [double]::IsNaN($PlaceReleaseWeight) -or
    [double]::IsInfinity($PlaceReleaseWeight) -or
    $ApproachWeight -le 0 -or $ApproachWeight -ge 1 -or
    $GraspWeight -le 0 -or $GraspWeight -ge 1 -or
    $LiftWeight -le 0 -or $LiftWeight -ge 1 -or
    $TransportWeight -le 0 -or $TransportWeight -ge 1 -or
    $PlaceReleaseWeight -le 0 -or $PlaceReleaseWeight -ge 1 -or
    [math]::Abs(
        $ApproachWeight + $GraspWeight + $LiftWeight +
        $TransportWeight + $PlaceReleaseWeight - 1.0
    ) -gt 1e-9
) {
    throw "Phase weights must be finite, in (0, 1), and sum to 1."
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
    $PreviousManifestPath = Join-Path $Output "run_manifest.json"
    if (-not (Test-Path $PreviousManifestPath)) {
        throw "Resume requires the original run_manifest.json: $PreviousManifestPath"
    }
    $PreviousManifest = Get-Content -Raw $PreviousManifestPath | ConvertFrom-Json
    $Locked = [ordered]@{
        model = $Model
        dataset = $Dataset
        repo_id = $RepoId
        seed = $Seed
        chunk_size = $ChunkSize
        action_steps = $ActionSteps
        batch_size = $BatchSize
        num_workers = $NumWorkers
        xyz_loss_weight = $XYZLossWeight
        rotation_loss_weight = $RotationLossWeight
        gripper_loss_weight = $GripperLossWeight
        approach_weight = $ApproachWeight
        grasp_weight = $GraspWeight
        lift_weight = $LiftWeight
        transport_weight = $TransportWeight
        place_release_weight = $PlaceReleaseWeight
        global_task_prompt = $GlobalTaskPrompt
        base_task_prompt = $BaseTaskPrompt
        auxiliary_task_prompts = $AuxiliaryTaskPrompts
        phase_balanced = [bool]$PhaseBalanced
        full_expert = [bool]$FullExpert
    }
    if ($AuxiliaryDataset) {
        $Locked.auxiliary_dataset = $AuxiliaryDataset
        $Locked.auxiliary_repo_id = $AuxiliaryRepoId
        $Locked.auxiliary_sample_weight = $AuxiliarySampleWeight
        $Locked.auxiliary_phase_groups = $AuxiliaryPhaseGroups
    }
    foreach ($Name in $Locked.Keys) {
        if ([string]$PreviousManifest.$Name -ne [string]$Locked[$Name]) {
            throw "Resume provenance mismatch for $Name"
        }
    }
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
    "--num_workers=$NumWorkers"
    "--steps=$Steps"
    "--eval_freq=0"
    "--log_freq=$LogFreq"
    "--save_freq=$SaveFreq"
    "--output_dir=$Output"
    "--seed=$Seed"
    "--job_name=$([System.IO.Path]::GetFileName($Output))"
)

if ($DecayLR -gt 0) {
    $TrainArgs += "--policy.scheduler_decay_lr=$DecayLR"
}

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
    $TrainArgs += "--peft.target_modules=`"all-linear`""
}
else {
    # "FullExpert" means full action-expert tuning, not full-model tuning.
    $TrainArgs += "--policy.freeze_vision_encoder=true"
    $TrainArgs += "--policy.train_expert_only=true"
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
    $ActualDecayLR = if ($DecayLR -gt 0) { $DecayLR } else { 0.0000025 }
    $TrainArgs += "--scheduler.decay_lr=$ActualDecayLR"
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
    auxiliary_dataset = $AuxiliaryDataset
    auxiliary_repo_id = $AuxiliaryRepoId
    auxiliary_sample_weight = $AuxiliarySampleWeight
        auxiliary_phase_groups = $AuxiliaryPhaseGroups
        base_task_prompt = $BaseTaskPrompt
        auxiliary_task_prompts = $AuxiliaryTaskPrompts
    output = $OutputFull
    steps = $Steps
    seed = $Seed
    full_expert = [bool]$FullExpert
    rank = if ($FullExpert) { $null } else { $Rank }
    image_size = $ImageSize
    chunk_size = $ChunkSize
    action_steps = $ActionSteps
    batch_size = $BatchSize
    num_workers = $NumWorkers
    learning_rate = $LearningRate
    warmup_steps = $WarmupSteps
    decay_steps = $DecaySteps
    decay_lr = if ($DecayLR -gt 0) { $DecayLR } else { 2.5e-6 }
    xyz_loss_weight = $XYZLossWeight
    rotation_loss_weight = $RotationLossWeight
    gripper_loss_weight = $GripperLossWeight
    approach_weight = $ApproachWeight
    grasp_weight = $GraspWeight
    lift_weight = $LiftWeight
    transport_weight = $TransportWeight
    place_release_weight = $PlaceReleaseWeight
    global_task_prompt = $GlobalTaskPrompt
    transition_oversample_factor = $TransitionOversampleFactor
    transition_oversample_window = $TransitionOversampleWindow
    phase_balanced = [bool]$PhaseBalanced
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
