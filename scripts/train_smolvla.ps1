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
    [double]$LearningRate = 0.0001,
    [int]$WarmupSteps = 1000,
    [int]$DecaySteps = 0,
    [int]$LogFreq = 20,
    [int]$SaveFreq = 0,
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

& $Python -c "import torch, lerobot, peft; assert torch.cuda.is_available(), 'CUDA is unavailable'"
if ($LASTEXITCODE -ne 0) {
    throw "The active Python environment is not ready. Run: conda activate vla_sim_gpu"
}

if ($SaveFreq -le 0) {
    $SaveFreq = $Steps
}
if ($DecaySteps -le 0) {
    $DecaySteps = $Steps
}
if ($ActionSteps -lt 1 -or $ActionSteps -gt $ChunkSize) {
    throw "ActionSteps must be in [1, ChunkSize]."
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
    "--batch_size=1"
    "--num_workers=0"
    "--steps=$Steps"
    "--eval_freq=0"
    "--log_freq=$LogFreq"
    "--save_freq=$SaveFreq"
    "--output_dir=$Output"
    "--wandb.enable=false"
)

if (-not $FullExpert) {
    $TrainArgs += "--peft.method_type=LORA"
    $TrainArgs += "--peft.r=$Rank"
}
if ($Resume) {
    if (-not $CheckpointPath) {
        $CheckpointPath = Get-ChildItem (Join-Path $Output "checkpoints") -Directory |
            Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
    }
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

Push-Location $Root
try {
    # Fresh LeRobot runs require a nonexistent output directory. Resume runs
    # require the existing checkpoint tree, so only precreate in that case.
    if ($Resume) {
        New-Item -ItemType Directory -Force -Path $Output | Out-Null
    }
    & $Python $TrainArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
