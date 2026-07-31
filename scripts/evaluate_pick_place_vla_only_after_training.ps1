param(
    [string]$Python = "C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe",
    [string]$OutputRoot = "outputs\pick_place_v2_native_bin\vla_only_global_20k",
    [string]$Dataset = "data\lerobot\pick_place_v2_native_bin_1000",
    [string]$Manifest = "configs\benchmarks\pick_place_test_v2_50.json",
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$RunRoot = Join-Path $Root $OutputRoot
$TrainingRoot = Join-Path $RunRoot "seed1000"
$TrainingLog = Join-Path $TrainingRoot "train.log"
$Checkpoint = Join-Path $TrainingRoot "checkpoints\020000\pretrained_model"
$ScreenOutput = Join-Path $RunRoot "vla_only_screen_5.json"
$TestOutput = Join-Path $RunRoot "vla_only_test_50.json"

while (-not (Test-Path -LiteralPath $TrainingLog)) {
    Start-Sleep -Seconds $PollSeconds
}
if (-not (Select-String -LiteralPath $TrainingLog -Pattern "End of training" -Quiet)) {
    throw "Training log exists without a successful completion marker: $TrainingLog"
}
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Training completed without the expected checkpoint: $Checkpoint"
}
if (Test-Path -LiteralPath $ScreenOutput) {
    throw "Refusing to overwrite existing VLA-only screen result: $ScreenOutput"
}
if (Test-Path -LiteralPath $TestOutput) {
    throw "Refusing to overwrite existing VLA-only test result: $TestOutput"
}

& $Python -B (Join-Path $PSScriptRoot "run_pick_place_vla_only.py") `
    --checkpoint $Checkpoint `
    --dataset-root (Join-Path $Root $Dataset) `
    --repo-id local/ur5e_pick_place_v2_native_bin `
    --manifest (Join-Path $Root $Manifest) `
    --episodes 5 `
    --output $ScreenOutput
if ($LASTEXITCODE -ne 0) {
    throw "Five-scene VLA-only screen failed with exit code $LASTEXITCODE"
}

& $Python -B (Join-Path $PSScriptRoot "run_pick_place_vla_only.py") `
    --checkpoint $Checkpoint `
    --dataset-root (Join-Path $Root $Dataset) `
    --repo-id local/ur5e_pick_place_v2_native_bin `
    --manifest (Join-Path $Root $Manifest) `
    --episodes 50 `
    --output $TestOutput
if ($LASTEXITCODE -ne 0) {
    throw "Fifty-scene VLA-only evaluation failed with exit code $LASTEXITCODE"
}
