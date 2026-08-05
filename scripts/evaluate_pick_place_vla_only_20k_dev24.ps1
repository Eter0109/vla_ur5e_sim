param(
    [string]$Python = "C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$RunRoot = "outputs\pick_place_v2_native_bin\vla_only_global_20k\seed1000"
$Development = Join-Path $RunRoot "development"
if (-not (Test-Path $Development)) {
    New-Item -ItemType Directory -Path $Development -Force | Out-Null
}
$Stem = "dev24_020000_samples2_replan8_seed1000"
$Output = Join-Path $Development "$Stem.json"

Push-Location $Root
try {
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python scripts\run_pick_place_vla_only.py `
            --checkpoint (Join-Path $RunRoot "checkpoints\020000\pretrained_model") `
            --dataset-root data\lerobot\pick_place_v2_native_bin_1000 `
            --repo-id local/ur5e_pick_place_v2_native_bin `
            --manifest configs\benchmarks\pick_place_dev_v1.json `
            --episodes 24 --replan-steps 8 --samples-per-plan 2 `
            --policy-seed 1000 --control-mode vla_raw_safety `
            --output $Output `
            1> (Join-Path $Development "$Stem.log") `
            2> (Join-Path $Development "$Stem.err.log")
        $Code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    exit $Code
}
finally {
    Pop-Location
}
