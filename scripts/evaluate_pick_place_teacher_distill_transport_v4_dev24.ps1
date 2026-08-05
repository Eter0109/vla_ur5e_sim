param(
    [string]$Python = "C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$RunRoot = "outputs\pick_place_v2_native_bin\teacher_distill_transport_v4_3k\seed1000"
$Development = Join-Path $RunRoot "development"
$Stem = "dev24_002000_samples2_replan8_seed1000"
$Output = Join-Path $Development "$Stem.json"
if (Test-Path (Join-Path $Root $Output)) {
    Write-Output "skip_existing output=$Output"
    exit 0
}

Push-Location $Root
try {
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python scripts\run_pick_place_vla_only.py `
            --checkpoint (Join-Path $RunRoot "checkpoints\002000\pretrained_model") `
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
