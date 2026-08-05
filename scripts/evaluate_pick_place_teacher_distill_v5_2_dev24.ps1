param(
    [string]$Python = "C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe",
    [string]$Step = "000400",
    [string]$RunRoot = "outputs\pick_place_v2_native_bin\teacher_distill_transport_v5_2_800\seed1000"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Development = Join-Path $RunRoot "development"
if (-not (Test-Path $Development)) {
    New-Item -ItemType Directory -Path $Development -Force | Out-Null
}
$Stem = "dev24_${Step}_samples2_replan8_seed1000"
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
            --checkpoint (Join-Path $RunRoot "checkpoints\$Step\pretrained_model") `
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
