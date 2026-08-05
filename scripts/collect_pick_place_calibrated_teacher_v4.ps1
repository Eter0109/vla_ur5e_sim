param(
    [string]$Python = "C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe",
    [int]$Episodes = 400
)

$Root = Split-Path -Parent $PSScriptRoot
$Dataset = Join-Path $Root "data\lerobot\pick_place_calibrated_teacher_v4_400"
$LogDirectory = Join-Path $Root "outputs\pick_place_v2_native_bin\teacher_distill_transport_v4_logs\seed1000"
$StandardOutput = Join-Path $LogDirectory "collect_calibrated_teacher_v4_400.log"
$StandardError = Join-Path $LogDirectory "collect_calibrated_teacher_v4_400.err.log"
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
$LaunchError = Join-Path $LogDirectory "collect_calibrated_teacher_v4_400.launch.err.log"
$ErrorActionPreference = "Stop"

try {
    if (Test-Path $Dataset) {
        throw "Refusing to overwrite teacher dataset: $Dataset"
    }
    Push-Location $Root
    try {
        $PreviousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $Python scripts\collect_pick_place_calibrated_rollouts.py `
                --checkpoint outputs\pick_place_v2_native_bin\vla_only_global_20k\seed1000\checkpoints\020000\pretrained_model `
                --dataset-root data\lerobot\pick_place_v2_native_bin_1000 `
                --repo-id local/ur5e_pick_place_v2_native_bin `
                --manifest configs\benchmarks\pick_place_collect_v1.json `
                --root data\lerobot\pick_place_calibrated_teacher_v4_400 `
                --output-repo-id local/ur5e_pick_place_calibrated_teacher_v4_400 `
                --episodes $Episodes --replan-steps 4 --samples-per-plan 2 `
                --policy-seed 1000 --negative-y-gain 1.3 --positive-x-gain 0.95 `
                1> $StandardOutput 2> $StandardError
            $Code = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $PreviousErrorActionPreference
        }
    }
    finally {
        Pop-Location
    }
    exit $Code
}
catch {
    $_ | Out-File -FilePath $LaunchError -Encoding utf8
    exit 1
}
