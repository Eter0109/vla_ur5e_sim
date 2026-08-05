param(
    [string]$Python = "C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe",
    [string]$RunRoot = "outputs\pick_place_v2_native_bin\teacher_distill_transport_v5_1_600\seed1000",
    [int]$SamplesPerPlan = 2
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Steps = @("000150", "000300", "000450", "000600")
Push-Location $Root
try {
    foreach ($Step in $Steps) {
        $Checkpoint = Join-Path $RunRoot "checkpoints\$Step\pretrained_model"
        $Development = Join-Path $RunRoot "development"
        $Output = Join-Path $Development "screen6_${Step}_samples${SamplesPerPlan}_replan8_seed1000.json"
        $Log = Join-Path $Development "screen6_${Step}_samples${SamplesPerPlan}_replan8_seed1000.log"
        $ErrorLog = Join-Path $Development "screen6_${Step}_samples${SamplesPerPlan}_replan8_seed1000.err.log"
        if (Test-Path $Output) {
            Write-Output "skip_existing checkpoint=$Step"
            continue
        }
        if (-not (Test-Path $Checkpoint)) {
            Write-Output "Checkpoint $Step not found yet, stopping loop."
            break
        }
        New-Item -ItemType Directory -Force -Path $Development | Out-Null
        $PreviousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $Python scripts\run_pick_place_vla_only.py `
                --checkpoint $Checkpoint `
                --dataset-root data\lerobot\pick_place_v2_native_bin_1000 `
                --repo-id local/ur5e_pick_place_v2_native_bin `
                --manifest configs\benchmarks\pick_place_screen_v1.json `
                --episodes 6 --replan-steps 8 --samples-per-plan $SamplesPerPlan `
                --policy-seed 1000 --control-mode vla_raw_safety `
                --output $Output 1> $Log 2> $ErrorLog
            $Code = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $PreviousErrorActionPreference
        }
        if ($Code -ne 0) {
            throw "Development screening failed for checkpoint $Step with exit code $Code"
        }
    }
}
finally {
    Pop-Location
}
