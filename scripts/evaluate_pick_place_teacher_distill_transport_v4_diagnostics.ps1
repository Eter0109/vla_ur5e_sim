param(
    [string]$Python = "C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe",
    [string]$RunRoot = "outputs\pick_place_v2_native_bin\teacher_distill_transport_v4_3k\seed1000"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Runs = @(
    @{ Step = "002000"; Samples = 2 },
    @{ Step = "000500"; Samples = 1 },
    @{ Step = "000500"; Samples = 2 }
)
Push-Location $Root
try {
    foreach ($Run in $Runs) {
        $Step = $Run.Step
        $Samples = $Run.Samples
        $Checkpoint = Join-Path $RunRoot "checkpoints\$Step\pretrained_model"
        $Development = Join-Path $RunRoot "development"
        $Stem = "screen6_${Step}_samples${Samples}_replan8_seed1000"
        $Output = Join-Path $Development "$Stem.json"
        if (Test-Path $Output) {
            Write-Output "skip_existing checkpoint=$Step samples=$Samples"
            continue
        }
        if (-not (Test-Path $Checkpoint)) {
            throw "Missing checkpoint: $Checkpoint"
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
                --episodes 6 --replan-steps 8 --samples-per-plan $Samples `
                --policy-seed 1000 --control-mode vla_raw_safety `
                --output $Output `
                1> (Join-Path $Development "$Stem.log") `
                2> (Join-Path $Development "$Stem.err.log")
            $Code = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $PreviousErrorActionPreference
        }
        if ($Code -ne 0) {
            throw "Diagnostic failed for checkpoint $Step samples=$Samples with exit code $Code"
        }
    }
}
finally {
    Pop-Location
}
