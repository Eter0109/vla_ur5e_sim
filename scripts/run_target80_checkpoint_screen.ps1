param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [Parameter(Mandatory = $true)]
    [string]$Output,
    [string]$Python = "C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Checkpoint = (Resolve-Path $Checkpoint).Path
$Output = [IO.Path]::GetFullPath((Join-Path $Root $Output))
if (Test-Path $Output) {
    throw "Refusing to overwrite checkpoint screen output: $Output"
}
New-Item -ItemType Directory -Path $Output | Out-Null

$PushCommon = @(
    "--checkpoint", $Checkpoint,
    "--dataset-root", "data\lerobot\multitask_robust_push_1500",
    "--repo-id", "local/multitask_robust_push_1500",
    "--episodes", "20",
    "--replan-steps", "4",
    "--temporal-decay", "0.75",
    "--policy-seed", "1000",
    "--samples-per-plan", "1",
    "--overwrite-development"
)
$PickCommon = @(
    "--checkpoint", $Checkpoint,
    "--dataset-root", "data\lerobot\multitask_robust_pick_place_1500",
    "--repo-id", "local/multitask_robust_pick_place_1500",
    "--episodes", "20",
    "--temporal-ensemble-decay", "0.75",
    "--replan-steps", "4",
    "--samples-per-plan", "2",
    "--control-mode", "vla_action_calibrated",
    "--closed-negative-y-gain", "1.8",
    "--policy-seed", "1000",
    "--overwrite-development"
)

& $Python scripts\run_push_vla_only_benchmark.py @PushCommon `
    --manifest configs\benchmarks\push_robust_development_nominal_v1_screen20.json `
    --output (Join-Path $Output "push_nominal.json")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python scripts\run_push_vla_only_benchmark.py @PushCommon `
    --manifest configs\benchmarks\push_robust_development_randomized_v1_screen20.json `
    --output (Join-Path $Output "push_randomized.json")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python scripts\run_pick_place_vla_only.py @PickCommon `
    --manifest configs\benchmarks\pick_place_robust_development_nominal_v1_screen20.json `
    --output (Join-Path $Output "pick_nominal.json")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python scripts\run_pick_place_vla_only.py @PickCommon `
    --manifest configs\benchmarks\pick_place_robust_development_randomized_v1_screen20.json `
    --output (Join-Path $Output "pick_randomized.json")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$PushNominal = Get-Content (Join-Path $Output "push_nominal.json") -Raw | ConvertFrom-Json
$PushRandomized = Get-Content (Join-Path $Output "push_randomized.json") -Raw | ConvertFrom-Json
$PickNominal = Get-Content (Join-Path $Output "pick_nominal.json") -Raw | ConvertFrom-Json
$PickRandomized = Get-Content (Join-Path $Output "pick_randomized.json") -Raw | ConvertFrom-Json
# Windows PowerShell 5 can preserve a JSON array as one pipeline object inside
# @(...), silently counting it as one rollout. Iterate the deserialized array
# directly so both Windows PowerShell and PowerShell 7 count every scene.
$PickNominalSuccesses = 0
foreach ($Row in $PickNominal) { if ($Row.success) { $PickNominalSuccesses++ } }
$PickRandomizedSuccesses = 0
foreach ($Row in $PickRandomized) { if ($Row.success) { $PickRandomizedSuccesses++ } }
$Summary = [ordered]@{
    schema_version = 1
    checkpoint = $Checkpoint
    checkpoint_sha256 = $PushNominal.checkpoint_sha256
    scope = "screening_only_not_final_evidence"
    push_nominal = [ordered]@{ successes = $PushNominal.summary.successes; episodes = 20; rate = $PushNominal.summary.success_rate }
    push_randomized = [ordered]@{ successes = $PushRandomized.summary.successes; episodes = 20; rate = $PushRandomized.summary.success_rate }
    pick_nominal = [ordered]@{ successes = $PickNominalSuccesses; episodes = 20; rate = $PickNominalSuccesses / 20.0 }
    pick_randomized = [ordered]@{ successes = $PickRandomizedSuccesses; episodes = 20; rate = $PickRandomizedSuccesses / 20.0 }
}
$Summary | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $Output "screening_summary.json") -Encoding UTF8
$Summary | ConvertTo-Json -Depth 5
