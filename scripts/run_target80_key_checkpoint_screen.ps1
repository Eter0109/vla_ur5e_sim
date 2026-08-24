param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [Parameter(Mandatory = $true)]
    [string]$Output,
    [string]$Python = "C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe",
    [string]$Reference = "outputs\multitask_robust\target80_screen_reference_lazy_bs2_final.json"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Checkpoint = (Resolve-Path $Checkpoint).Path
$Output = [IO.Path]::GetFullPath((Join-Path $Root $Output))
$Reference = [IO.Path]::GetFullPath((Join-Path $Root $Reference))
if (-not (Test-Path $Reference)) {
    throw "Target-80 screen reference is missing: $Reference"
}
$ReferenceData = Get-Content $Reference -Raw | ConvertFrom-Json
$PushReference = [int]$ReferenceData.splits.push_randomized.successes
$PickReference = [int]$ReferenceData.splits.pick_nominal.successes
if (Test-Path $Output) {
    throw "Refusing to overwrite key checkpoint screen output: $Output"
}
New-Item -ItemType Directory -Path $Output | Out-Null

& $Python scripts\run_push_vla_only_benchmark.py `
    --checkpoint $Checkpoint `
    --dataset-root data\lerobot\multitask_robust_push_1500 `
    --repo-id local/multitask_robust_push_1500 `
    --manifest configs\benchmarks\push_robust_development_randomized_v1_screen20.json `
    --episodes 20 --replan-steps 4 --temporal-decay 0.75 `
    --policy-seed 1000 --samples-per-plan 1 `
    --output (Join-Path $Output "push_randomized.json") --overwrite-development
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python scripts\run_pick_place_vla_only.py `
    --checkpoint $Checkpoint `
    --dataset-root data\lerobot\multitask_robust_pick_place_1500 `
    --repo-id local/multitask_robust_pick_place_1500 `
    --manifest configs\benchmarks\pick_place_robust_development_nominal_v1_screen20.json `
    --episodes 20 --temporal-ensemble-decay 0.75 --replan-steps 4 `
    --samples-per-plan 2 --control-mode vla_action_calibrated `
    --closed-negative-y-gain 1.8 --policy-seed 1000 `
    --output (Join-Path $Output "pick_nominal.json") --overwrite-development
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Push = Get-Content (Join-Path $Output "push_randomized.json") -Raw | ConvertFrom-Json
$Pick = Get-Content (Join-Path $Output "pick_nominal.json") -Raw | ConvertFrom-Json
# Do not wrap this in @(...): Windows PowerShell 5 would count the complete
# JSON array as a single item instead of enumerating the individual scenes.
$PickSuccesses = 0
foreach ($Row in $Pick) { if ($Row.success) { $PickSuccesses++ } }
$Promoted = (
    [int]$Push.summary.successes -ge $PushReference -and
    $PickSuccesses -ge $PickReference -and
    ([int]$Push.summary.successes -gt $PushReference -or $PickSuccesses -gt $PickReference)
)
$Summary = [ordered]@{
    schema_version = 1
    checkpoint = $Checkpoint
    checkpoint_sha256 = $Push.checkpoint_sha256
    scope = "key_screening_only_not_final_evidence"
    promoted_to_four_split_screen = $Promoted
    reference = [ordered]@{
        push_randomized = $ReferenceData.splits.push_randomized
        pick_nominal = $ReferenceData.splits.pick_nominal
    }
    candidate = [ordered]@{
        push_randomized = [ordered]@{ successes = $Push.summary.successes; episodes = 20; rate = $Push.summary.success_rate }
        pick_nominal = [ordered]@{ successes = $PickSuccesses; episodes = 20; rate = $PickSuccesses / 20.0 }
    }
}
$Summary | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $Output "key_screening_summary.json") -Encoding UTF8
$Summary | ConvertTo-Json -Depth 6
