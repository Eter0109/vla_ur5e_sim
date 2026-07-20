<#
Run the P3 A/B/C/D matrix serially.  It never reads test_v2 and only evaluates
completed 15k checkpoints on validation_v2.  Use -RunTraining only after A is
finished; the default mode is a safe resume/status pass.
#>
param(
    [string]$Python = "python",
    [string]$Dataset = "data\lerobot\expert_500demos",
    [string]$RepoId = "local/ur5e_custom_lift",
    [string]$OutputRoot = "outputs\p3_ablation_20260719",
    [switch]$RunTraining,
    [switch]$ResumeEvaluation
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Manifest = Join-Path $Root "data\manifests\validation_v2.json"
$Preset = Join-Path $Root "configs\legacy_20260719.json"
$OutputRoot = if ([IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot } else { Join-Path $Root $OutputRoot }

$Runs = [ordered]@{
    A = @{ Output = "outputs\smolvla_ablation_a_15k_seed1000"; PeakLR = 0.00002; Factor = 1; Window = 0; Existing = $true }
    B = @{ Output = "outputs\smolvla_ablation_b_15k_seed1000"; PeakLR = 0.00002; Factor = 3; Window = 5; Existing = $false }
    C = @{ Output = "outputs\smolvla_ablation_c_15k_seed1000"; PeakLR = 0.00005; Factor = 1; Window = 0; Existing = $false }
    D = @{ Output = "outputs\smolvla_fullexpert_cosine_15k"; PeakLR = 0.00005; Factor = 3; Window = 5; Existing = $true }
}

function Get-Checkpoint([hashtable]$Run) {
    return Join-Path $Root (Join-Path $Run.Output "checkpoints\015000\pretrained_model")
}

function Test-CompletedTraining([string]$Group, [hashtable]$Run) {
    $checkpoint = Get-Checkpoint $Run
    if (-not (Test-Path $checkpoint)) { return $false }
    if ($Group -eq "D") { return $true }
    $runRoot = Join-Path $Root $Run.Output
    return (Test-Path (Join-Path $runRoot "train.log")) -and (Test-Path (Join-Path $runRoot "run_manifest.json")) -and (Test-Path (Join-Path $runRoot "source.patch"))
}

function Invoke-Training([string]$Group, [hashtable]$Run) {
    if (Test-CompletedTraining $Group $Run) { return }
    if (-not $RunTraining) {
        Write-Warning "$Group is not complete. Re-run with -RunTraining after verifying no other training process owns its output."
        return
    }
    if ($Run.Existing) {
        throw "$Group should be an existing completed run, but its 15k checkpoint is missing: $(Get-Checkpoint $Run)"
    }
    $resumeArgs = @()
    if (Test-Path (Join-Path $Root (Join-Path $Run.Output "checkpoints"))) {
        $resumeArgs = @("-Resume")
    }
    & (Join-Path $PSScriptRoot "train_smolvla.ps1") `
        -Python $Python -Dataset $Dataset -RepoId $RepoId -Output $Run.Output `
        -Steps 15000 -BatchSize 8 -LearningRate $Run.PeakLR -WarmupSteps 1500 `
        -DecaySteps 15000 -DecayLR 0.000001 -SaveFreq 5000 -LogFreq 20 -Seed 1000 `
        -RotationLossWeight 0.01 -GripperLossWeight 2.0 `
        -TransitionOversampleFactor $Run.Factor -TransitionOversampleWindow $Run.Window `
        -FullExpert -WandbMode offline @resumeArgs
    if (-not (Test-CompletedTraining $Group $Run)) { throw "$Group training ended without a complete 15k provenance bundle." }
}

function Test-CompletedEvaluation([string]$Path) {
    if (-not (Test-Path $Path)) { return $false }
    $metadata = "$Path.meta.json"
    if (-not (Test-Path $metadata)) { return $false }
    $result = Get-Content -Raw $Path | ConvertFrom-Json
    $meta = Get-Content -Raw $metadata | ConvertFrom-Json
    return $result.Count -eq 40 -and $meta.status -eq "completed"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
foreach ($group in $Runs.Keys) {
    $run = $Runs[$group]
    Invoke-Training $group $run
    if (-not (Test-CompletedTraining $group $run)) {
        Write-Output "P3 is waiting for $group. No later training or evaluation was started."
        return
    }
    $rollout = Join-Path $OutputRoot "validation_$group.json"
    if (Test-CompletedEvaluation $rollout) { continue }
    $rolloutArgs = @(
        (Join-Path $PSScriptRoot "run_rollouts.py"), "--checkpoint", (Get-Checkpoint $run),
        "--dataset-root", $Dataset, "--repo-id", $RepoId, "--manifest", $Manifest,
        "--benchmark-role", "development", "--episodes", "40", "--experiment-config", $Preset,
        "--output", $rollout
    )
    if ($ResumeEvaluation -and (Test-Path $rollout)) { $rolloutArgs += "--resume" }
    & $Python @rolloutArgs
    if (-not (Test-CompletedEvaluation $rollout)) { throw "$group validation did not complete 40 scenes." }
}

$paths = @("A", "B", "C", "D") | ForEach-Object { Join-Path $OutputRoot "validation_$_.json" }
if (($paths | Where-Object { -not (Test-CompletedEvaluation $_) }).Count -eq 0) {
    & $Python (Join-Path $PSScriptRoot "summarize_p3_matrix.py") `
        --a $paths[0] --b $paths[1] --c $paths[2] --d $paths[3] `
        --output (Join-Path $OutputRoot "matrix_summary.json")
}
