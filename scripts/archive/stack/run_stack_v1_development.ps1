# Archived Stack v1 launcher; it depends on the removed legacy run_rollouts.py.
param(
    [string]$Python = "python",
    [string]$Dataset = "data\lerobot\stack_v1_3000",
    [string]$OutputRoot = "outputs\stack_v1",
    [int]$WaitForTrainingPid = 0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$RepoId = "local/ur5e_stack_v1"
$ScreenManifest = "configs\benchmarks\stack_screen_v1.json"
$DevManifest = "configs\benchmarks\stack_dev_v1.json"
$ExperimentConfig = "configs\stack_v1.json"
$EvaluationRoot = Join-Path $OutputRoot "evaluations"

function Invoke-StackRollout {
    param(
        [string]$Checkpoint,
        [string]$Manifest,
        [int]$Episodes,
        [string]$Output
    )
    & $Python scripts/run_rollouts.py `
        --checkpoint $Checkpoint `
        --experiment-config $ExperimentConfig `
        --dataset-root $Dataset `
        --repo-id $RepoId `
        --manifest $Manifest `
        --episodes $Episodes `
        --horizon 250 `
        --benchmark-role development `
        --output $Output
    if ($LASTEXITCODE -ne 0) {
        throw "Rollout failed: $Output"
    }
}

if ($WaitForTrainingPid -gt 0) {
    if (Get-Process -Id $WaitForTrainingPid -ErrorAction SilentlyContinue) {
        Write-Host "Waiting for training process $WaitForTrainingPid..."
        Wait-Process -Id $WaitForTrainingPid -ErrorAction Stop
    }
    else {
        Write-Host "Training process $WaitForTrainingPid already exited; checking checkpoints."
    }
}

$SelectedDevelopmentResults = @()
foreach ($Seed in 1000, 1001, 1002) {
    $SeedRoot = Join-Path $OutputRoot "seed$Seed"
    $CheckpointRoot = Join-Path $SeedRoot "checkpoints"
    $FinalCheckpoint = Join-Path $CheckpointRoot "040000\pretrained_model\config.json"
    if (-not (Test-Path $FinalCheckpoint)) {
        throw "Seed $Seed did not produce its final checkpoint; refusing development evaluation."
    }
    $CheckpointDirs = @(Get-ChildItem -LiteralPath $CheckpointRoot -Directory | Sort-Object Name)
    if ($CheckpointDirs.Count -eq 0) {
        throw "No checkpoints found for seed $Seed."
    }
    $SeedEvaluationRoot = Join-Path $EvaluationRoot "seed$Seed"
    $ScreenRoot = Join-Path $SeedEvaluationRoot "screen"
    $DevRoot = Join-Path $SeedEvaluationRoot "development"
    New-Item -ItemType Directory -Force -Path $ScreenRoot, $DevRoot | Out-Null

    $ScreenResults = @()
    foreach ($CheckpointDir in $CheckpointDirs) {
        $Checkpoint = Join-Path $CheckpointDir.FullName "pretrained_model"
        if (-not (Test-Path (Join-Path $Checkpoint "config.json"))) {
            continue
        }
        $ScreenResult = Join-Path $ScreenRoot "screen_step$($CheckpointDir.Name).json"
        Invoke-StackRollout -Checkpoint $Checkpoint -Manifest $ScreenManifest -Episodes 24 -Output $ScreenResult
        $ScreenResults += $ScreenResult
    }
    if ($ScreenResults.Count -lt 3) {
        throw "Need at least three valid checkpoints for seed $Seed."
    }

    $ScreenSelection = Join-Path $SeedEvaluationRoot "screen_selection.json"
    & $Python scripts/select_stack_checkpoints.py @ScreenResults --top 3 --output $ScreenSelection
    if ($LASTEXITCODE -ne 0) {
        throw "Screen checkpoint selection failed for seed $Seed."
    }
    $TopScreenResults = @(Get-Content -Raw -LiteralPath $ScreenSelection | ConvertFrom-Json)
    if ($TopScreenResults.Count -ne 3) {
        throw "Expected three screened checkpoints for seed $Seed."
    }

    $DevResults = @()
    foreach ($Entry in $TopScreenResults) {
        $Match = [regex]::Match(([string]$Entry.path), "screen_step(\\d+)\\.json$")
        if (-not $Match.Success) {
            throw "Cannot recover checkpoint step from $($Entry.path)"
        }
        $Step = $Match.Groups[1].Value
        $Checkpoint = Join-Path $CheckpointRoot "$Step\pretrained_model"
        $DevResult = Join-Path $DevRoot "dev_step$Step.json"
        Invoke-StackRollout -Checkpoint $Checkpoint -Manifest $DevManifest -Episodes 120 -Output $DevResult
        $DevResults += $DevResult
    }

    $DevSelection = Join-Path $SeedEvaluationRoot "development_selection.json"
    & $Python scripts/select_stack_checkpoints.py @DevResults --top 1 --output $DevSelection
    if ($LASTEXITCODE -ne 0) {
        throw "Development checkpoint selection failed for seed $Seed."
    }
    $BestDev = @(Get-Content -Raw -LiteralPath $DevSelection | ConvertFrom-Json)
    if ($BestDev.Count -ne 1) {
        throw "Expected exactly one selected development result for seed $Seed."
    }
    $SelectedDevelopmentResults += [string]$BestDev[0].path
}

$Promotion = Join-Path $EvaluationRoot "promotion.json"
& $Python scripts/check_stack_promotion.py @SelectedDevelopmentResults --output $Promotion
if ($LASTEXITCODE -ne 0) {
    throw "Development promotion gate did not pass. Blind evaluation remains locked."
}
Write-Host "Development promotion passed: $Promotion"
