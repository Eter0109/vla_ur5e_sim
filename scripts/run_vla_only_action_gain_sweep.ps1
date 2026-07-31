param(
    [string]$Python = "C:\Users\22680\anaconda3\envs\vla_sim_gpu\python.exe",
    [string]$ExperimentRoot = "outputs\pick_place_v2_native_bin\vla_only_action_gain_sweep",
    [int]$PolicySeed = 2000,
    [double]$TransportPositiveXGain = 1.0,
    [double[]]$NegativeYGains = @(1.3, 1.4, 1.5),
    [int]$FullCandidateLimit = 1,
    [string]$TestManifestName = "pick_place_test_v2_50.json",
    [int]$SamplesPerPlan = 1,
    [switch]$SkipTest
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ExperimentPath = Join-Path $Root $ExperimentRoot
$Dataset = Join-Path $Root "data\lerobot\pick_place_v2_native_bin_1000"
$ScreenManifest = Join-Path $Root "configs\benchmarks\pick_place_screen_v1.json"
$TestManifest = Join-Path $Root "configs\benchmarks\$TestManifestName"
$Checkpoint = Join-Path $Root (
    "outputs\pick_place_v2_native_bin\vla_only_global_20k\seed1000\" +
    "checkpoints\020000\pretrained_model"
)
$CandidateRoot = Join-Path $ExperimentPath "candidate_screens"
$ComparisonOutput = Join-Path $ExperimentPath "comparison.json"

New-Item -ItemType Directory -Force -Path $ExperimentPath | Out-Null
New-Item -ItemType Directory -Force -Path $CandidateRoot | Out-Null

function Read-EvaluationResults {
    param([string]$Path)

    $Parsed = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    foreach ($Result in $Parsed) {
        Write-Output $Result
    }
}

function Invoke-GainEvaluation {
    param(
        [double]$Gain,
        [string]$Manifest,
        [int]$Episodes,
        [string]$Output
    )
    if (Test-Path -LiteralPath $Output) {
        $ExistingFile = Get-Item -LiteralPath $Output
        if ($ExistingFile.Length -gt 2) {
            $Existing = @(Read-EvaluationResults -Path $Output)
            if ($Existing.Count -eq $Episodes) {
                return
            }
            throw "Existing evaluation is incomplete; refusing to overwrite: $Output"
        }
    }
    $StdoutLog = "${Output}.stdout.log"
    $StderrLog = "${Output}.stderr.log"
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python -B (Join-Path $PSScriptRoot "run_pick_place_vla_only.py") `
        --checkpoint $Checkpoint `
        --dataset-root $Dataset `
        --repo-id local/ur5e_pick_place_v2_native_bin `
        --manifest $Manifest `
        --episodes $Episodes `
        --policy-seed $PolicySeed `
        --closed-negative-y-gain $Gain `
        --transport-positive-x-gain $TransportPositiveXGain `
        --samples-per-plan $SamplesPerPlan `
        --output $Output `
        1> $StdoutLog `
        2> $StderrLog
    $PythonExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference
    if ($PythonExitCode -ne 0) {
        throw "Gain evaluation failed with exit code $PythonExitCode"
    }
}

$Candidates = @()
foreach ($Gain in $NegativeYGains) {
    $GainName = ([string]$Gain).Replace(".", "p")
    $Output = Join-Path $CandidateRoot "gain_${GainName}_screen8_seeded.json"
    Invoke-GainEvaluation -Gain $Gain -Manifest $ScreenManifest -Episodes 8 -Output $Output
    $Results = @(Read-EvaluationResults -Path $Output)
    $XYErrors = @(
        foreach ($Result in $Results) {
            [double]$Result.place_conditions.xy_error_m
        }
    )
    $Candidates += [pscustomobject]@{
        gain = $Gain
        successes = @($Results | Where-Object success).Count
        grasps = @($Results | Where-Object ever_grasped).Count
        mean_xy_error_m = [double](($XYErrors | Measure-Object -Average).Average)
    }
}

$RankedCandidates = @($Candidates |
    Sort-Object `
        @{ Expression = "successes"; Descending = $true }, `
        @{ Expression = "grasps"; Descending = $true }, `
        @{ Expression = "mean_xy_error_m"; Descending = $false } |
    Where-Object { $_.successes -eq ($Candidates | Measure-Object successes -Maximum).Maximum } |
    Select-Object -First $FullCandidateLimit)

$FullCandidates = @()
foreach ($Candidate in $RankedCandidates) {
    $GainName = ([string]$Candidate.gain).Replace(".", "p")
    $Output = Join-Path $ExperimentPath "gain_${GainName}_screen24_seeded.json"
    $LegacyOutput = Join-Path $ExperimentPath "best_gain_${GainName}_screen24_seeded.json"
    if (-not (Test-Path -LiteralPath $Output) -and (Test-Path -LiteralPath $LegacyOutput)) {
        $Output = $LegacyOutput
    }
    Invoke-GainEvaluation `
        -Gain $Candidate.gain -Manifest $ScreenManifest -Episodes 24 -Output $Output
    $Results = @(Read-EvaluationResults -Path $Output)
    $XYErrors = @(
        foreach ($Result in $Results) {
            [double]$Result.place_conditions.xy_error_m
        }
    )
    $FullCandidates += [pscustomobject]@{
        gain = $Candidate.gain
        successes = @($Results | Where-Object success).Count
        grasps = @($Results | Where-Object ever_grasped).Count
        mean_xy_error_m = [double](($XYErrors | Measure-Object -Average).Average)
        result = $Output
    }
}

$Best = $FullCandidates |
    Sort-Object `
        @{ Expression = "successes"; Descending = $true }, `
        @{ Expression = "grasps"; Descending = $true }, `
        @{ Expression = "mean_xy_error_m"; Descending = $false } |
    Select-Object -First 1
$Best24Output = $Best.result
$Best24 = @(Read-EvaluationResults -Path $Best24Output)
$Best24Successes = $Best.successes
$BestGainName = ([string]$Best.gain).Replace(".", "p")

$Test50Output = $null
$Test50Successes = $null
if ($Best24Successes -ge 22 -and -not $SkipTest) {
    $Test50Output = Join-Path $ExperimentPath "best_gain_${BestGainName}_test50_seeded.json"
    Invoke-GainEvaluation `
        -Gain $Best.gain -Manifest $TestManifest -Episodes 50 -Output $Test50Output
    $Test50 = @(Read-EvaluationResults -Path $Test50Output)
    $Test50Successes = @($Test50 | Where-Object success).Count
}

[ordered]@{
    policy_seed = $PolicySeed
    samples_per_plan = $SamplesPerPlan
    checkpoint = $Checkpoint
    test_manifest = $TestManifest
    calibration = [ordered]@{
        method = "model-action horizontal-transport direction lock; no object or target pose"
        transport_positive_x_gain = $TransportPositiveXGain
    }
    candidates_screen8 = $Candidates
    candidates_screen24 = $FullCandidates
    selected = [ordered]@{
        gain = $Best.gain
        successes = $Best24Successes
        grasps = @($Best24 | Where-Object ever_grasped).Count
        episodes = $Best24.Count
        result = $Best24Output
    }
    promotion_gate = [ordered]@{
        required_screen24_successes = 22
        passed = $Best24Successes -ge 22
    }
    test50 = if ($Test50Output) {
        [ordered]@{
            successes = $Test50Successes
            episodes = 50
            result = $Test50Output
        }
    } else {
        $null
    }
} | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 $ComparisonOutput
Write-Output "comparison=$ComparisonOutput"
