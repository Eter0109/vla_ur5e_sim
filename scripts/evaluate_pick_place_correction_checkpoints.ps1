param(
    [string]$Python = "python",
    [string]$RunRoot = "outputs\pick_place_v2_native_bin\correction_negative_y_6k\seed1000",
    [string]$DatasetRoot = "data\lerobot\pick_place_v2_native_bin_1000",
    [string]$Manifest = "configs\benchmarks\pick_place_screen_v1.json",
    [int]$PolicySeed = 2000
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = (Get-Command $Python -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$checkpoints = Get-ChildItem (Join-Path $RunRoot "checkpoints") -Directory |
    Sort-Object Name
if (-not $checkpoints) {
    throw "No numbered checkpoints found under $RunRoot"
}

foreach ($checkpoint in $checkpoints) {
    $output = Join-Path $RunRoot "development\screen24_$($checkpoint.Name)_seed$PolicySeed.json"
    if (Test-Path $output) {
        Write-Host "skip_existing checkpoint=$($checkpoint.Name)"
        continue
    }
    & $Python (Join-Path $PSScriptRoot "run_pick_place_vla_only.py") `
        --checkpoint (Join-Path $checkpoint.FullName "pretrained_model") `
        --dataset-root $DatasetRoot `
        --repo-id "local/ur5e_pick_place_v2_native_bin" `
        --manifest $Manifest --episodes 24 --policy-seed $PolicySeed --output $output
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
