param([string]$Python = "python", [string]$Checkpoint, [string]$OutputRoot)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
for ($index = 0; $index -lt 24; $index++) {
    $output = Join-Path $OutputRoot ("scene_{0:D2}.json" -f $index)
    if (Test-Path $output) { continue }
    & $Python (Join-Path $PSScriptRoot "run_pick_place_vla_only.py") `
        --checkpoint $Checkpoint --dataset-root "data\lerobot\pick_place_v2_native_bin_1000" `
        --repo-id "local/ur5e_pick_place_v2_native_bin" `
        --manifest "configs\benchmarks\pick_place_screen_v1.json" `
        --scene-index $index --replan-steps 8 --policy-seed 2000 --output $output
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
