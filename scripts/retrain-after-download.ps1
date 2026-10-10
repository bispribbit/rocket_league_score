# Waits for a running ballchasing_downloader to exit, then rebuilds the training CSVs and the
# shipped model bundle (data/skill_model.bin). Logs go to target/retrain_after_download.log.
#
# Usage (from the repository root, detached so it survives the terminal):
#   Start-Process powershell -ArgumentList "-NoProfile -File scripts/retrain-after-download.ps1" -WindowStyle Hidden

$ErrorActionPreference = "Stop"
$log = "target/retrain_after_download.log"

function Write-Log([string]$message) {
    "$(Get-Date -Format o) $message" | Out-File -FilePath $log -Append -Encoding utf8
}

Write-Log "waiting for ballchasing_downloader to finish"
while (Get-Process ballchasing_downloader -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 300
}

Write-Log "downloader finished; extracting stats"
cargo run --release -p skill_model_training --bin extract_stats *>> $log
if ($LASTEXITCODE -ne 0) { Write-Log "extract_stats failed ($LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Log "training"
cargo run --release -p skill_model_training --bin train *>> $log
if ($LASTEXITCODE -ne 0) { Write-Log "train failed ($LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Log "checking the bundle on the sample replays"
cargo test --release -p skill_model_training --test analyze_replays *>> $log
Write-Log "done (analyze_replays exit code $LASTEXITCODE)"
