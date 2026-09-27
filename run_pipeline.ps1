$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "======================================================================"
Write-Host "THUNDERSTORM NOWCASTING PROTOTYPE PIPELINE"
Write-Host "======================================================================"

Write-Host ""
Write-Host "[1/6] Running dataset validation..."
python src\datasets\lightning_dataset.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Dataset validation FAILED."
    exit 1
}

Write-Host ""
Write-Host "[2/6] Running physics baseline..."
python src\inference\physics_nowcast.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Physics baseline FAILED."
    exit 1
}

Write-Host ""
Write-Host "[3/6] Running motion baseline..."
python src\inference\motion_nowcast.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Motion baseline FAILED."
    exit 1
}

Write-Host ""
Write-Host "[4/6] Running temporal multimodal inference..."
python src\inference\temporal_multimodal_nowcast.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Temporal multimodal inference FAILED."
    exit 1
}

Write-Host ""
Write-Host "[5/6] Building real-data training corpus..."
python src\preprocessing\build_training_corpus.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Training corpus build FAILED."
    exit 1
}

Write-Host ""
Write-Host "[6/6] Running quantitative validation..."
python src\validation\compare_nowcasts.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Quantitative validation FAILED."
    exit 1
}

Write-Host ""
Write-Host "======================================================================"
Write-Host "PIPELINE COMPLETE"
Write-Host "======================================================================"

Write-Host ""
Write-Host "Generated outputs:"
Write-Host "  - Physics nowcast"
Write-Host "  - Motion nowcast"
Write-Host "  - Temporal multimodal nowcast"
Write-Host "  - Real-data training corpus"
Write-Host "  - Quantitative comparison"

Write-Host ""
Write-Host "Training status:"
Write-Host "  ConvLSTM training is NOT executed."
Write-Host "  Current corpus contains only one independent event."
Write-Host "  Acquire additional real events before supervised training."
Write-Host ""
