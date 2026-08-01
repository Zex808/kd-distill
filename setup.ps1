# setup.ps1 — Windows: create venv, install deps, launch UI
# Run: .\setup.ps1
# Re-launch only: .\setup.ps1 -SkipInstall

param([switch]$SkipInstall)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $SkipInstall) {
    # Create venv
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

    # PyTorch with CUDA 12.1 — change cu121 → cpu if no GPU
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet

    # Remaining deps
    pip install -r requirements.txt --quiet

    # Ollama (optional — comment out if not using)
    # winget install Ollama.Ollama
} else {
    .\.venv\Scripts\Activate.ps1
}

# Create data dirs if missing
New-Item -ItemType Directory -Force -Path data, checkpoints | Out-Null

# Launch
Write-Host "✅ Launching KD Pipeline at http://localhost:7860" -ForegroundColor Green
python app.py
