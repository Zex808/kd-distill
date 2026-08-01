#!/bin/bash
# setup.sh — Linux/Mac: create venv, install, launch
# Run: chmod +x setup.sh && ./setup.sh
# Re-launch only: ./setup.sh --skip-install

set -euo pipefail

SKIP_INSTALL=0
[[ "${1:-}" == "--skip-install" ]] && SKIP_INSTALL=1

if [[ $SKIP_INSTALL -eq 0 ]]; then
    python3 -m venv .venv
    source .venv/bin/activate

    # PyTorch CUDA 12.1 — change cu121 → cpu if no GPU
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

    pip install -r requirements.txt -q

    # Ollama (optional)
    # curl -fsSL https://ollama.com/install.sh | sh
else
    source .venv/bin/activate
fi

mkdir -p data checkpoints

echo "✅ Launching KD Pipeline at http://localhost:7860"
python app.py
