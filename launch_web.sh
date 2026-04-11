#!/bin/bash
# MedSAM2 Web Interface launcher
# Usage: bash launch_web.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== MedSAM2 Web Interface ==="

# Check for checkpoints
PT_FILES=$(ls checkpoints/*.pt 2>/dev/null | wc -l | tr -d ' ')
if [ "$PT_FILES" -eq 0 ]; then
  echo ""
  echo "  No model checkpoints found!"
  echo "  Downloading MedSAM2_latest.pt from HuggingFace..."
  echo ""
  mkdir -p checkpoints
  HF_URL="https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"
  if command -v curl > /dev/null 2>&1; then
    curl -L -o checkpoints/MedSAM2_latest.pt "$HF_URL" --progress-bar
  else
    wget -P checkpoints "$HF_URL"
  fi
  echo ""
fi

echo ""
echo "  Starting server at http://localhost:9000"
echo "  Device: $(python3 -c 'import torch; print("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")' 2>/dev/null)"
echo "  Press Ctrl+C to stop"
echo ""

python3 medsam_app/main.py
