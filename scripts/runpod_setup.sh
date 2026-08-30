#!/bin/bash
# RunPod setup script for Audio Flamingo 3 / Music Flamingo inference
# Idempotent — safe to re-run; skips steps already completed.
#
# Usage:
#   1. Upload project to /workspace/music-evalkit on RunPod
#   2. Run: cd /workspace/music-evalkit && bash scripts/runpod_setup.sh

set -e  # Exit on error

echo "=== RunPod Setup for music-evalkit ==="

# Install uv to persistent volume (survives pod restarts)
UV_INSTALL_DIR="/workspace/.local/bin"
export PATH="$UV_INSTALL_DIR:$PATH"
if ! [ -f "$UV_INSTALL_DIR/uv" ]; then
    echo "Installing uv to $UV_INSTALL_DIR..."
    curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
else
    echo "uv: already installed"
fi

# Store HuggingFace models on /workspace (network storage, not the small system disk)
export HF_HOME=/workspace/hf_cache

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.11
else
    echo "venv: already exists"
fi
source .venv/bin/activate

# Install PyTorch if not already importable
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo "PyTorch: already installed ($(python -c 'import torch; print(torch.__version__)'))"
fi

# Install flash-attn if not already importable
# Prebuilt wheel required: building from source produces ABI mismatch (CXX11_ABI=True vs PyTorch's False)
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
if ! python -c "from flash_attn.flash_attn_interface import flash_attn_func" 2>/dev/null; then
    echo "Installing flash-attn (prebuilt wheel)..."
    if ! uv pip install "$FLASH_ATTN_WHEEL"; then
        echo "WARNING: flash-attn install failed, continuing without it (inference will be slower)"
    fi
else
    echo "flash-attn: already installed"
fi

# Install accelerate if not already importable
if ! python -c "import accelerate" 2>/dev/null; then
    echo "Installing accelerate..."
    uv pip install accelerate
else
    echo "accelerate: already installed"
fi

# Install peft if not already importable
if ! python -c "import peft" 2>/dev/null; then
    echo "Installing peft..."
    uv pip install peft
else
    echo "peft: already installed"
fi

# Install project (always re-run — fast if nothing changed)
echo "Installing music-evalkit..."
uv pip install -e .

# Verify installation
echo ""
echo "=== Verifying installation ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import flash_attn; print('flash-attn: OK')" || echo "WARNING: flash-attn not available, inference will use standard attention"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "from music_evalkit.models.providers.flamingo import FlamingoClient; print('FlamingoClient: OK')"

# Pre-download model weights (skips if already cached)
echo ""
echo "=== Pre-downloading model weights ==="
MODELS=("nvidia/audio-flamingo-3-hf" "nvidia/music-flamingo-hf")
for model in "${MODELS[@]}"; do
    cache_name=$(echo "$model" | sed 's|/|--|g')
    if [ -d "${HF_HOME}/hub/models--${cache_name}/snapshots" ]; then
        echo "${model}: already cached"
    else
        echo "Downloading ${model}..."
        python -c "from huggingface_hub import snapshot_download; snapshot_download('${model}')"
    fi
done

# Mark setup as complete (used by runpod_run.py to detect prior setup)
touch .setup_complete

echo ""
echo "=== Setup complete! ==="
echo "Preprocess data with:"
echo "  python scripts/runpod_run.py preprocess"
echo "Run inference with:"
echo "  python scripts/runpod_run.py run --task all --provider audio_flamingo"
