#!/usr/bin/env bash
# Bootstrap the embodied-navigation research environment in user space (no sudo, no system changes).
#   - Miniforge (conda + mamba) under ~/miniforge3; conda-forge goes through the TUNA mirror
#   - Python 3.9 env "embnav": habitat-sim 0.3.3 (headless + bullet) and PyTorch 2.8
#     (the PyPI build targets CUDA 12.8, which ships sm_120 kernels for the RTX 5060)
# Re-runnable: finished steps are skipped.
set -eo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-embnav}"
PIP_MIRROR="${PIP_MIRROR:-https://mirror.nju.edu.cn/pypi/web/simple/}"
MINIFORGE_URL="https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh"

log() { echo "[setup_env $(date +%H:%M:%S)] $*"; }

# 1. Miniforge (installed without touching ~/.bashrc)
if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
  log "Downloading Miniforge"
  tmp="$(mktemp -d)"
  curl -fL --retry 3 -o "$tmp/miniforge.sh" "$MINIFORGE_URL"
  bash "$tmp/miniforge.sh" -b -p "$CONDA_ROOT"
  rm -rf "$tmp"
fi

# conda-forge via TUNA; the aihabitat channel has no mirror and stays on anaconda.org.
cat > "$CONDA_ROOT/.condarc" <<'EOF'
channels:
  - conda-forge
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
channel_priority: flexible
show_channel_urls: true
EOF

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# 2. Env with habitat-sim (only py3.9 builds exist for 0.3.x)
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "Creating env $ENV_NAME with habitat-sim 0.3.3"
  conda create -y -n "$ENV_NAME" -c aihabitat -c conda-forge \
    python=3.9 "habitat-sim=0.3.3=py3.9_headless_bullet_linux_*" "numpy<2"
fi
conda activate "$ENV_NAME"

# Env-local pip mirror (writes $CONDA_PREFIX/pip.conf, not ~/.config/pip)
pip config --site set global.index-url "$PIP_MIRROR"

# uv in the base env: parallel downloads (single pip connections can stall at ~200 KB/s here)
[ -x "$CONDA_ROOT/bin/uv" ] || "$CONDA_ROOT/bin/pip" install -q -i "$PIP_MIRROR" uv
export UV_INDEX_URL="$PIP_MIRROR" UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=8

# 3. PyTorch
if ! python -c "import torch" 2>/dev/null; then
  log "Installing PyTorch 2.8 (CUDA 12.8)"
  "$CONDA_ROOT/bin/uv" pip install --python "$(command -v python)" "torch==2.8.0" "torchvision==0.23.0" "numpy<2"
fi

log "Smoke test"
python - <<'EOF'
import torch, habitat_sim
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda")
    print("cuda matmul ok:", bool(torch.isfinite((x @ x).sum())))
print("habitat_sim", habitat_sim.__version__, "cuda_enabled", habitat_sim.cuda_enabled)
EOF
log "Done"
