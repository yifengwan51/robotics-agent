#!/usr/bin/env bash
# Second-stage envs (run after setup_env.sh):
#   embnav_thor : Python 3.9 + AI2-THOR 5.0.0 for EB-Navigation (simulator + OpenAI-style client only; no torch)
#   vlm         : Python 3.12 + vLLM for serving the local VLM over an OpenAI-compatible API
# Set SKIP_VLM=1 to build only embnav_thor.
set -eo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
PIP_MIRROR="${PIP_MIRROR:-https://mirror.nju.edu.cn/pypi/web/simple/}"
VLLM_SPEC="${VLLM_SPEC:-vllm==0.32.0}"
export UV_INDEX_URL="$PIP_MIRROR" UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=8
UV="$CONDA_ROOT/bin/uv"
log() { echo "[setup_env2 $(date +%H:%M:%S)] $*"; }

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
[ -x "$UV" ] || "$CONDA_ROOT/bin/pip" install -q -i "$PIP_MIRROR" uv

make_env() {  # name python_version
  if ! conda env list | awk '{print $1}' | grep -qx "$1"; then
    log "Creating env $1 (python $2)"
    conda create -y -n "$1" -c conda-forge "python=$2" pip
  fi
  conda activate "$1"
  pip config --site set global.index-url "$PIP_MIRROR"
}

make_env embnav_thor 3.9
if ! python -c "import ai2thor, cyclonedds, pytest" 2>/dev/null; then
  log "Installing AI2-THOR stack + DDS bridge deps"
  "$UV" pip install --python "$(command -v python)" "ai2thor==5.0.0" "gym==0.26.2" "numpy==1.26.4" \
    "openai>=1.58,<2" pydantic typing_extensions pillow opencv-python-headless hydra-core omegaconf pyyaml \
    tqdm matplotlib "cyclonedds==0.10.2" pytest
fi
python -c "import ai2thor, openai, cyclonedds; print('ai2thor', ai2thor.__version__, 'openai', openai.__version__)"
conda deactivate

if [ "${SKIP_VLM:-0}" != 1 ]; then
  make_env vlm 3.12
  if ! python -c "import vllm" 2>/dev/null; then
    log "Installing $VLLM_SPEC"
    "$UV" pip install --python "$(command -v python)" "$VLLM_SPEC" openai pillow
  fi
  python - <<'EOF'
import torch, vllm
print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available(),
      "cap", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
EOF
fi
log "Done"
