#!/usr/bin/env bash
# Install FaceFusion into an isolated Colab environment.
# Usage:
#   bash scripts/provision/faceswap/provision.sh
set -Eeuo pipefail

WORK="${FACEFUSION_WORK:-/content/faceswap}"
REPO="$WORK/facefusion"
ENV="$WORK/env"
REF="${FACEFUSION_REF:-master}"
LOG="$WORK/provision.log"

mkdir -p "$WORK"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n=== [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
retry() {
  local attempt=1
  until "$@"; do
    if (( attempt >= 5 )); then return 1; fi
    echo "Retry $attempt/5 failed; waiting $((attempt * 3))s"
    sleep $((attempt * 3))
    attempt=$((attempt + 1))
  done
}

command -v nvidia-smi >/dev/null || die "NVIDIA GPU is required"
command -v ffmpeg >/dev/null || die "ffmpeg is required"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
step "1/5 GPU: $GPU"

step "2/5 Clone FaceFusion"
if [[ ! -d "$REPO/.git" ]]; then
  retry git clone --filter=blob:none https://github.com/facefusion/facefusion.git "$REPO" \
    || die "FaceFusion clone failed"
fi
git -C "$REPO" fetch --tags --prune
git -C "$REPO" checkout "$REF"
git -C "$REPO" pull --ff-only || true
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
echo "FaceFusion commit: $COMMIT"

step "3/5 Create isolated Python 3.10 environment"
if [[ ! -x "$ENV/bin/python" ]]; then
  python3 -m venv "$ENV"
fi
"$ENV/bin/python" -m pip install --upgrade pip setuptools wheel

step "4/5 Install CUDA runtime and FaceFusion dependencies"
cd "$REPO"
if [[ -f facefusion.py && -f install.py ]]; then
  retry "$ENV/bin/python" install.py --onnxruntime cuda --skip-conda \
    || die "FaceFusion CUDA installation failed"
elif [[ -f pyproject.toml ]]; then
  retry "$ENV/bin/pip" install -e . \
    || die "FaceFusion package installation failed"
  retry "$ENV/bin/pip" install "onnxruntime-gpu>=1.19,<2" \
    || die "onnxruntime-gpu installation failed"
else
  die "Unsupported FaceFusion repository layout"
fi

step "5/5 Runtime self-check"
ENTRY="$REPO/facefusion.py"
[[ -f "$ENTRY" ]] || die "facefusion.py is missing"
"$ENV/bin/python" "$ENTRY" --help >/dev/null
"$ENV/bin/python" - <<'PY'
import onnxruntime as ort
providers = ort.get_available_providers()
print("ONNX providers:", providers)
if "CUDAExecutionProvider" not in providers:
    raise SystemExit("CUDAExecutionProvider is unavailable")
PY

cat > "$WORK/READY" <<EOF
installed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
facefusion_commit=$COMMIT
gpu=$GPU
python=$("$ENV/bin/python" -V 2>&1)
EOF

step "DONE: $WORK/READY"
