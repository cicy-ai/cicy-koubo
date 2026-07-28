#!/bin/bash
set -uo pipefail

BASE=https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx

download() {
  local url=$1 out=$2 part="${2}.part" attempt=1
  mkdir -p "$(dirname "$out")"
  [ -s "$out" ] && { echo "cache hit: $out"; return 0; }
  while true; do
    echo "download: $out (attempt $attempt/5)"
    if curl -fL --connect-timeout 15 --max-time 0 --retry 3 --retry-delay 2 \
      --retry-all-errors -C - -o "$part" "$url"; then
      mv -f "$part" "$out"
      return 0
    fi
    [ "$attempt" -ge 5 ] && return 1
    attempt=$((attempt + 1))
    sleep $((attempt * 3))
  done
}

export -f download

TMP_LIST=$(mktemp)
trap 'rm -f "$TMP_LIST"' EXIT
cat > "$TMP_LIST" <<EOF
$BASE/face_attr_epoch_12_220318.onnx face_attr_detect/face_attr_epoch_12_220318.onnx
$BASE/pfpld_robust_sim_bs1_8003.onnx face_detect_utils/resources/pfpld_robust_sim_bs1_8003.onnx
$BASE/scrfd_500m_bnkps_shape640x640.onnx face_detect_utils/resources/scrfd_500m_bnkps_shape640x640.onnx
$BASE/model_float32.onnx face_detect_utils/resources/model_float32.onnx
$BASE/dinet_v1_20240131.pth landmark2face_wy/checkpoints/anylang/dinet_v1_20240131.pth
$BASE/79999_iter.onnx pretrain_models/face_lib/face_parsing/79999_iter.onnx
$BASE/GFPGANv1.4.onnx pretrain_models/face_lib/face_restore/gfpgan/GFPGANv1.4.onnx
$BASE/xseg_211104_4790000.onnx xseg/xseg_211104_4790000.onnx
$BASE/wenetmodel.pt wenet/examples/aishell/aidata/exp/conformer/wenetmodel.pt
EOF

xargs -P 4 -n 2 bash -c 'download "$0" "$1"' < "$TMP_LIST"
