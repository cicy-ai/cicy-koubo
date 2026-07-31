#!/usr/bin/env bash
# Swap one authorized/synthetic source face onto a target video.
# Usage:
#   bash scripts/provision/faceswap/run.sh FACE.png TARGET.mp4 OUTPUT.mp4
set -Eeuo pipefail

[[ $# -eq 3 ]] || {
  echo "Usage: $0 FACE_IMAGE TARGET_VIDEO OUTPUT_VIDEO" >&2
  exit 2
}

SOURCE="$(realpath "$1")"
TARGET="$(realpath "$2")"
OUTPUT="$(realpath -m "$3")"
WORK="${FACEFUSION_WORK:-/content/faceswap}"
REPO="$WORK/facefusion"
PY="$WORK/env/bin/python"
ENTRY="$REPO/facefusion.py"
TMP="$WORK/output-$(date +%s)-$$.mp4"

step() { printf '\n=== [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

[[ -f "$WORK/READY" ]] || die "FaceFusion is not installed; run provision.sh first"
[[ -x "$PY" && -f "$ENTRY" ]] || die "FaceFusion runtime is incomplete"
[[ -s "$SOURCE" ]] || die "Source face is missing or empty: $SOURCE"
[[ -s "$TARGET" ]] || die "Target video is missing or empty: $TARGET"
mkdir -p "$(dirname "$OUTPUT")"
rm -f "$TMP"

step "1/4 Validate source and target"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
  -of default=nw=1 "$TARGET"

step "2/4 FaceFusion GPU inference"
cd "$REPO"
"$PY" "$ENTRY" headless-run \
  --source-paths "$SOURCE" \
  --target-path "$TARGET" \
  --output-path "$TMP" \
  --processors face_swapper face_enhancer \
  --face-swapper-model inswapper_128_fp16 \
  --face-enhancer-model gfpgan_1.4 \
  --face-enhancer-blend 65 \
  --face-selector-mode one \
  --face-mask-types box occlusion \
  --execution-providers cuda \
  --execution-thread-count 4 \
  --output-video-fps 25 \
  --skip-audio

[[ -s "$TMP" ]] || die "FaceFusion produced no output"

step "3/4 Normalize 9:16, 1080x1920, 25fps, no audio"
ffmpeg -y -hide_banner -loglevel error -i "$TMP" \
  -vf "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=25" \
  -an -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p -movflags +faststart \
  "$OUTPUT"
rm -f "$TMP"

step "4/4 Verify output"
DURATION="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUTPUT")"
FRAMES="$(ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=nb_read_frames -of csv=p=0 "$OUTPUT")"
"$PY" - "$DURATION" "$FRAMES" <<'PY'
import sys
duration = float(sys.argv[1])
frames = int(sys.argv[2])
if duration <= 0 or frames <= 0:
    raise SystemExit("invalid output media")
print(f"verified duration={duration:.3f}s frames={frames}")
PY

echo "OUTPUT=$OUTPUT"
