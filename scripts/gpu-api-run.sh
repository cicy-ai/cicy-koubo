#!/usr/bin/env bash
set -euo pipefail

: "${CICY_KOUBO_GPU_IMAGE:=cicy-koubo-gpu:2026.07.29-api}"
ENV_FILE=/etc/cicy-koubo/api.env
STATE_DIR=/var/lib/cicy-koubo-api
CURRENT=/opt/cicy-koubo-api/current

test -r "$ENV_FILE"
test -d "$STATE_DIR"

args=(
  run --rm
  --name cicy-koubo-api
  --gpus all
  --env-file "$ENV_FILE"
  -p 127.0.0.1:8770:8770
  -v "$STATE_DIR:/var/lib/cicy-koubo-api"
)

# A host-side release overrides the API code baked into the image. Models and
# inference environments always remain in the immutable image layers.
if test -L "$CURRENT" && test -d "$CURRENT/gpu_api"; then
  args+=(-v "$CURRENT:/opt/cicy-koubo-api/current:ro")
fi

exec docker "${args[@]}" "$CICY_KOUBO_GPU_IMAGE"
