#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="${1:-cicy-koubo-gpu:latest}"

if docker buildx version >/dev/null 2>&1; then
  exec docker buildx build --load --progress=plain -f "$ROOT/docker/gpu/Dockerfile" -t "$TAG" "$ROOT"
fi
exec docker build -f "$ROOT/docker/gpu/Dockerfile" -t "$TAG" "$ROOT"
