#!/usr/bin/env bash
set -euo pipefail

export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "${1:-status}" in
  status)
    nvidia-smi
    test -f /content/mt/READY
    test -f /content/cosy/COSY_READY
    echo "cicy-koubo GPU image ready"
    ;;
  musetalk)
    shift
    exec bash /content/mt/synthesize.sh "$@"
    ;;
  cosy)
    shift
    exec /content/cosy/env/bin/python /content/cosy/cosyvoice_tts.py "$@"
    ;;
  serve)
    shift
    python_tag="$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
    app="/opt/cicy-koubo/package/app/app-${python_tag}.pyc"
    test -f "$app"
    exec python3 "$app" "$@"
    ;;
  shell)
    exec bash
    ;;
  *)
    exec "$@"
    ;;
esac
