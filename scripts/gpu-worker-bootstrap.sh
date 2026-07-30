#!/usr/bin/env bash
set -euo pipefail

: "${CICY_KOUBO_API:?CICY_KOUBO_API is required}"
: "${CICY_KOUBO_JOB_ID:?CICY_KOUBO_JOB_ID is required}"
: "${CICY_KOUBO_WORKER_TOKEN:?CICY_KOUBO_WORKER_TOKEN is required}"
: "${CICY_KOUBO_ACCESS_TOKEN:?CICY_KOUBO_ACCESS_TOKEN is required}"

case "$CICY_KOUBO_WORKER_TOKEN" in
  wk_[0-9a-f]*) ;;
  *) echo "invalid CICY_KOUBO_WORKER_TOKEN" >&2; exit 2 ;;
esac
case "$CICY_KOUBO_ACCESS_TOKEN" in
  ga_[0-9a-f]*) ;;
  *) echo "invalid CICY_KOUBO_ACCESS_TOKEN" >&2; exit 2 ;;
esac

GPU_IMAGE="${CICY_KOUBO_GPU_IMAGE:-cicy-koubo-gpu:2026.07.29-api}"
API_RELEASE_DIR=/opt/cicy-koubo-api/current
STATE_DIR=/var/lib/cicy-koubo
mkdir -p "$STATE_DIR"
chmod 700 "$STATE_DIR"
mkdir -p "$API_RELEASE_DIR/gpu_api"
curl -fsS --retry 3 \
  "${CICY_KOUBO_API%/}/api/koubo/gpu-api.py" \
  -o "$API_RELEASE_DIR/gpu_api/server.py"
curl -fsS --retry 3 \
  "${CICY_KOUBO_API%/}/api/koubo/cosyvoice-tts.py" \
  -o "$API_RELEASE_DIR/cosyvoice_tts.py"
printf '%s\n' '"""CiCy Koubo GPU API package."""' \
  > "$API_RELEASE_DIR/gpu_api/__init__.py"
python3 -m py_compile \
  "$API_RELEASE_DIR/gpu_api/server.py" \
  "$API_RELEASE_DIR/cosyvoice_tts.py"
grep -q 'zh-yue' "$API_RELEASE_DIR/cosyvoice_tts.py"
grep -q 'CAPABILITIES_VERSION' "$API_RELEASE_DIR/gpu_api/server.py"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl nginx python3
rm -rf /var/lib/apt/lists/*

nvidia-smi >/dev/null
docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'

docker rm -f cicy-koubo-worker >/dev/null 2>&1 || true
docker_args=(
  run -d
  --name cicy-koubo-worker
  --restart unless-stopped
  --gpus all
  --entrypoint python3
  -e "CICY_KOUBO_ACCESS_TOKEN=${CICY_KOUBO_ACCESS_TOKEN}"
  -e KOUBO_API_HOST=0.0.0.0
  -e KOUBO_API_PORT=8770
  -e KOUBO_API_STATE_DIR=/var/lib/cicy-koubo-api
  -p 127.0.0.1:8770:8770
  -v "$STATE_DIR:/var/lib/cicy-koubo-api"
  -v "$API_RELEASE_DIR/cosyvoice_tts.py:/content/cosy/cosyvoice_tts.py:ro"
)
if [ -d "$API_RELEASE_DIR/gpu_api" ]; then
  docker_args+=(
    -e PYTHONPATH=/opt/cicy-koubo-api/current
    -v "$API_RELEASE_DIR:/opt/cicy-koubo-api/current:ro"
  )
fi
docker "${docker_args[@]}" "$GPU_IMAGE" \
  -m gunicorn.app.wsgiapp \
  --bind 0.0.0.0:8770 \
  --workers 1 \
  --threads 8 \
  --timeout 0 \
  --access-logfile - \
  gpu_api.server:app

for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:8770/live >/dev/null; then break; fi
  sleep 2
done
curl -fsS \
  -H "Authorization: Bearer ${CICY_KOUBO_ACCESS_TOKEN}" \
  http://127.0.0.1:8770/v1/health \
  | python3 -c '
import json, sys
data = json.load(sys.stdin)
required = {"zh-CN", "zh-yue", "en", "fr", "de", "es", "it", "vi", "id",
            "ms", "th", "ko", "ru", "ar", "km", "lo"}
assert data.get("ok") is True
assert str(data.get("capabilities_version", "")).startswith("2026.07.30-multilingual-")
assert required.issubset(set(data.get("tts_languages") or []))
assert len(str(data.get("tts_script_sha256") or "")) == 64
'

cat > /etc/nginx/sites-available/cicy-koubo-worker <<EOF
server {
    listen 0.0.0.0:8780;
    client_max_body_size 5g;
    proxy_request_buffering off;
    proxy_buffering off;
    proxy_read_timeout 7200s;

    if (\$http_authorization != "Bearer ${CICY_KOUBO_ACCESS_TOKEN}") {
        return 401;
    }

    location / {
        proxy_set_header Host \$host;
        proxy_set_header X-Forwarded-Proto https;
        proxy_pass http://127.0.0.1:8770;
    }
}
EOF
ln -sf /etc/nginx/sites-available/cicy-koubo-worker /etc/nginx/sites-enabled/cicy-koubo-worker
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx

INSTANCE_ID="$(curl -fsS --max-time 3 \
  http://100.100.100.200/latest/meta-data/instance-id)"
PUBLIC_IPV4="$(curl -fsS --max-time 3 \
  http://100.100.100.200/latest/meta-data/eipv4 2>/dev/null || \
  curl -fsS --max-time 3 http://100.100.100.200/latest/meta-data/public-ipv4)"
case "$PUBLIC_IPV4" in
  *[!0-9.]*|"") echo "invalid public IPv4" >&2; exit 3 ;;
esac
ENDPOINT="http://${PUBLIC_IPV4}:8780"
curl -fsS -X POST "${CICY_KOUBO_API%/}/api/koubo/worker/register" \
  -H "Authorization: Bearer ${CICY_KOUBO_WORKER_TOKEN}" \
  -H "Content-Type: application/json" \
  --data "$(python3 - "$INSTANCE_ID" "$ENDPOINT" <<'PY'
import json, sys
print(json.dumps({"instance_id": sys.argv[1], "endpoint": sys.argv[2]}))
PY
)"

cat > /etc/cicy-koubo-worker.env <<EOF
CICY_KOUBO_API=${CICY_KOUBO_API}
CICY_KOUBO_JOB_ID=${CICY_KOUBO_JOB_ID}
CICY_KOUBO_WORKER_TOKEN=${CICY_KOUBO_WORKER_TOKEN}
EOF
chmod 0600 /etc/cicy-koubo-worker.env
cat > /usr/local/bin/cicy-koubo-heartbeat <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source /etc/cicy-koubo-worker.env
while true; do
  curl -fsS -X POST "${CICY_KOUBO_API%/}/api/koubo/worker/heartbeat" \
    -H "Authorization: Bearer ${CICY_KOUBO_WORKER_TOKEN}" \
    -H "Content-Type: application/json" \
    --data '{"stage":"worker_ready"}' >/dev/null || true
  sleep 60
done
EOF
chmod 0755 /usr/local/bin/cicy-koubo-heartbeat
cat > /etc/systemd/system/cicy-koubo-heartbeat.service <<'EOF'
[Unit]
Description=CiCy Koubo worker heartbeat
After=network-online.target nginx.service
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/cicy-koubo-heartbeat
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now cicy-koubo-heartbeat

echo "CiCy Koubo worker ready: ${ENDPOINT}"
