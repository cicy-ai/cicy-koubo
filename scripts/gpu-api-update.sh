#!/usr/bin/env bash
set -euo pipefail

: "${CICY_KOUBO_UPDATE_URL:?CICY_KOUBO_UPDATE_URL is required}"
: "${CICY_KOUBO_WORKER_TOKEN:?CICY_KOUBO_WORKER_TOKEN is required}"

ROOT=/opt/cicy-koubo-api
RELEASES="$ROOT/releases"
CURRENT="$ROOT/current"
PUBLIC_KEY=/etc/cicy-koubo/update-signing-key.pem
mkdir -p "$RELEASES"

case "$CICY_KOUBO_UPDATE_URL" in
  https://*) ;;
  *) echo "update URL must use HTTPS" >&2; exit 2 ;;
esac

tmp="$(mktemp -d /opt/cicy-koubo-update.XXXXXX)"
trap 'rm -rf "$tmp"' EXIT

curl -fsS --retry 3 \
  -H "Authorization: Bearer ${CICY_KOUBO_WORKER_TOKEN}" \
  "$CICY_KOUBO_UPDATE_URL/manifest.json" -o "$tmp/manifest.json"

read -r version sha256 archive signature <<EOF
$(python3 - "$tmp/manifest.json" <<'PY'
import json, re, sys
m = json.load(open(sys.argv[1]))
version = str(m.get("version", ""))
sha = str(m.get("sha256", ""))
archive = str(m.get("archive", ""))
signature = str(m.get("signature", ""))
if not re.fullmatch(r"[0-9A-Za-z._-]{1,64}", version):
    raise SystemExit("invalid version")
if not re.fullmatch(r"[0-9a-f]{64}", sha):
    raise SystemExit("invalid sha256")
if not re.fullmatch(r"[0-9A-Za-z._-]{1,128}", archive):
    raise SystemExit("invalid archive")
if not re.fullmatch(r"[0-9A-Za-z._-]{1,128}", signature):
    raise SystemExit("invalid signature")
print(version, sha, archive, signature)
PY
)
EOF

curl -fsS --retry 3 \
  -H "Authorization: Bearer ${CICY_KOUBO_WORKER_TOKEN}" \
  "$CICY_KOUBO_UPDATE_URL/$archive" -o "$tmp/release.tar.gz"
curl -fsS --retry 3 \
  -H "Authorization: Bearer ${CICY_KOUBO_WORKER_TOKEN}" \
  "$CICY_KOUBO_UPDATE_URL/$signature" -o "$tmp/release.sig"

printf '%s  %s\n' "$sha256" "$tmp/release.tar.gz" | sha256sum -c -
test -s "$PUBLIC_KEY"
openssl dgst -sha256 -verify "$PUBLIC_KEY" \
  -signature "$tmp/release.sig" "$tmp/release.tar.gz"

target="$RELEASES/$version"
test ! -e "$target"
mkdir "$target"
tar -xzf "$tmp/release.tar.gz" --strip-components=1 -C "$target"
python3 -m py_compile "$target/gpu_api/"*.py

previous="$(readlink -f "$CURRENT" 2>/dev/null || true)"
ln -sfn "$target" "$ROOT/.current.new"
mv -Tf "$ROOT/.current.new" "$CURRENT"

if ! systemctl restart cicy-koubo-api \
  || ! curl -fsS --retry 10 --retry-delay 1 http://127.0.0.1:8770/live >/dev/null; then
  if test -n "$previous"; then
    ln -sfn "$previous" "$ROOT/.current.new"
    mv -Tf "$ROOT/.current.new" "$CURRENT"
    systemctl restart cicy-koubo-api
  fi
  echo "update failed; rolled back" >&2
  exit 1
fi

echo "updated cicy-koubo-api to $version"
