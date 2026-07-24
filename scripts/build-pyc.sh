#!/bin/bash
# 把 src/app.py 编成多版本 pyc 放进 app/(npm 只发 pyc,不发源码)
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f app/app-*.pyc
build() { # $1=python可执行 $2=版本tag
  "$1" -c "import py_compile;py_compile.compile('src/app.py','app/app-$2.pyc',dfile='app.py',doraise=True)" \
    && echo "  built app-$2.pyc ($($1 --version 2>&1))"
}
build python3.10 310 || echo "  skip 310"
UV11=$(uv python find 3.11 2>/dev/null || true); [ -n "$UV11" ] && build "$UV11" 311 || echo "  skip 311"
UV12=$(uv python find 3.12 2>/dev/null || true); [ -n "$UV12" ] && build "$UV12" 312 || echo "  skip 312"
build python3.13 313 || echo "  skip 313"
ls -la app/app-*.pyc
