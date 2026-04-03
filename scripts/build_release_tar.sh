#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f VERSION ]]; then
  echo "Error: VERSION file not found"
  exit 1
fi

ver="$(tr -d '[:space:]' < VERSION)"
if [[ -z "$ver" ]]; then
  echo "Error: VERSION is empty"
  exit 1
fi

if [[ "$ver" != v* ]]; then
  ver="v$ver"
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker command not found"
  exit 1
fi

mkdir -p "release/$ver"

echo "[1/2] build minxinagent:$ver"
docker build \
  --platform linux/amd64 \
  -f libs/cli/Dockerfile \
  -t "minxinagent:$ver" \
  .

echo "[2/2] save tar"
docker save -o "release/$ver/minxinagent-$ver-linux-amd64.tar" "minxinagent:$ver"

echo "Done: release/$ver/minxinagent-$ver-linux-amd64.tar"
