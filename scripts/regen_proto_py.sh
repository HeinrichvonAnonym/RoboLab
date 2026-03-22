#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${ROOT}/scripts/gen"
shopt -s nullglob
mapfile -t protos < <(find "${ROOT}/proto" -name '*.proto' -type f | sort)
if ((${#protos[@]} == 0)); then
  echo "No .proto files under ${ROOT}/proto" >&2
  exit 1
fi
protoc -I"${ROOT}/proto" --python_out="${ROOT}/scripts/gen" "${protos[@]}"
echo "Generated Python under ${ROOT}/scripts/gen for ${#protos[@]} file(s)"
