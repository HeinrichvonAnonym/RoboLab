#!/usr/bin/env bash
# Explicit network/bootstrap step. Does not upgrade distro packages or other submodules.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/webrtc_versions.sh"
[[ "$(uname -sm)" == "Linux x86_64" ]] || { echo "WebRTC profile requires Linux x86_64" >&2; exit 1; }
git -C "$ROOT" submodule update --init -- third_party/depot_tools third_party/webrtc/src
for entry in "third_party/depot_tools:$DEPOT_TOOLS_REVISION" "third_party/webrtc/src:$WEBRTC_REVISION"; do
  path="${entry%:*}"; rev="${entry##*:}"
  [[ "$(git -C "$ROOT/$path" rev-parse HEAD)" == "$rev" ]] || { echo "Unexpected revision: $path" >&2; exit 1; }
  git -C "$ROOT/$path" diff --cached --quiet || { echo "Preserving staged submodule changes: $path" >&2; exit 1; }
  if ! git -C "$ROOT/$path" diff --quiet || ! git -C "$ROOT/$path" diff --cached --quiet; then
    [[ "$path" == third_party/webrtc/src ]] && webrtc_compat_is_applied || { echo "Preserving modified submodule: $path" >&2; exit 1; }
  fi
done
export DEPOT_TOOLS_UPDATE=0 DEPOT_TOOLS_METRICS=0
export PATH="$ROOT/third_party/depot_tools:$PATH"
export CIPD_CACHE_DIR="$ROOT/third_party/webrtc/.cache/cipd"
export VPYTHON_VIRTUALENV_ROOT="$ROOT/third_party/webrtc/.cache/vpython"
export XDG_CACHE_HOME="$ROOT/third_party/webrtc/.cache"
mkdir -p "$XDG_CACHE_HOME"
cd "$ROOT/third_party/webrtc"
cat > .gclient <<'EOF'
solutions = [{
  'name': 'src',
  'url': 'https://webrtc.googlesource.com/src.git',
  'managed': False,
  'custom_deps': {},
  # At this pinned revision the sole unnamed hook downloads upstream test media.
  # No action suppresses that hook; do not download tests we do not build.
  'custom_hooks': [{'name': ''}, {'name': 'Generate component metadata for tests'},
                   {'name': 'sysroot_x86'}, {'name': 'sysroot_x64'}],
  'custom_vars': {
    'checkout_android': False, 'checkout_ios': False,
    'checkout_configuration': 'minimal',
    'checkout_instrumented_libraries': False,
  },
}]
target_os = ['linux']
target_os_only = True
EOF
# gclient requires a clean root checkout even when the revision is unchanged.
# Remove only our exact verified patch during sync and restore it on failure too.
restore_compat() {
  if ! webrtc_compat_is_applied; then
    git -C src diff --quiet && git -C src diff --cached --quiet || {
      echo "Preserving unexpected source changes; compatibility patch was not restored" >&2; return 1;
    }
    git -C src apply --check "$ROOT/config/webrtc/libstdcxx11.patch" || return 1
    git -C src apply "$ROOT/config/webrtc/libstdcxx11.patch"
  fi
}
if webrtc_compat_is_applied; then
  git -C src apply --reverse "$ROOT/config/webrtc/libstdcxx11.patch"
fi
trap restore_compat EXIT
rm -f .prepared
gclient sync --no-history --revision "src@$WEBRTC_REVISION" --jobs "${WEBRTC_SYNC_JOBS:-4}"
gclient revinfo --actual > dependencies.resolved.txt
restore_compat
trap - EXIT
printf '%s\n' "$WEBRTC_REVISION" "$DEPOT_TOOLS_REVISION" > .prepared
echo "WebRTC dependencies prepared. Next: bash scripts/build_webrtc.sh"
