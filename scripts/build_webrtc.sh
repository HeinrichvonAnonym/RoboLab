#!/usr/bin/env bash
# Offline build only. setup_webrtc.sh owns every network/bootstrap operation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/webrtc_versions.sh"
SOURCE="$ROOT/third_party/webrtc/src"
OUTPUT="${ROBOLAB_WEBRTC_BUILD_DIR:-$ROOT/build/third_party/webrtc}"
OUTPUT="$(realpath -m "$OUTPUT")"
for entry in "third_party/depot_tools:$DEPOT_TOOLS_REVISION" "third_party/webrtc/src:$WEBRTC_REVISION"; do
  path="${entry%:*}"; rev="${entry##*:}"
  [[ "$(git -C "$ROOT/$path" rev-parse HEAD 2>/dev/null)" == "$rev" ]] || { echo "Pinned $path missing. Run: bash scripts/setup_webrtc.sh" >&2; exit 1; }
done
[[ -f "$ROOT/third_party/webrtc/.prepared" ]] || { echo "Run: bash scripts/setup_webrtc.sh" >&2; exit 1; }
webrtc_compat_is_applied || { echo "WebRTC compatibility patch missing or locally changed. Run: bash scripts/setup_webrtc.sh" >&2; exit 1; }
GN="$SOURCE/buildtools/linux64/gn"
NINJA="$SOURCE/third_party/ninja/ninja"
CLANG="$SOURCE/third_party/llvm-build/Release+Asserts/bin/clang++"
for tool in "$GN" "$NINJA" "$CLANG"; do
  [[ -x "$tool" ]] || { echo "Missing $tool. Run: bash scripts/setup_webrtc.sh" >&2; exit 1; }
done
export DEPOT_TOOLS_UPDATE=0 DEPOT_TOOLS_METRICS=0
# Do not call depot_tools wrappers: they may bootstrap or download a tool.
export PATH="/usr/bin:/bin:$PATH"
mkdir -p "$OUTPUT"
cp "$ROOT/config/webrtc/args.gn" "$OUTPUT/args.gn"
python3 - "$ROOT" "$OUTPUT" <<'PY'
import json,sys,shlex
from pathlib import Path
root,out=map(Path,sys.argv[1:])
original=root/'third_party/webrtc/src/third_party/llvm-build/Release+Asserts'
overlay=out/'toolchain'
(overlay/'bin').mkdir(parents=True,exist_ok=True)
# Keep the pinned toolchain untouched. Apply the same system-GCC selection to
# C/C++ compilation and to host tools linked by rustc/GN.
for entry in original.iterdir():
    if entry.name!='bin':
        dest=overlay/entry.name
        if not dest.exists():dest.symlink_to(entry,target_is_directory=entry.is_dir())
for entry in (original/'bin').iterdir():
    dest=overlay/'bin'/entry.name
    if entry.name in ('clang','clang++'):
        if dest.is_symlink():dest.unlink()
        dest.write_text('#!/usr/bin/env bash\nexec '+shlex.quote(str(root/'scripts/webrtc_clang_wrapper.sh'))+' '+shlex.quote(str(entry))+' "$@"\n')
        dest.chmod(0o755)
    elif not dest.exists():dest.symlink_to(entry)
with (out/'args.gn').open('a') as f:
    f.write('clang_base_path = '+json.dumps(str(overlay))+'\n')
PY
cd "$SOURCE"
"$GN" gen "$OUTPUT" --fail-on-unused-args
"$NINJA" -C "$OUTPUT" -j "${WEBRTC_BUILD_JOBS:-6}" webrtc
python3 - "$ROOT" "$OUTPUT" "$WEBRTC_REVISION" "$DEPOT_TOOLS_REVISION" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:3])
manifest={'webrtc_revision':sys.argv[3],'depot_tools_revision':sys.argv[4],
          'args_sha256':hashlib.sha256((out/'args.gn').read_bytes()).hexdigest(),
          'profile_sha256':hashlib.sha256((root/'config/webrtc/args.gn').read_bytes()).hexdigest(),
          'compat_patch_sha256':hashlib.sha256((root/'config/webrtc/libstdcxx11.patch').read_bytes()).hexdigest(),
          'clang_version':__import__('subprocess').check_output([str(out/'toolchain/bin/clang++'),'--version'],text=True).splitlines()[0],
          'resolved_dependencies':str(root/'third_party/webrtc/dependencies.resolved.txt'),
          'stdlib':'system libstdc++','library':str(out/'obj/libwebrtc.a')}
(out/'robolab-build.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
echo "Built $OUTPUT/obj/libwebrtc.a"
