#!/usr/bin/env bash
# Clang otherwise selects the incomplete GCC 12 runtime installation on this host.
# Select headers/runtime from the installed distro C++ compiler, without changing it.
set -euo pipefail
compiler="$1"
shift
gcc_install_dir="$(dirname "$(/usr/bin/g++ -print-libgcc-file-name)")"
exec "$compiler" "--gcc-install-dir=$gcc_install_dir" "$@"
