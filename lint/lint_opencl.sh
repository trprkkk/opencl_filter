#!/usr/bin/env bash
# Structural lint of the OpenCL kernel sources.
#
# gcc treats a ".cl" file as a linker input and will NOT parse it, so we copy
# each kernel to a ".c" temp file and run a genuine C parse (-x c) through the
# emulation shim lint/oc_shim.h. This only catches syntax/typo errors (the shim
# does NOT reproduce OpenCL semantics); real semantic/device validation still
# requires an OpenCL compiler on a rig (see docs/MV_PORT_SPEC.md §7).
#
# Usage:  lint/lint_opencl.sh  [file.cl ...]   (default: all src/opencl/**/*.cl)
set -u
here="$(cd "$(dirname "$0")" && pwd)"
shim="$here/oc_shim.h"
CC="${CC:-gcc}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if [ "$#" -eq 0 ]; then
  files="$(cd "$here/.." && find src/opencl -name '*.cl' | sort)"
else
  files="$*"
fi

fails=0
runs=0
for f in $files; do
  for pxdef in "uchar 255" "ushort 65535"; do
    set -- $pxdef
    c="$tmp/$(basename "$f" .cl)_$1.c"
    cp "$f" "$c"
    if $CC -fsyntax-only -x c -DPX="$1" -DPX_MAX="$2" -include "$shim" "$c" 2>"$tmp/err"; then
      echo "OK    $f (-DPX=$1)"
    else
      echo "FAIL  $f (-DPX=$1)"
      sed -n '1,15p' "$tmp/err"
      fails=$((fails+1))
    fi
    runs=$((runs+1))
  done
done
echo "lint: $runs checks, $fails failures"
[ "$fails" -eq 0 ]
