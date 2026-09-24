# GRunT — vendored verbatim (no port; no CUDA exists upstream)

GRunT (Gavino's Run-Time) is a CPU-only AviSynth script-runtime plugin:
`GRTConfig`, `GScriptClip`, `GFrameEvaluate`, `GConditionalFilter` (×2
overloads), `GWriteFile`, `GWriteFileIf`, plus classic `ScriptClip` /
`FrameEvaluate` / `ConditionalFilter` / `WriteFile` aliases. Upstream ships
no CUDA for it, so there is nothing to port — this directory is a byte-exact
snapshot, wired into CMake as an optional Windows target.

## Provenance

- Upstream: `https://github.com/rigaya/AviSynthCUDAFilters`
- Commit: `68aef6e9b8a63d0dceb8e7f68b52b27431aa361f9`
- Files (sha256 at vendoring):
  - `GRunT.cpp` (`GRunT/GRunT.cpp`):
    `ddcd38cac5a7cb066501c148f26489bb4117bc589d31f17303a0100387d70cf3`
  - `GRunTversion.h` (`GRunT/GRunTversion.h`):
    `79c1dc2de29a0dc6c039540401d484cf614ec3f65e60457247140cf974b6d615`
  - `KVersion_upstream_common.h` (copy of upstream `common/KVersion.h`,
    which `GRunTversion.h` includes):
    `ee9f4fa233f97979a05b8308230272afa774343425e0759fdb236b50336ea288`
- Line endings/encoding preserved as-is (mixed CRLF/LF, ASCII).
- License: GPL-2.0-or-later (see the header of `GRunT.cpp`).

NOT vendored (deliberately): `GRunT.rc`, `GRunT.vcxproj*(.filters/.user)`
— Windows IDE artifacts; the CMake target below replaces the build story.

## Build (Windows + AviSynthPlus SDK)

```bat
cmake -S . -B build
cmake --build build --target GRunT --config Release
```

with the `AVISYNTH_SDK` environment variable pointing at an AviSynthPlus
checkout/install (headers under `%AVISYNTH_SDK%\include`). Produces
`GRunT.dll`. The target is skipped silently on non-Windows or without the
headers (same optional pattern as the OpenCL host targets).

## Verification status

- Byte-identity vs upstream at the commit above (sha256, re-check with
  `sha256sum` against a fresh clone).
- Parse-checked only: `g++ -fsyntax-only -x c++ -std=c++17` against real
  AviSynthPlus `avisynth.h` with a stub `windows.h` (empty + `<ctype.h>`,
  which the real header provides transitively) and MSVC keywords
  (`__declspec`, `__stdcall`) defined away. Never compiled for real or run
  here (no Windows, no AviSynth) — first duty on a Windows rig is the
  CMake build above.
