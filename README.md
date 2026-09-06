# opencl_filter

Vendor-agnostic **OpenCL** re-implementations of video filters originally written
for **NVIDIA CUDA** in [`rigaya/AviSynthCUDAFilters`](https://github.com/rigaya/AviSynthCUDAFilters)
(AviSynthNeo CUDA filters). The goal is to run the same algorithms on AMD /
Intel / Apple / CPU OpenCL devices — no NVIDIA GPU required.

The OpenCL porting style follows the proven OpenCL filters already shipped in
[`rigaya/QSVEnc`](https://github.com/rigaya/QSVEnc) (`QSVPipeline/rgy_filter_*`,
e.g. the `--vpp-kfm`/`--vpp-degrain` OpenCL pipelines): one `.cl` kernel set per
filter stage plus a small host wrapper that compiles the kernels, sets up
buffers and dispatches them, with a scalar CPU reference used for validation.

## Scope & status (honest picture)

`AviSynthCUDAFilters` is a **suite**, not one filter:

| Project | What it is | Port status here |
|---|---|---|
| **KTGMC** | QTGMC-style deinterlacer (motion compensated) | In progress — **Milestone 1 done** (see below) |
| KNNEDI3 | NNEDI3 neural-net upscaler | not started (next candidate) |
| KFM | filter family (KDeband, Deblock, CombingAnalyze, …) | not started |
| AvsCUDA / GRunT / masktools | CUDA-aware dispatch + helpers | out of scope unless requested |

A faithful KTGMC port is large: `KTGMC/Kernel.cu` (~3,560 lines) + `MVKernel.cu`
(~3,540) + `MV.cpp` (~5,550) implement the whole QTGMC algorithm including
multi-level motion estimation, block search and motion compensation. This repo
therefore progresses in **milestones**; each milestone's kernels are translated
faithfully and validated bit-for-bit against an independent reference.

### Milestone 1 (current)

A validated core of the *per-plane, no-motion* KTGMC kernels (the parts that need
no motion vectors), plus the host-side `ResamplingProgram`:

- `kt_resample_v` — vertical resampler with host-built Mitchell/other FIR program
  (the "Bob" field-interpolation engine).
- `kt_makediff` / add-diff (masktools `mt_makediff`/`mt_adddiff` equivalent).
- `kt_rg_box3x3` — RemoveGrain modes 11/12 ([1 2 1]) and 20 (3×3 mean).
- `kt_removegrain_clip` — RemoveGrain modes 1–4 (8-neighbour sort network).
- `kt_repair_clip` — Repair modes 1–4 (9-neighbour sort network).
- `kt_vertical_cleaner_median` — VerticalCleaner mode 1 (vertical median).
- `kt_to_full_range` — limited→full-range luma/chroma conversion.
- `kt_merge` — weighted merge (32767 fixed point, as `KMerge`).

Each is a faithful scalar OpenCL port (see the per-file notes about why scalar is
mathematically identical). See [`docs/PORT_PLAN.md`](docs/PORT_PLAN.md) for the
full kernel-by-kernel roadmap and how to extend the set.

## Layout

```
docs/PORT_PLAN.md              # port strategy + per-kernel roadmap + licensing
src/opencl/ktgmc/kernels/      # OpenCL kernel sources (.cl)
sim/ktgmc_cpu_ref.cpp          # scalar CPU mirror of the kernels (validates logic)
python/run_validation.py       # independent Python golden + cross-check harness
Makefile                       # make test  (no OpenCL required)
```

## How the port is validated (no GPU/OpenCL needed)

This sandbox has no NVIDIA GPU, no CUDA toolkit, and no AviSynthNeo — and the
Debian package mirrors are unreachable, so even a CPU OpenCL ICD (pocl) could not
be installed. To still make the port *provably* correct, each algorithm exists in
**two independent implementations** that must agree bit-for-bit:

1. `python/run_validation.py` — a Python golden reference written directly from
   the CUDA math.
2. `sim/ktgmc_cpu_ref.cpp` — the scalar CPU mirror that the OpenCL kernels are
   transliterated from.

The OpenCL `.cl` bodies mirror the CPU mirror expression-for-expression
(float accumulation only in the resampler, matching the original CUDA `float`
path). On any machine with an OpenCL ICD you can additionally compile/run the
`.cl` and diff against the CPU mirror.

```sh
make test        # builds the CPU ref and runs the 8-bit + 16-bit cross-checks
```

Expected output:

```
[bitdepth 8] PASS
[bitdepth 16] PASS
```

## Running on a real rig (future milestones)

- Port the motion-search / compensation stages (`MVKernel.cu`, `MV.cpp`) so the
  full KTGMC deinterlacer can be assembled.
- Add AviSynthNeo plugin glue (AviSynth host adaptation is separate from the
  kernels; see `docs/PORT_PLAN.md` §"AviSynth integration").
- Optional: reintroduce 4-channel vectorization as a pure performance pass.

## Licensing

The ported kernels are derived from `rigaya/AviSynthCUDAFilters`, which is a fork
of `nekopanda/AviSynthCUDAFilters`. Per that repository's README the KTGMC code
is **GPL** (KFM is MIT). Derived code should therefore be distributed under the
same terms as the upstream KTGMC sources. Confirm against the exact upstream
`LICENSE` before any release; this repo currently carries no license by default.
