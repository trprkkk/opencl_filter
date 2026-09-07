# KFM — CUDA → OpenCL port status & map

This grounds the port of the **KFM** filter family
(`rigaya/AviSynthCUDAFilters/KFM`, MIT licensed) to OpenCL, following the same
conventions as the KTGMC port (`src/opencl/ktgmc/...`). KFM kernels live under
`src/opencl/kfm/kernels/`. Status legend matches the KTGMC files:
`// ALG-VERIFIED` = integer algorithm cross-checked (make test) against an
independent CPU mirror + Python golden; `// RIG-VERIFY` = faithful source port,
device run pending.

Why KFM: it is **MIT** (cleanest to reuse/redistribute) and mostly per-plane /
per-pixel / self-contained filters, several of which ship an exact CPU reference
in the same `.cu` file — so they are verifiable in this sandbox the same way the
KTGMC kernels were. KDeband was chosen first (see docs/PORT_PLAN.md §5).

## Registered filters (by source file)

| File | AVS functions | Port status here |
|---|---|---|
| `KDeband.cu` | `KTemporalNR`, `KDeband`, `KEdgeLevel` | **KDeband core done**; KTemporalNR & KEdgeLevel next |
| `KFMKernel.cu` | `KPatchCombe`, `KFMSwitch`, `KFMPad`, `KFMDecimate`, `AssumeDevice` | not started |
| `CombingAnalyze.cu` | `KFMSuper`, `KCleanSuper`, `KPreCycleAnalyze(_Show)`, `KFMSuperShow`, `KTelecine(_Super)`, `KSwitchFlag`, `KContainsCombe`, `KCombeMask`, `KRemoveCombe` | not started |
| `Deblock.cu` | `KDeblock`, `QPClip`, `ShowQP`, `FrameType` | not started |
| `DecombeUCF.cu` | `KCFieldDiff`, `KCFrameDiffDup`, `KNoiseClip`, `KAnalyzeNoise`, `KDecombUCF*` | not started |
| `MergeStatic.cu` | `KTemporalDiff`, `KAnalyzeStatic`, `KMergeStatic` | not started |

## Verified

### `KDeband` (`kf_deband_reduce_banding`, src/opencl/kfm/kernels/kfm_deband.cl)

The debanding core, faithful to the authoritative CPU twin `cpu_reduce_banding`
(= CUDA `kl_reduce_banding`) in `KDeband.cu`. Per-pixel, integer-exact: for each
pixel it draws a deterministic pseudo-random sampling offset (clamped to the
image so no border padding is needed), averages in-range neighbours per
`sample_mode` (0/1/2) and `blur_first`, and replaces the pixel only when the
difference is within `thresh`. `// ALG-VERIFIED`:
`python/run_kfm_deband.py` (300 cases) cross-checks a verbatim CPU mirror
`sim/kfm_deband_ref.cpp` against an independent Python golden over sample modes
0-2, blur_first on/off, 8/16-bit, range/thresh sweeps. The `rand` pseudo-random
byte stream is the host-side CUDA `XorShift(seed 0)` (the `CreateDebandRandom`
generator), reproduced in the Python golden and passed as input to the mirror;
`thresh` is host-scaled via `scaleParam` (`thresh*(1<<(bits-8))+0.5`).

Note (faithful config): upstream builds ONE `rand` buffer of `width*height*2`
from the luma-plane dimensions and reuses it per plane with each plane's own
`width*height` stride; the `offset = y*pitch+x` rand indexing stays in bounds
when per-plane `pitch == per-plane width`, which is the assumed/host config.

## Next candidates (verifiable in this sandbox)

- `KEdgeLevel` (`kl_edgelevel`/`_repair`, `kl_el_to444`/`from444`) — per-pixel
  edge map; has `cpu_edgelevel` CPU twins. Larger (edgelevel + repair + 4:4:4
  pack/unpack), but deterministic integer.
- `KTemporalNR` (`kl_temporal_nr` + `cpu_temporal_nr`) — temporal averaging over
  `nframes` with `mid`; deterministic, but its `average_pixel` uses float32
  `(int)(sum/cnt + 0.5f)` so a faithful/bit-exact cross-check needs the mirror
  AND golden to use float32 division (not double) — model with numpy float32.
- `kl_copy` / `kl_fill` helpers (already covered generically by KTGMC `kt_copy`).

## Notes / licence

KFM is MIT; distribute derived files under MIT terms (differs from KTGMC's GPL).
Grounding sources: this sandbox re-clones `AviSynthCUDAFilters` to
`/tmp/avs_cuda/KFM`.
