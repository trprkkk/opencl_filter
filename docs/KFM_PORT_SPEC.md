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
| `KDeband.cu` | `KTemporalNR`, `KDeband`, `KEdgeLevel` | **KDeband core done**; **KEdgeLevel done**; **KTemporalNR done** |
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

### `KEdgeLevel` (kf_edgelevel / kf_edgelevel_repair / kf_el_to444 /
kf_el_from444, src/opencl/kfm/kernels/kfm_edgelevel.cl)

The edge-enhancement/visualisation filter. Four kernels are ported together
(they are the compose/decompose helpers KEdgeLevel drives on its uv path). All
are `// ALG-VERIFIED` via `python/run_kfm_edgelevel.py` (500 cases), which
cross-checks the CPU mirror `sim/kfm_edgelevel_ref.cpp` (faithful to the
authoritative CPU twins `cpu_edgelevel`, `cpu_edgelevel_repair`,
`cpu_el_from444`; for `el_to444` it replicates the CUDA `kl_el_to444`, which
differs from `cpu_el_to444` only via a border-clamp) against an independent
Python golden.

- `kf_edgelevel` — border ring (`<=1+selective` px in) copies through (check ?
  `SCALE(EDGE_CHECK_NONE=16)` : src); interior scans a horizontal and vertical
  window of radius `2+selective` around each pixel, keeps the axis with the
  larger min/max spread, and in selective mode records the max consecutive
  gradient `hdiffmax`; `rdiff = hdiffmax/(float)(hmax-hmin)`,
  `factor = clamp((0.55f-rdiff)*10,0,1) - clamp((0.35f-rdiff)*10,0,1)`
  (selective) else `1.0f`. When `spread > thrs && factor>0`:
  - check (visualise): `src>avg ? (factor==1?WHITE(50):BRIGHT(120))
    : (factor==1?BLACK(240):DARK(180))`, else `SCALE(NONE=16)`.
  - enhance: `factorY=(str*factor)*0.0625f`;
    `dst = clamp(src + (int)((src-avg)*factorY), hmin, hmax)` then
    `clamp(0,maxv)`; uv planes use `factorUV=strUV*0.0625f` with the plane-local
    min/max of `dev_el_min_max` (same 5/7-sample window).
  All float work is IEEE float32 with no FMA contraction; the mirror is built
  `-ffp-contract=off` and the Python golden emulates float32 per operation, so
  results are bit-exact.
- `kf_edgelevel_repair` — fixed N (`3` upstream). When `el != src`, collects the
  eight 3×3 neighbours, sorts them (Batcher odd-even 8-element network,
  `IntCompareAndSwap`), and `dst = clamp(el, min(src,a[N-1]),
  max(src,a[8-N]))`; else copies src. The CUDA/CPU originals read the full 3×3
  window with no border guard (borders rely on the padded AviSynth plane), so
  this kernel is ALG-VERIFIED over the interior `(1..width-2, 1..height-2)`;
  the border ring needs a padded source on the rig (`// RIG-VERIFY` there).
- `kf_el_to444` — bilinear chroma up-sampler (`BW=1<<logUVx`, `BH=1<<logUVy`):
  `(v00+v10+1)>>1`, `(v00+v01+1)>>1`, `(v00+v10+v01+v11+2)>>2`, with the last
  row/col edge-replicated (the CUDA guards `x+1<width` / `y+1<height` fall back
  to `v00`). Grid is source `(width,height)` dims; dst is `BW*width × BH*height`.
- `kf_el_from444` — chroma down-sample: `dst[x+y*dstPitch] =
  src[BW*x + BH*y*srcPitch]`.

Host seam (multi-pass / 4:2:x): KEdgeLevel's real `GetFrameT` runs up to
`numel=max((repair+1)/2,1)` strengthening passes (dst/tmp swap), then `repair`
passes of `edgelevel_repair<3>`, selects per-iteration behaviour by
`table_idx = ((show&&last)?4:0)+(repair>0?2:0)+(uv?1:0)` over the 8 template
tuples (check × selective × uv), and up/down-converts U/V to/from 4:4:4 via
`el_to444`/`el_from444` so edgelevel can sharpen chroma; the subsampled-plane
sizes passed to those helpers are host-determined (a `logUVx/y` seam). Those
multi-pass/host loops are recorded here as notes only; each `.cl` kernel is a
single-plane / single-iteration transliteration verified on a self-consistent
(source-dims) configuration.

### `KTemporalNR` (kf_temporal_nr, src/opencl/kfm/kernels/kfm_temporalnr.cl)

Temporal noise reducer, faithful to the authoritative CPU twin `cpu_temporal_nr`
(= CUDA `kl_temporal_nr` arithmetic) in `KDeband.cu`. Column-wise / per-pixel:
for the pixel at (x,y), centre = `frames[mid]`; it averages the pixels at the
same location across the `nframes = 2*dist+1` frames whose value is within
`thresh` of the centre (`count += 1; sum += ref` when `absdiff(ref,centre)
<= thresh`), then `avg = (float)sum/count + 0.5f` and truncates
(`// ALG-VERIFIED` via `python/run_kfm_temporalnr.py`, 300 cases: CPU mirror
`sim/kfm_temporalnr_ref.cpp` vs an independent float32-exact Python golden,
dist 0-6 / 8-16-bit / thresh sweeps). `count` is always ≥1 (the centre frame
gives diff 0 ≤ thresh), so no division-by-zero and no explicit clamp is needed
(matches the CPU twin). Because each output element reads only its own column,
the CUDA 1- vs 4-wide vectorisation is irrelevant — a scalar port is identical.

Two fidelity notes (host / device seams):
- **Layout**: the CUDA host launches once *per plane* passing a
  `TemporalNRPtrs` holding `nframes` separate input pointers. OpenCL takes the
  planes packed into one buffer (`plane i` at `i*frame_stride`); per-pixel
  arithmetic is identical, only the pointer-array vs packed-frame host layout
  differs.
- **Division**: this OpenCL transliteration (like `cpu_temporal_nr`) uses normal
  float32 `/`. The real CUDA *device* kernel uses the approximate `__fdividef`
  (≤ ~2 ulp error), a CUDA-specific hardware-division intrinsic with no OpenCL
  equivalent; it can differ from correctly-rounded division only at a rounding
  boundary. OpenCL `/` mirrors the CPU reference (which is what ALG-VERIFIED
  checks against); a rig comparing against literal CUDA device output would need
  `-cl-fp32-correctly-rounded-enabled` (best effort) and is a `// RIG-VERIFY`
  item (~ulp-level, not algorithmic).

## Next candidates (verifiable in this sandbox)

- `kl_copy` / `kl_fill` helpers (already covered generically by KTGMC `kt_copy`).
- Remaining KFM families are in Deblock.cu / CombingAnalyze.cu / DecombeUCF.cu /
  MergeStatic.cu / KFMKernel.cu (registered in the table above; most need more
  host glue / motion state, so they are deferred as heavier milestones).

## Notes / licence

KFM is MIT; distribute derived files under MIT terms (differs from KTGMC's GPL).
Grounding sources: this sandbox re-clones `AviSynthCUDAFilters` to
`/tmp/avs_cuda/KFM`.
