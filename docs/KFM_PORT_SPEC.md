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
| `KFMKernel.cu` | `KPatchCombe`, `KFMSwitch`, `KFMPad`, `KFMDecimate`, `AssumeDevice` | inventoried: host-only, no device kernels (see below; its only kernel dep `kl_merge` is ported as `kf_merge_block`) |
| `CombingAnalyze.cu` | `KFMSuper`, `KCleanSuper`, `KPreCycleAnalyze(_Show)`, `KFMSuperShow`, `KTelecine(_Super)`, `KSwitchFlag`, `KContainsCombe`, `KCombeMask`, `KRemoveCombe` | not started |
| `Deblock.cu` | `KDeblock`, `QPClip`, `ShowQP`, `FrameType` | **all 11 device kernels transcribed** (`kf_deblock`, `kf_make_qp_table`, `kf_deblock_show`, `kf_max_vh/v/h`, `kf_scale_qp`, `kf_sharpen_coeff`, `kf_merge_deblock` ALG-VERIFIED; `kf_sharpen`, `kf_show_sharpen_coeff` in the separate provisional `kfm_deblock_rig.cl`, `// RIG-VERIFY`, handoff spec in `docs/RIG_HANDOFF_KDEBLOCK.md`; QPClip is a no-op pass-through) |
| `DecombeUCF.cu` | `KCFieldDiff`, `KCFrameDiffDup`, `KNoiseClip`, `KAnalyzeNoise`, `KDecombUCF*` | **device kernels 8/8 done** (KNoiseClip + 7 reductions, see below; only host pipelines remain) |
| `MergeStatic.cu` | `KTemporalDiff`, `KAnalyzeStatic`, `KMergeStatic` | **all 6 pipeline kernels done** (KDeband.cu-style core; KAnalyzeStatic host glue in `kfm_filterbase.cl`/`kfm_mergestatic.cl`) |

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

### `MergeStatic.cu` kernels (kf_compare_frames / kf_min_frames /
kf_and_coefs / kf_merge_static, src/opencl/kfm/kernels/kfm_mergestatic.cl)

One source file registers three AVS filters (`KTemporalDiff`, `KAnalyzeStatic`,
`KMergeStatic`) and defines four device kernels; all four are ported and
`// ALG-VERIFIED` via `python/run_kfm_mergestatic.py` (580 cases) against the
CPU mirror `sim/kfm_mergestatic_ref.cpp` (cpu_compare_frames / cpu_min_frames /
cpu_merge_static are exact twins in MergeStatic.cu; `cpu_and_coefs` is an exact
twin in KFMFilterBase.cu — see the KFMFilterBase section below). The CUDA
kernels process 4-wide (uchar4/ushort4) vectors over `width4=width>>2`; each
channel is independent, so the scalar port is bit-identical when plane width is
a multiple of 4 (a faithful host config; CUDA never writes the `width%4`
trailing columns).

- `kf_compare_frames` — **KTemporalDiff** core: per pixel `dst =
  max(f0..f4)-min(f0..f4)` across the frames `[n-2..n+2]` (temporal spread; high
  ⇒ motion). Integer. The filter is just this per Y/U/V, so it is end-to-end
  ALG-VERIFIED (modulo host glue).
- `kf_min_frames` — KAnalyzeStatic sub-step: `dst = min(f0,f1,f2)` of the 3
  temporal-diff frames. Integer.
- `kf_and_coefs` — KAnalyzeStatic combing∧static sub-step, float32 (no FMA):
  `combe = clamp(val(dstp)*invcombe - 1, -0.5, 0.5)`;
  `diff = clamp(val(diffp)*(-invdiff) + 1, -0.5, 0.5)`;
  `dstp = trunc(max(combe+diff,0)*128 + 0.5)` (result coefficient ∈ [0,128],
  feeds kf_merge_static). `invcombe=1.0f/thcombe`, `invdiff=1.0f/thdiff` are
  host constants.
- `kf_merge_static` — **KMergeStatic** core: per pixel `dst =
  (coef*v30+(128-coef)*v60+64)>>7`, coef ∈ [0,128] from the static flag plane
  (coef=128 ⇒ take the 30fps field, 0 ⇒ keep the 60fps frame); host copies the
  60fps frame into dst first. Integer. The filter is this over Y/U/V, so it is
  end-to-end ALG-VERIFIED (modulo host CopyFrame glue).

Assembly / fidelity notes (honest): **KTemporalDiff** and **KMergeStatic** are
self-contained per-plane filters — only host AviSynth glue (frame fetch,
NewVideoFrame, CopyFrame) is needed to realise them on the rig. **KAnalyzeStatic**
is a pipeline whose full kernel set is now ported+verified: `CompareFields` →
`kf_calc_combe`, `MergeUVCoefs` → `kf_merge_uvcoefs`, `ExtendCoefs` →
`kf_extend_coef2`, the temporal-min → `kf_min_frames`, `AndCoefs` →
`kf_and_coefs`, and `ApplyUVCoefs` → `kf_apply_uvcoefs_420` (the last four of
these live in `kfm_filterbase.cl`, see below). It requires YV12 subsampling
(logUVx=logUVy=1). What remains is the *host* assembly glue — the VPAD-mirror
pad of frame n, the launch geometry over the padded buffer, and the sequencing
of the two `MergeUVCoefs/ExtendCoefs` phases and the 3-frame temporal-diff read —
which is a `// RIG-VERIFY` seam (device-bound), so the assembled KAnalyzeStatic
filter is not run here. `kf_and_coefs` float contraction is a `// RIG-VERIFY`
item (CUDA may fuse `a*b+c` into fma; <1 ulp difference).

### `KFMFilterBase` coefficient kernels (kf_calc_combe / kf_merge_uvcoefs /
kf_extend_coef2 / kf_apply_uvcoefs_420 / kf_padv / kf_padh / kf_merge_block /
kf_average / kf_max / kf_merge_uvflags / kf_copy_border / kf_analyze_frame /
kf_copy_pad / kf_copy_pad_2plane / kf_max_extend_blocks_h /
kf_max_extend_blocks_v / kf_copy / kf_copy_2plane / kf_fill,
src/opencl/kfm/kernels/kfm_filterbase.cl)

`KFMFilterBase.cu` is the shared base class and defines the coefficient kernels
that KAnalyzeStatic is assembled from (`cpu_*` twins exist in the same file).
Nineteen are ported here and `// ALG-VERIFIED` via `python/run_kfm_filterbase.py`
(3100 cases) against the CPU mirror `sim/kfm_filterbase_ref.cpp` and an
independent Python golden. All are per-pixel/per-plane integer ops (no float),
so the CUDA 4-wide vectorisation is equivalent to a scalar port.

- `kf_calc_combe` — `CompareFields` core: combing measure at each pixel from the
  5 vertical taps y-2..y+2, `combe = |a + 4c + e - 3(b+d)| >> 2`, clamped
  `[0,255]` (regardless of bit depth — upstream casts the clamped int). It reads
  rows y-2..y+2 with no border guard, so upstream feeds a vertically
  mirror-padded frame (`VPAD=4`); interior rows 2..height-3 are ALG-VERIFIED,
  the top/bottom rows are RIG-VERIFY on the padded plane.
- `kf_merge_uvcoefs` — `MergeUVCoefs` core: in-place `fY = max(fY, max(fU,fV))`
  with the UV coeff read at subsampled offset `(x>>logUVx, y>>logUVy)`.
- `kf_extend_coef2` — `ExtendCoefs` core = the CUDA `kl_extend_coef2` device
  kernel: `dst = max(src)` over vertical rows y-1..y+1 with y clamped to
  `[0,height-1]`. (Upstream's CPU *fallback* branch instead runs
  `cpu_extend_coef` over the interior + `cpu_copy_border`, which copies the
  extreme rows straight through — differing from the device kernel at row 0 and
  row height-1. OpenCL targets the GPU, so this port follows the device kernel;
  the divergence is upstream's and only touches 2 rows of a band coefficient.)
- `kf_apply_uvcoefs_420` — `ApplyUVCoefs` core (YV12): sets `fU=fV` to the
  rounded 2×2 average of the Y coefficient plane.
- `kf_padv` / `kf_padh` — the shared in-place mirror-pad helpers
  (`cpu_padv`/`kl_padv`, `cpu_padh`/`kl_padh` — exact twins): `dst` points at
  the interior origin of a buffer with `vpad`/`hpad` spare rows/columns;
  `padv` mirrors rows (`-y-1 ← y`, `height+y ← height-y-1`), `padh` mirrors
  columns. Each kernel is race-free in one launch (reads interior, writes pad
  only); the 2D pad is padv-then-padh sequenced by the host (as upstream's
  DeblockPlane does: padh over `height+2*vpad`). Grids 2D `(width,vpad)` /
  `(hpad,height)`; preconditions `vpad <= height`, `hpad <= width` (always true
  upstream: pad counts are 8 for Deblock, 1 for CombingAnalyze flag planes).
  Verified solo plus the composed padv→padh order, 8/16-bit.
- `kf_merge_block` — the `MergeBlock` masked blender (`cpu_merge`/`kl_merge`
  — exact twins; driven per plane by KPatchCombe/KFMSwitch): `dst =
  (flag*src60 + (128-flag)*src24 + 64) >> 7` with the uchar comb flag (uchar
  even for 16-bit pixels, as upstream). Same formula family as
  `kf_merge_static` but a different twin/roles; no clamping — the full uchar
  flag domain is transcribed verbatim (flag > 128 drives the addend negative
  through an arithmetic `>> 7`, with the `(PX)` cast wrapping exactly like
  `VHelper::cast_to`), and the 0..255 flag sweep pins that path. 8/16-bit.
- `kf_average` — field-pair temporal mean (`cpu_average`/`kl_average` — exact
  twins; uchar4/ushort4): `dst = (src0+src1) >> 1` per pixel (floor mean; sums
  are non-negative so the shift is exact and the result always in range; odd
  sums pin the floor path). 8/16-bit.
- `kf_max` — per-pixel max (`cpu_max`/`kl_max` — exact twins; uint8_t ONLY
  instantiation upstream, the uchar4 line is commented out — so the uchar
  kernel at arbitrary width is exact parity). Faithful oddity: upstream
  assigns the int tmp straight to the uint8 dst (its `cast_to` line is
  commented out); the value is always in range so this equals a plain store.
- `kf_merge_uvflags` — `MergeUVFlags` core (`cpu_merge_uvflags` /
  `kl_merge_uvflags` — exact twins; uint8, scalar): in-place `fY |= ((fU|fV)
  << 4)` with the UV flag read at subsampled offset `(x>>logUVx, y>>logUVy)`.
  Race-free (each lane touches only its own fY element; U/V read-only); the
  shift is int arithmetic and the uint8_t `|=` store wraps mod 256 (full
  uchar sweep pins the wrap).
- `kf_copy_border` — the extreme-row copy behind the `kf_extend_coef2` header
  note (`cpu_copy_border`/`kl_copy_border` — exact twins; CPU-fallback helper
  of the ExtendCoefs fallback path): for y in [0,vborder), rows y and
  height-y-1 straight from src to dst (bottom index verbatim
  `(height - y - 1)`). dst is pre-filled to differ everywhere, so stray writes
  show; the `vborder*2 > height` lane overlap is covered (benign: colliding
  lanes write identical values). 8/16-bit.
- `kf_analyze_frame` — `CompareFields` flag classifier
  (`cpu_analyze_frame`/`kl_analyze_frame` — exact twins; uchar4/ushort4
  sources via `LaunchAnalyzeFrame`, uchar flags; distinct from the uchar2
  block-based `kl_analyze_frame` that lives in CombingAnalyze.cu): `t =
  |a+4c+e-3(b+d)|` UNSHIFTED from base rows y-1/y/y+1 and sref rows y/y+1
  (no `>> 2` here, unlike `kf_calc_combe`; max 6*PX_MAX, no int overflow),
  `diff = |mref[y]-base[y]|`, `flag = (t>threshS ? SHIMA:0) |
  (t>threshLS ? LSHIMA:0) | (diff>threshM ? MOVE:0)` with MOVE=1, SHIMA=2,
  LSHIMA=4 (KFM.h). Sources point at the interior origin of VPAD-padded planes
  (host pad layout/offset is a RIG-VERIFY seam as with the pads) — but the
  border outputs are defined by the pad, so ALL height rows are verified with
  padded inputs, with crafted fields pinning the strict-`>` boundaries at
  t/diff = 0, 1, maxv and 6*maxv. 8/16-bit.
- `kf_copy_pad` — padded-frame copy (`kl_copy_pad`; uchar4/ushort4; driven by
  `CopyFrameAndPad` with hpad = 0; NO upstream CPU twin exists — the CPU
  fallback is CopyFrame + PadFrame — so it is checked against the device
  kernel's own semantics, as `kf_extend_coef2` is). Copies src into dst's
  interior while mirror-padding hpad/vpad around it (dst at interior origin,
  src at plane origin). The uchar4 lane-reversal quirk cancels EXACTLY
  (reversed lanes over mirrored vector columns compose to the plain
  per-pixel mirror — proof in the `.cl` comment), so the scalar port is the
  plain mirror; the verification mirrors the VECTOR algorithm with the swap
  while the golden is the plain pixel mirror, so their agreement IS the
  empirical proof — plus a copy→padv→padh composition cross-check per case.
  8/16-bit, mult-4 widths, hpad mult-4 (production hpad is 0: the swap never
  even runs upstream).
- `kf_copy_pad_2plane` — dual-plane padded copy (`kl_copy_pad_2plane`;
  U/V pair; same per-plane semantics). CUDA spreads planes over blockIdx.z;
  the `.cl` covers both planes per thread over a 2D grid — plane-independent
  values, identical results (upstream keeps the two-launch form commented at
  the call site). 8/16-bit.
- `kf_max_extend_blocks_h` / `kf_max_extend_blocks_v` — the `ExtendBlocks`
  ping-pong passes (`kl_max_extend_blocks_h/v`; uint8_t + uchar4
  instantiations, both used by CombingAnalyze — no uint16 — so uchar is exact
  parity, at any width for uint8 and mult-4 for uchar4 lanes whose max() is
  per-lane). Out-of-place, race-free: the far edge self-copies (no neighbour
  beyond), the near edge takes the neighbour outright (col 0 / row 0 discard
  self), interior takes max(self, neighbour); verbatim branch order (far-edge
  check first, so nBlk == 1 self-copies — pinned). Verified solo plus
  composed h→v (dst→tmp→dst) against the in-place 3-pass
  `cpu_max_extend_blocks` algorithm — the mirror runs the ping-pong while the
  golden runs the in-place path, so their agreement proves the composition
  equals the CPU twin (for nBlkX,nBlkY ≥ 2; at 1 the CPU twin reads out of
  bounds, so production block counts stay ≥ 2).
- `kf_copy` — plain same-pitch plane copy (`kl_copy`; uint8/uint16/uchar4/
  ushort4 instantiations, lanes independent). Covers the CombingAnalyze
  field-copy production shape (pitch*2 over height/2 rows).
- `kf_copy_2plane` — dual-plane copy (`kl_copy_2plane`; CombingAnalyze UV
  field copy + KDeband UV copy). Upstream's blockIdx.z plane selector becomes
  the 3rd grid dimension (launch contract: exactly 2).
- `kf_fill` — plane fill with a runtime value (KDeband `kl_fill` twin; also
  covers FilterBase `kl_fill<pixel_t, 0>`, whose template value is only ever 0
  upstream). Production use is UV/flag-plane zeroing.

These, together with `kf_min_frames` and `kf_and_coefs` (in kfm_mergestatic.cl),
make KAnalyzeStatic's kernel set complete (see the MergeStatic section for the
host-assembly seam). The pad helpers additionally close the KDeblock pad-kernel
gap (see below) — only the pad/merge *host sequencing* remains.

### `KFMKernel.cu` inventory (host-only — no device kernels)

All 875 lines read (upstream `8e086bb`): the file defines **zero**
`__global__`/`__device__` functions and issues zero kernel launches. Its five
filters, with dispositions:

- `KPatchCombe` — pulldown-aware 24p/30p frame selection + one `MergeBlock`
  call per frame. Kernel-complete via `kf_merge_block`; the frame-index
  bookkeeping is AviSynth host logic.
- `KFMSwitch` — 60/30/24/UCF frame-switch state machine + timecode-file
  writer + `MergeBlock` calls. Kernel-complete via `kf_merge_block`; the
  switching/durations logic is host. (Its `VisualizeFlag` is a CPU-only debug
  visualisation, not a device kernel — not ported.)
- `KFMPad` — VPAD vertical pad via `CopyFrameAndPad` (device path:
  `kl_copy_pad`/`kl_copy_pad_2plane`; CPU path: copy + `kl_padv`). Dispatch
  is host; the pad kernels are ported as `kf_copy_pad`/`kf_copy_pad_2plane`
  (see above), so KFMPad is kernel-complete too.
- `KFMDecimate` — pure frame-index remapping from a durations file. No pixels
  touched; nothing to port.
- `AssumeDevice` — pure cache-hint filter. Nothing to port.

So KFMKernel.cu is closed: nothing further to transcribe here.

### `KNoiseClip` (kf_noise_clip, src/opencl/kfm/kernels/kfm_noiseclip.cl)

A self-contained **8-bit-only** AVS filter from DecombeUCF.cu
(`KNoiseClip(clip, noise, nmin_y, range_y, nmin_uv, range_uv)`, a src-vs-noise
difference/activity map used by the KDecombUCF family). It is driven per plane
by a single kernel (`cpu_noise_clip` / `kl_noise_clip` — exact CPU twin), so it
is kernel-complete (only AviSynth host glue remains). Per pixel (integer-exact):
`out = dev_limitter((src - noise + 256) >> 1, nmin, range)`, where
`dev_limitter` maps `s == 128` (equal) to 128, the band
`(127-range)<s<(128-nmin)` to 0 / otherwise 56 below 128, the band
`(128+nmin)<s<(129+range)` to 255 / otherwise 199 above. `// ALG-VERIFIED` via
`python/run_kfm_noiseclip.py` (300 cases: CPU mirror `sim/kfm_noiseclip_ref.cpp`
vs an independent Python golden over nmin/range sweeps). Y uses `nmin_y`/
`range_y`; U,V use `nmin_uv`/`range_uv`. (Host requires plane width %4==0 — the
scalar port is exact for any width.)

### DecombeUCF reductions (`kfm_decombeucf.cl`, 7 kernels)

The remaining `__global__` kernels of DecombeUCF.cu (after `kl_noise_clip`
above): `kf_init_uint64`, `kf_calculate_field_diff` (gated 5-tap
`CalcCombe(a,b,c,d,e) = abs(a+c*4+e-(b+d)*3)` sum over a ±2-row padded plane,
8/16-bit), `kf_init_block_sum`, `kf_add_block_sum` (per-block
sumAbs/sumSig with `BLOCK_SIZE` 4/8/16/32 as a runtime arg, 8/16-bit),
`kf_block_sum_max` (per-cell `sumAbs+sumSig*4` max over interleaved int
quads, 0-floored), `kf_analyze_noise` (4-way `|.-128|`/`|delta|` census,
8-bit), `kf_analyze_diff` (combe sum0 + field-mixed TFF sum1, 8-bit).
Upstream's warp-shuffle `dev_reduce`/`dev_reduceN` trees become full
`__local` halving trees (int add/max are associative and commutative, so
every tree shape is value-identical); the `+=`/`atomicAdd`/`atomicMax`
accumulation onto init values is preserved, including the 64-bit result
accumulators (`// RIG-VERIFY`: needs `cl_khr_int64_base_atomics` or OpenCL
2.0 atomics on device; the 32×16 reduce kernels require local size 32×16).
Together with `kf_noise_clip`, all 8 DecombeUCF.cu device kernels are
ported — only the multi-clip host pipelines (`KDecombUCF*`) remain.
`// ALG-VERIFIED` via `python/run_kfm_decombeucf.py` (810 cases: CPU
mirror `sim/kfm_decombeucf_ref.cpp` vs an independent Python golden,
integer-exact, nonzero-init accumulation and padded-plane crafts covered).

### `KDeblock` core (kf_deblock, src/opencl/kfm/kernels/kfm_deblock.cl)

The heart of the KDeblock deblocking filter (`kl_deblock` in Deblock.cu), a
faithful scalar transcription. `// ALG-VERIFIED` via `python/run_kfm_deblock.py`
(300 cases) against the CPU mirror `sim/kfm_deblock_ref.cpp` and an independent
float32-exact Python golden. The 8×8 DCT/IDCT is the fixed float32 butterfly of
Devblock.cu (`dev_dct8`/`dev_idct8`, `S1..S2` constants); the mirror is built
`-ffp-contract=off` and the golden rounds every float32 op, so they match
bit-for-bit. Per 8×8 block it applies `count = 1<<quality` differently-offset
DCT→hard-threshold→IDCT reconstructions and accumulates into a 16-bit
block-parity plane (`out`). The AC threshold is
`qp_apply_thresh(qp,thresh_a,thresh_b)*((1<<2)+strength)-1`; the DC coefficient
(index 0) is never thresholded.

Fidelity notes (honest):
- `kl_deblock` always runs DCT→hardthresh→IDCT. The *CPU-only* fallback
  `cpu_deblock`/`cpu_deblock_avx` additionally has a `thresh <= 0` identity
  shortcut (multiply by 64, skip the transform) that the device kernel does NOT
  have. This OpenCL port follows the **device kernel** (the transform path).
- The CUDA device stores the block DCT in a 9-stride shared buffer and
  `dev_hardthresh` walks it with an 8-stride loop; on the valid 64 coefficients
  that is equivalent to thresholding every non-DC coefficient (the CUDA padding
  cells are unused), which is what the scalar 8-stride port does. Output
  identical.
- Host seam (RIG-VERIFY): KDeblock::DeblockPlane first mirror-pads the plane
  (8 px/side) into `src` (pad kernels ported as `kf_padv`/`kf_padh`, see the
  KFMFilterBase section — only their host sequencing is left) and later merges
  the accumulator — that merge is transcribed as `kf_merge_deblock` (see below)
  but its accumulator-layout reconciliation is reasoned, not run. The core DCT
  stage plus the QP-table/show helpers are verified on a self-consistent
  padded-src config.
- `QPClip` (same source file) is a no-op host filter that only copies frame
  properties to a 2×2 Y8 frame — no kernel, so nothing to port/verify there.

### `KDeblock` QP table + show (`kf_make_qp_table` / `kf_deblock_show`,
src/opencl/kfm/kernels/kfm_deblock.cl)

- `kf_make_qp_table` (twin of `kl_make_qp_table`/`cpu_make_qp_table`, part of
  the `QPForDeblock` helper that feeds KDeblock): downsamples macroblock (16px)
  QP plane(s) into the per-8px-block uint16 QP table `kf_deblock` consumes.
  Source macroblock at `(min(x>>qp_shift_x, w-1), min(y>>qp_shift_y, h-1))`;
  element-max of an optional second (non-B) QP source; a per-macroblock DC luma
  level blends the normalized block/non-block distortions `b`/`nonb`
  (`norm_qscale` for QP scale type 0/1/2/3 = MPEG1/2/H264/VP56) via
  `ratio = min(1, dc*dc_coeff)`, `qp = max(1,(int)(b*ratio+nonb*(1-ratio)+.5))`.
  Constant (`force_qp`) mode when no source QP plane exists. Null-vs-present
  CUDA pointers are carried as presence booleans. Float blend emulated float32.
- `kf_deblock_show` (twin of `kl_deblock_show`/`cpu_deblock_show`, KDeblock
  `show==2`): paints each 8px QP block `230` if deblocking is enabled for it
  (`qp_apply_thresh(qp) >= qp>>1`) else `16`, into the visible plane; blocks
  tile without overlap from an origin offset of −4 (deterministic).
  Both `// ALG-VERIFIED` via `python/run_kfm_deblock_qp.py`
  (200+200 cases, integer-exact, float32-emulated blend) against
  `sim/kfm_deblock_qp_ref.cpp`.

### `KDeblock` sharpen (`kf_sharpen`, `kf_show_sharpen_coeff`,
src/opencl/kfm/kernels/kfm_deblock_rig.cl) — `// RIG-VERIFY` transcriptions

Faithful scalar transcriptions of the remaining self-contained Deblock.cu
device kernels, kept in a **separate provisional file**
(`kfm_deblock_rig.cl`, bannered PROVISIONAL/UNVERIFIED) so their
needs-verification status is unmistakable next to the ALG-VERIFIED
`kfm_deblock.cl`. Lanes/pixels are independent in all of them, so the scalar
form is lane-identical to the vector CUDA kernels. They are **not** covered by
`make test` (no CPU mirror / Python golden yet) and must be checked on a real
OpenCL device before use. The full verification handoff spec for another
agent (upstream line map, per-kernel traps, mirror+golden recipe, graduation
checklist) is `docs/RIG_HANDOFF_KDEBLOCK.md`.

Graduated to `kfm_deblock.cl` (`// ALG-VERIFIED` via
`python/run_kfm_deblock_aux.py`, 1593 cases vs `sim/kfm_deblock_aux_ref.cpp`):
`kf_max_vh`/`kf_max_v`/`kf_max_h` (radius-`R` box-max dilation, 8px-margin
padded harness, radius 1..8 with 5 = production; `max_vh` cross-checked via
the separable `max_h` o `max_v` identity), `kf_scale_qp` (full-range inputs,
mod-256 wrap pins, out-of-range scale types), `kf_sharpen_coeff` (`qp`
swept 0..65535 with `q` = 24/25 boundary emphasis; LUT bytes diffed against
upstream) and `kf_merge_deblock` (250 merge cases: quality 1..6 x bits
8/10/12/16, spike crafts pinning the k/X/L summation, dither-boundary flips;
15 end-to-end cases on real `kf_deblock` accumulators; 5 handoff-section-5
packing identity checks — packed ushort2 == scalar ushort proven, not
reasoned). The remaining sharpen pair is also pinned by the same runner
(P/W modes: quirk configs, c==0 identity, bilinear ramps — deterministic
behaviour only; the texture-gap device run is still open). Still in the
rig file:

- `kf_sharpen` (twin of `kl_sharpen`, SharpenFilter core): 3×3 edge-clamped
  min/max window (transcribes the device form verbatim, including the upstream
  `min(x+1,height-1)` quirk), `c = bilinear(coeff,x/8,y/8)/255`,
  `dst = (int)clamp(s+(s-u)*c+0.5, l, h)` with the unsharp sharing dst pitch.
  The CUDA texture fetch (Clamp/Linear/NormalizedFloat) is replaced by manual
  float32 bilinear (`kf_sharpen_bilinear`, the CPU twins' verbatim
  expression) — structurally exact (the clamp never engages: width % 8 == 0
  plus the qp-sized coeff margin keep taps in bounds), but HW fixed-point vs
  float32 rounding can differ at truncation boundaries, so a device-run
  comparison is mandatory (see the handoff doc §4.5).
- `kf_show_sharpen_coeff` (twin of `kl_show_sharpen_coeff`, the `show`
  visualiser): `dst = (int)bilinear(coeff,x/8,y/8)` — exactly the CPU twin's
  expression, so mirror+golden can pin it; the texture gap vs the device still
  needs the device run to close.

### `CombingAnalyze.cu` (complete — all 15 device kernels,
src/opencl/kfm/kernels/kfm_combinganalyze.cl)

Fifteen kernels (KSwitchFlag/KCombeMask/KRemoveCombe/KCleanSuper/KContainsCombe
stages, the KFMSuper block analyzer, the FMCount census) plus the two 8-tap
helpers, all `// ALG-VERIFIED` via `python/run_kfm_combinganalyze.py`
(2731 cases) against the CPU mirror `sim/kfm_combinganalyze_ref.cpp` and an
independent Python golden. Every kernel has an exact `cpu_*` twin upstream.

- `kf_calc_combe8` / `kf_calc_diff8` — the `__host__ __device__` 8-tap helpers
  (`calc_combe`/`calc_diff`): combe = diffT-diffE-diffO over 7+4+4 absdiffs
  (smooth fields give ~0, alternating combs large positive, can go negative),
  diff = 4 field-pair absdiffs. Tap-level verified incl. sign flips.
- `kf_copy_first` — `.x` lane extract from super-frame vectors (`lanes` covers
  the template stride; 2 = production uchar2, 4 = uchar4).
- `kf_combe_to_flag` — 2x2 round-half-up quarter-mean downsample (the host
  launches it over the flag interior, skipping row 0/col 0 — RIG-VERIFY seam).
- `kf_sum_box3x3` — quartered 3x3 smooth, min with maxv (NOT /9 — upstream
  always quarters); 1-px halo contract, verified over all outputs; ping-pongs
  through a tmp frame upstream (never in-place).
- `kf_binary_flag` — `(Y>=thY||C>=thC)?128:0`, verified in-place (dst == srcY
  upstream; race-free by lane-locality).
- `kf_bilinear_h/v` — separable upscale, `(s0*c0+s1*c1+HALF)>>SHIFT` with
  HALF = SCALE/2 (SCALE/SHIFT as args; production (4,2)/(8,3)); arithmetic >>
  for the negative top row/col; 1-col/1-row halo; always in [0,255], no clamp.
- `kf_temporal_soften` — float32 3-frame mean, truncated toward zero; output
  depends only on the byte sum t (all adds exact), so the t = 0..765 sweep is
  exhaustive. The `(1.0f/3.0f)` constant fold is the only RIG-VERIFY item
  (must fold to 0x3EAAAAAB, as every mainstream compiler does).
- `kf_remove_combe2` — 4x4-combe-gated vertical `(a+2b+c+2)>>2` (8/16-bit);
  VPAD-halo contract on src; combe `.x` gate with `>=` boundary pins.
- `kf_clean_super` — per-plane super cleaner (CUDA z-splits the U/V pair; the
  host launches per plane, as the cpu twin does): zero `.x` where prev.y <=
  thresh && cur.y <= thresh.
- `kf_init_contains_durty_block` / `kf_contains_durty_block` — OR-scan of the
  flag plane into one int (the scan's `*work = 1` is an idempotent race).

- `kf_super_analyze` — the KFMSuper block analyzer (`kl_analyze_frame` /
  `cpu_analyze_frame`; uchar2 flags, parity as an int arg, 8/16-bit pixels):
  per 4x4-stride cell, 8 taps accumulate top/bottom combe + top/bottom field
  diffs (TFF/BFF tap routing), written as clamp(sum>>shift) at col bx+1, rows
  2*(by+1)+{0,1}. The serial per-cell loop replaces the warp reduction
  (integer sums are order-exact — it is the cpu twin); border cells write
  nothing (flag col 0 / rows 0-1 stay sentinel-pinned; their content is a
  host-seam detail, as upstream). Interlaced/split/constant crafts pin the
  combe path, the 255 clamp, and the all-zero path.
- `kf_init_fmcount` — `<<<1,2>>>` zero-fill of the two `FMCount` slots
  (`{move, shima, lshima}` split into 3 arrays of 2).
- `kf_count_cmflags` / `kf_count_cmflags_2planes` — the FMCount threshold
  census (block-reduce + global atomics into slot `i ^ !parity`; 2planes
  fuses U+V with per-item counts 0..2). Faithful structure: per-item 0/1
  counts, work-group tree reduction over fixed 512-item (32x16) groups in
  `__local` memory, one atomic per group per field (skipped at 0). Verified
  serially (order-exact for ints, onto zero and nonzero inits, with `>=`
  boundary pins); the 2planes golden runs two single-plane passes, proving
  fused == U+V composition (upstream's CPU path calls the single twin 3x).

With these, every `__global__` kernel in CombingAnalyze.cu is ported — only
the host-only filter classes (KFMSuper/KPreCycleAnalyze/KSwitchFlag/
KCombeMask/KContainsCombe/KRemoveCombe/KCleanSuper dispatch, AviSynth glue)
remain, plus the noted RIG-VERIFY seams (flag-interior offsets, halo/pad
layouts, the 1/3-fold constant, unwritten flag borders).

## Next candidates (verifiable in this sandbox)

- `kl_copy` / `kl_fill` helpers (already covered generically by KTGMC `kt_copy`).
- `KTemporalDiff` / `KMergeStatic` host glue (they are now kernel-complete; the
  AviSynth frame-fetch / NewVideoFrame / CopyFrame seam could be recorded as the
  next concrete on-rig step).
- Deblock remaining: the SharpenFilter host (coeff-frame allocation,
  GaussResize unsharp clip, per-plane dispatch), the KDeblock pad/merge host
  sequencing (`kf_padv`/`kf_padh` themselves are done), and the rig proofs
  (merge accumulator-layout reconciliation + the sharpen-pair texture-gap
  device comparison — see above and the handoff doc). With `kf_sharpen` /
  `kf_show_sharpen_coeff` transcribed, all 11 Deblock.cu device kernels now
  exist as OpenCL. Of the eight `kfm_deblock_rig.cl` kernels packaged for
  handoff in `docs/RIG_HANDOFF_KDEBLOCK.md`, six have since graduated to
  `// ALG-VERIFIED` in `kfm_deblock.cl`; only the sharpen pair remains
  `// RIG-VERIFY` (pinned P/W behaviour — still mandates a device run).
- Remaining KFM families: the DecombeUCF.cu host pipelines (KDecombUCF* —
  heavy multi-clip sequencing; all 8 device kernels are done: KNoiseClip +
  the 7 reductions in kfm_decombeucf.cl). Deblock QPClip is a pure
  host/props filter with no device kernel
  (FrameType is CPU-only); ShowQP's kernel `kl_scale_qp` is transcribed
  (`kf_scale_qp`, `// RIG-VERIFY`), its frame-assembly is host.
- `KFMFilterBase.cu` is now fully ported: `kl_copy_pad`/`kl_copy_pad_2plane`
  (also backing `KFMPad`) and `kl_max_extend_blocks_h/v` are `kf_copy_pad` /
  `kf_copy_pad_2plane` / `kf_max_extend_blocks_h/v` above, completing the
  CombingAnalyze batch from this file. `kl_copy`/`kl_fill` stay covered by
  KTGMC `kt_copy` (memcpy-equivalent, no port planned). The uchar2
  block-based `kl_analyze_frame` in CombingAnalyze.cu is a different kernel
  and stays with that file's batch.

## Notes / licence

KFM is MIT; distribute derived files under MIT terms (differs from KTGMC's GPL).
Grounding sources: this sandbox re-clones `AviSynthCUDAFilters` to
`/tmp/avs_cuda/KFM`.
urces: this sandbox re-clones `AviSynthCUDAFilters` to
`/tmp/avs_cuda/KFM`.
