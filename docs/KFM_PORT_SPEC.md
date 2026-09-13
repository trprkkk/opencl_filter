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
| `Deblock.cu` | `KDeblock`, `QPClip`, `ShowQP`, `FrameType` | **KDeblock done** (`kf_deblock`, `kf_make_qp_table`, `kf_deblock_show` ALG-VERIFIED; `kf_merge_deblock`, `kf_max_vh/v/h`, `kf_scale_qp`, `kf_sharpen_coeff` in the separate provisional `kfm_deblock_rig.cl`, `// RIG-VERIFY`, handoff spec in `docs/RIG_HANDOFF_KDEBLOCK.md`; QPClip is a no-op pass-through) |
| `DecombeUCF.cu` | `KCFieldDiff`, `KCFrameDiffDup`, `KNoiseClip`, `KAnalyzeNoise`, `KDecombUCF*` | **KNoiseClip done** (see below) |
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
kf_extend_coef2 / kf_apply_uvcoefs_420 / kf_padv / kf_padh,
src/opencl/kfm/kernels/kfm_filterbase.cl)

`KFMFilterBase.cu` is the shared base class and defines the coefficient kernels
that KAnalyzeStatic is assembled from (`cpu_*` twins exist in the same file).
Six are ported here and `// ALG-VERIFIED` via `python/run_kfm_filterbase.py`
(1130 cases) against the CPU mirror `sim/kfm_filterbase_ref.cpp` and an
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

These, together with `kf_min_frames` and `kf_and_coefs` (in kfm_mergestatic.cl),
make KAnalyzeStatic's kernel set complete (see the MergeStatic section for the
host-assembly seam). The pad helpers additionally close the KDeblock pad-kernel
gap (see below) — only the pad/merge *host sequencing* remains.

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

### `KDeblock` merge / DC-mask / ShowQP / sharpen LUT (`kf_merge_deblock`,
`kf_max_vh/v/h`, `kf_scale_qp`, `kf_sharpen_coeff`,
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

- `kf_merge_deblock` (twin of `kl_merge_deblock`/`cpu_merge_deblock`): per
  visible pixel sums the 4 parity-slice accumulator rows
  (`tmp_ipitch_rows` = bh*8 rows each), scales by `1/(1<<shift)`
  (`shift` = mergeShift = quality+6-deblockShift), adds the Bayer dither
  `g_ldither[y&7][(x>>2)&1][x&3]/64` (table contents verbatim from Deblock.cu;
  note the middle index runs over ushort4 columns), clamps with `fmin(v,maxv)`
  (`maxv` = (1<<bits)-1) and C-truncates to `PX` — matching CUDA's
  `VHelper::cast_to` (truncation, no lower clamp) and per-component
  `min(float4,float)`. Grid is visible pixels; `vis_width` must be a multiple
  of 4 (pass `width & ~3`, matching CUDA's `width>>2` lanes). The `tmp` base /
  pitch conventions mirror the CUDA call exactly (base pre-offset by +8 ushorts
  / +8 rows, pitch in ushort4 units). Layout reconciliation: CUDA `kl_deblock`
  stores packed ushort2 at `blockIdx.x*4` while `kf_deblock` writes scalar
  ushort at `bx*8` — the same bytes when the byte stride matches
  (ushort2-packed == row-major ushort, incl. the atomicAdd-packed halves on
  little-endian), so the merge consumes `kf_deblock`'s output directly with
  `tmp_pitch_u4 = acc_pitch_ushort >> 2`. Reasoned, not run.
- `kf_max_vh` / `kf_max_v` / `kf_max_h` (twins of `kl_max_vh`/`kl_max_v`/
  `kl_max_h`; `cpu_max_v`/`cpu_max_h` are exact CPU twins, `kl_max_vh` is
  device-only upstream): radius-`R` box-max dilation of the DC luma mask in
  the `QPForDeblock` helper (upstream always instantiates `RADIUS=5`; radius
  is a kernel arg here). Reads span ±radius around every pixel, so the caller
  must pass the interior of a plane padded by ≥ radius (same edge contract as
  the CUDA host, which passes pad+8+8*pitch with an 8 px margin).
- `kf_scale_qp` (twin of `kl_scale_qp`/`cpu_scale_qp`, the ShowQP filter
  kernel): per-pixel `(uchar)norm_qscale(src, scale_type)` via `kf_norm_qscale`
  (int→uchar wraps mod 256 exactly like the CUDA assignment).
- `kf_sharpen_coeff` (twin of `kl_sharpen_coeff`/`cpu_sharpen_coeff`): QP-block
  → sharpen-strength LUT `q = qp>>3; dst = (q>=25) ? 255 : g_sharpen_coeff[q]`
  (30-entry table verbatim). Feeds the SharpenFilter, not KDeblock itself.

## Next candidates (verifiable in this sandbox)

- `kl_copy` / `kl_fill` helpers (already covered generically by KTGMC `kt_copy`).
- `KTemporalDiff` / `KMergeStatic` host glue (they are now kernel-complete; the
  AviSynth frame-fetch / NewVideoFrame / CopyFrame seam could be recorded as the
  next concrete on-rig step).
- Deblock remaining: `kl_sharpen` + `kl_show_sharpen_coeff` (both sample the
  coeff plane through a CUDA texture object — need a sampler/image redesign +
  the SharpenFilter host, incl. the GaussResize unsharp clip; note the device
  `kl_sharpen` clamps `x+1` by `height-1`, an upstream quirk to preserve), the
  KDeblock pad/merge host sequencing (`kf_padv`/`kf_padh` themselves are done),
  and the rig proof of the merge accumulator-layout reconciliation (see
  above). Upgrade path: the six `kfm_deblock_rig.cl` kernels are packaged for
  handoff in `docs/RIG_HANDOFF_KDEBLOCK.md` — `kf_max_v/h`, `kf_scale_qp`,
  `kf_sharpen_coeff` have exact CPU twins and `kf_merge_deblock`/`kf_max_vh`
  are deterministic float32 / integer work, so all six are future ALG-VERIFY
  candidates under the mirror+golden method.
- Remaining KFM families: CombingAnalyze.cu (KFMSuper/…; super-frame motion
  state) and the rest of DecombeUCF.cu (KDecombUCF* — heavy multi-clip host
  pipelines). KFMKernel.cu and Deblock QPClip are pure host/props filters with
  no device kernels (FrameType is CPU-only); ShowQP's kernel `kl_scale_qp` is
  transcribed (`kf_scale_qp`, `// RIG-VERIFY`), its frame-assembly is host.

## Notes / licence

KFM is MIT; distribute derived files under MIT terms (differs from KTGMC's GPL).
Grounding sources: this sandbox re-clones `AviSynthCUDAFilters` to
`/tmp/avs_cuda/KFM`.
