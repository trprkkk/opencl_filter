# MV motion engine — CUDA → OpenCL port specification

This document grounds the port of the KTGMC motion-compensation engine
(`KTGMC/MVKernel.cu` ≈3540 lines + `KTGMC/MV.cpp` ≈5550 lines) so each kernel
can be transliterated faithfully and audited against the original. It is the
reference for `src/opencl/ktgmc/kernels/ktgmc_motion.cl`.

> Status: the full MV engine is a large, multi-stage effort. Everything in the
> motion file is a faithful source port of the CUDA kernels **but is NOT
> runtime-verified here** (no OpenCL/CUDA device in the build sandbox). All
> motion kernels carry a `// RIG-VERIFY` marker until they are cross-checked on
> real hardware against the CUDA implementation (see §7).

## 1. Pipeline overview

The KTGMC motion path mirrors AviSynth `mvtools`, split into three AVS filter
stages plus the QTGMC script that chains them:

1. **KMSuper** — builds the *super frame*: the current frame upsampled (2× in
   each axis for `pel=2`, or 4× for `pel=4`) and stored **separated by sub-pel
   offset**, plus padded borders, as a small pyramid of levels. Each level is
   used as the reference for motion search at that resolution.
2. **KMAnalyse** — per level and per **batch** of rows, runs block motion
   search and writes one `VECTOR{x,y,sad}` per block, coarse→fine
   (predictors from the coarser level + spatial/temporal neighbours).
3. **KMCompensate / KMDegrain1/2** — uses the MV arrays to fetch each block
   from the super frames and write a compensated / weighted denoised frame.

## 2. Core data structures (from source)

From `common/KMV.h` and `KTGMC/MVKernel.cu`:

```cpp
struct VECTOR { int x; int y; int sad; };           // KMV.h (also {x,y} used as short2 internally)
struct LevelInfo { int nBlkX; int nBlkY; };          // KMV.h

// MVKernel.cu
struct SearchBlock {          // one per block, passed in a "search batch"
  int   data[12];  // [0..3]=CLIP_RECT (DxMax,DyMax,DxMin,DyMin); [4..9]=ref-MV indices
                   //  (Left, Up, bottom-right-from-coarse); [10..11]=PRED_X/Y
  sad_t dataf[5];  // [0..3]=penalties (zero,global,0,predictor/new); [4]=lambda
};
struct CostResult { sad_t cost; short2 xy; };
struct SearchBatch { /* holds per-block pointers + ref/MV arrays (see source) */ };

// MVKernel.cu template structs used by degrain/compensate kernels
template<typename pixel_t,int N> struct DegrainBlockData{ const short* winOver; const pixel_t* pSrc; const pixel_t* pB[N],*pF[N]; int WSrc,WRefB[N],WRefF[N]; };
template<typename pixel_t,int N> struct DegrainArgData { const VECTOR* mvB[N],*mvF[N]; const pixel_t* pSrc,*pRefB[N],*pRefF[N]; bool isUsableB[N],isUsableF[N]; };
// both are passed by "union { struct d; uint32 m[LEN]; }" (pointer arithmetic trick)
```

The AVS-side plane parameters live in `MV.cpp` (`MVPlaneParam`, block sizes,
`nBlkX/nBlkY`, overlap, `nPel`, level chain). The search host state machine
(predictor selection, meander direction, level loop, global-MV estimation,
scene-change) is in `MV.cpp` and must be reproduced on the host for a full
plugin; the kernels below are the device-side pieces.

## 3. Kernel inventory (MVKernel.cu, by line) and role

| CUDA | Role | MV stage |
|---|---|---|
| `memcpy_kernel` (26) | plain byte copy | generic |
| `kl_copy_pad` (39) | copy plane into padded buffer, mirror edges | super build |
| `kl_pad_frame_h/v` (65/84) | in-place horizontal/vertical edge replication | super build |
| `kl_vertical/horizontal_wiener` (107/135) | 6-tap Wiener interpolation (already ported → `kt_*_wiener` in ktgmc_simple.cl) | super "sharp" |
| `kl_RB2B_bilinear_filtered(_with_pad)` (175/229) | bilinear 1/8,3/8,3/8,1/8 sub-pel sampler | super |
| `dev_clip_mv`, `dev_check_mv`, `dev_sq_norm` (315/321/326) | MV helpers | search |
| `dev_get_ref_block` (333) | pick ref block pointer incl. sub-pel plane offset | search |
| `load4pix(_Aligned)` (357/368) | 4-byte vector load + funnel-shift (alignment trick) | search |
| `dev_calc_sad` (375) | block SAD (Y + optional chroma) + reduce | search |
| `MinCost`, `dev_reduce_result` (479/490) | keep lowest-cost candidate, block reduce | search |
| `dev_expanding_search_1/2` (523/635) | square expanding diamond pattern cost | search |
| `dev_hex2_search_1` (715) | hexagonal (hex2) refinement pattern | search |
| `dev_read_pixels` (782) | shared-memory tile load for hex search | search |
| `Search` (860) | one block batch search driver kernel (uses all above) | search |
| `kl_calc_all_sad` (1135) | compute SAD across full refs (initial) | search/scene |
| `kl_prepare_search` (1267) | build per-block SearchBlock data | search |
| `kl_most_freq_mv` / `kl_mean_global_mv` (1383/1447) | global-MV estimation | search |
| `kl_interpolate_prediction` (1567) | coarse→fine MV upscale/prediction | search |
| `kl_load_mv_batch/load/store`, `kl_write_default_mv`, `kl_init_const_vec` | MV array I/O between levels | search |
| `kl_init_scene_change`, `kl_scene_change(_x2)` (1761/1767/1790) | per-frame scene change | search |
| `dev_degrain_weight`, `dev_norm_weights` (1843/1857) | per-block weight → 256 scale | degrain |
| `kl_prepare_degrain` (1930), `kl_degrain_2x3` (2021) | block-weighted temporal denoise | degrain |
| `kl_short_to_byte` (2157) | convert tmp to output | degrain |
| `kl_prepare_compensate` (2188), `kl_compensate_2x3` (2239) | motion compensation fetch | compensate |
| `kl_short_to_byte_or_copy_src` (2371) | compensate output | compensate |

## 4. CUDA → OpenCL equivalence rules (logical identity)

These make each kernel transliteratable while keeping **identical results**:

| CUDA device construct | OpenCL equivalent |
|---|---|
| `threadIdx/blockIdx/blockDim` (x,y,z), `gridDim.z` for A/B plane pairs | `get_local_id`, `get_group_id`, `get_local_size`, `get_global_id`; keep same indexing arithmetic |
| `__shared__ T s[N]` | `__local T s[N]`; same `barrier(CLK_LOCAL_MEM_FENCE)` for `__syncthreads` |
| warp shuffle reduce `dev_reduce_warp*` over integer SAD | reduction to `__local`, then a final partial by lane 0 — integer addition is order-independent ⇒ identical |
| `__vabsdiff4(a,b,sad)` / `__sad` | `sad += |aByte-bByte|` per pixel; the packed-4 / funnel-shift (`__funnelshift_rc`) / 4-byte-aligned load are *load* optimizations only — scalar per-pixel absolute difference yields the same sums |
| `atomicAdd(int* , int)` | `atomic_add(global int*, int)` (order irrelevant for integer sums) |
| `uchar4/ushort4/int4` packed channels | scalar loops over the 4 channels with identical per-channel math |
| `short2` | store as 2-element or packed via two ints in `int2` |
| `min/max/clamp`, `>>5` on non-negative int | identical in OpenCL C |
| `__restrict__` | `__restrict` |

Caveat for floats (rare here): keep *identical* operation order inside each
output (e.g. `dev_degrain_weight` truncates `(int)`, order is fixed).

## 5. Super-frame layout notes (needed before Search is meaningful)

`dev_get_ref_block` (NPEL 2 / 4) treats the reference as a set of **NPEL×NPEL
sub-pel planes** stacked `nImgPitch` apart: integer MV maps to plane offset
`si = sx + sy*NPEL`, then `x=vx/NPEL, y=vy/NPEL` into that plane. Porting the
`Search` kernels therefore requires the host to lay the super frame out with
`NPEL*NPEL` planes each `nPitch×nImgHeight`; this layout is constructed in
`MV.cpp` `KMSuper::Create` (the Wiener/RB2B kernels write these planes). Getting
the plane pitch/stride arithmetic exact is the crux of on-rig verification.

## 6. What has been ported / verified

In `src/opencl/ktgmc/kernels/ktgmc_simple.cl` (bit-for-bit CPU+Python verified,
8 & 16 bit): the two Wiener interpolation kernels (`kt_vertical_wiener`,
`kt_horizontal_wiener`) and `kt_plane_sad` (frame SAD metric) plus the 22
per-plane KTGMC kernels.

`src/opencl/ktgmc/kernels/ktgmc_motion.cl` holds the self-contained MV pieces:
- **ALG-VERIFIED** (CPU + Python golden, `make test`): the degrain weight
  helpers `kt_degrain_weight`/`kt_norm_weights` (`run_motion_core.py`, 706
  cases); the MV-aux integer kernels `kt_write_default_mv`, `kt_scene_change`
  /`_x2`, `kt_short_to_byte`, `kt_short_to_byte_or_copy_src` (`run_mv_aux.py`,
  310 cases); and the coarse→fine MV upsampler `kt_interpolate_prediction`
  (`run_mv_interp.py`, 200 random cases, bilinear parity 4-neighbour weights);
  and the per-row global-MV refinement `kt_mean_global_mv` (average of vectors
  within 6 of the median estimate; `run_mv_mean.py`, 400 rowgroups — upstream
  does a 1024-thread staged/shuffle reduction, but the integer sums are
  order-independent so a serial per-row accumulation matches exactly).
  The per-block search setup `kt_prepare_search` (search bounds, predictor
  slot indices, prior-level vector copy, penalties and the lambda schedule;
  ANALYZE_SYNC=1) is ALG-VERIFIED (`run_mv_searchprep.py`, 300 cases).
  The MV I/O quartet `kt_load_mv`, `kt_store_mv`, `kt_load_mv_batch` (the degrain
  / compensate per-batch split + out copy-through), `kt_init_const_vec` (split /
  recombine VECTOR int3 with the int2-vector + int-sad buffers; write the two
  per-row sentinels slot -2 = zero-vector, slot -1 = globalMV*nPel) is
  ALG-VERIFIED (`run_mv_io.py`, 200+60 cases).
  The reduced-plane builder `kt_rb2b_bilinear_filtered` (separable (1,3,3,1)/8
  anti-aliased 1:2 downsample, KMSuper `ReduceTo`) is ALG-VERIFIED
  (`run_mv_rb2b.py`, 200 cases): the single-pass OpenCL form recomputes the two
  separate phase roundings and matches the CPU `RB2BilinearFiltered` reference
  bit-for-bit.
  Its CUDA-only fused twin `kt_rb2b_bilinear_filtered_with_pad` — the same
  1:2 downsample done as ONE weighted 4×4-tap filter with a single +32/64
  rounding that also fills the destination hpad/vpad border (MV.cpp
  `ReduceToPad`; the CPU path instead runs the separable core + a separate
  `Pad()`, so the two forms are numerically distinct) — is ALG-VERIFIED
  (`run_mv_rb2b_pad.py`, 250 cases) against an independent C++ mirror that
  transliterates the fused CUDA kernel verbatim.
  Note: upstream `kl_write_default_mv` sets `.x` twice (a typo for `.sad`); we
  implement the intended default.
  The degrain/compensate OVERLAP pixel-combiner core — `kt_degrain_patch`
  (Degrain1to6_C weighted denoise of a block) + `kt_overlap_out` (feathered
  Overlaps_C window accumulation + Short2Bytes, per output pixel over its <= 2x2
  covering blocks, with edge source-copy) — is ALG-VERIFIED (`run_mv_degrain.py`,
  300 cases): its per-pixel summation matches a faithful MV.cpp-staging CPU
  mirror (`sim/ktgmc_degrain_ref.cpp`) bit-for-bit for 8/16-bit, overlap=0 and
  overlap=nBlkSize/2.  These are the only degrain/compensate kernels that are
  verifiable in-sandbox; they take the per-block reference-plane element offsets
  (refBaseB/F) as resolved inputs — that resolution is the host/rig seam (§6.1).
- **RIG-VERIFY** (faithful source ports, device run pending): the frame padding
  / mirror-copy kernels (`kt_copy_pad`, `kt_pad_frame_h/v`), `kt_init_scene_change`,
  `kt_most_freq_mv`, the block-search pure helpers (`kt_clip_mv`/`kt_check_mv`/
  `kt_sq_norm`/`kt_ref_block_offset`), and the first block-level kernel
  `kt_calc_all_sad` (per-block SAD vs the MV-selected ref block). These follow
  the authoritative host model in `docs/BLOCKSEARCH_MODEL.md`. That doc records
  the key fact that the shipped CUDA launches compile with `CPU_EMU=true` — the
  search runs the deterministic, CPU-ordered scalar path (not warp shuffles), so
  a faithful scalar OpenCL port can reproduce it exactly.
  `kt_most_freq_mv` returns the smallest most-frequent component (bit-exact vs
  CUDA whenever the mode is unique, confirmed by a 20 000-row simulation); when
  several values tie for the mode, CUDA's winner is an artifact of its 1024-thread
  reduction tree and is not reproduced — reconcile on the rig only if bit-exact
  tie output is required.
All remaining kernels are **not** independently dispatchable/verifiable in this
sandbox because their inputs come from the `SearchBatchData` super-frame host
struct (built by the `MV.cpp` host state machine) and/or use warp/block
reduction geometry. Concretely, the unported set splits into:

- **Now ported (RIG-VERIFY):** `kl_calc_all_sad` and the pure helpers
  `dev_check_mv` / `dev_clip_mv` / `dev_sq_norm` / `dev_get_ref_block`
  (`kt_ref_block_offset`), which the block kernels share.
- **Intrinsic to the warp/block `Search` driver** (only meaningful inside the
  block kernel; cannot be validated standalone): `Search`, `dev_read_pixels`,
  `dev_expanding_search_1/2`, `dev_hex2_search_1`, `MinCost`,
  `dev_reduce_result`, `load4pix(_Aligned)`, and `kl_prepare_search`'s consumer
  loop.
- **Degrain / compensate block kernels** (need `DegrainBlockData`/`ArgData`
  super-frame pointers + MV arrays): `kl_prepare_degrain`, `kl_degrain_2x3`,
  `kl_prepare_compensate`, `kl_compensate_2x3`.

### 6.1 Why the degrain / compensate per-pixel kernels stay rig-bound

The two `_prepare` kernels (`kl_prepare_degrain`, `kl_prepare_compensate`) are
per-block and self-contained in principle: for block (bx,by) they pick one of 9
overlap windows via `wby = 3*((by+nBlkY-3)/(nBlkY-2))`, `wbx = (bx+nBlkX-3)/(nBlkX-2)`,
slot `wby+wbx`; compute block origins `offx=bx*blkStep`, `offy=by*blkStep`
(`blkStep=nBlkSize/2`) and `offsetS = nPad+offx + (nPad+offy)*nPitchSuper`;
resolve each B/F ref to an element offset via the ALG-verified `dev_get_ref_block`
(= `kt_ref_block_offset`); and normalise `WSrc/WRefB/WRefF` via the ALG-verified
weight helpers. In MV.cpp the CPU reference (`KMDegrainCore::Proc`,
`KMCompensateCore::Proc`, overlap branch) instead spaces blocks at `StepX =
nBlkSizeX-nOverlapX` and reads `GetBlock(ix,iy).x = ix*StepX`, then maps an
output pixel to the super plane at `block.x*nPel + mv…` through
`KMPlane::GetPointer` (adds padding). So a *faithful device transliteration* must
decide which block-geometry / super-plane coordinate model it implements — the
MV.cpp `nPel`/padding/pel semantics (`nOverlap`, `nPel`, `nHPad/nVPad`,
`nLogxRatio/yRatio`) are the ground truth, and they are exactly the super-frame
host model this doc §5 flags as rig-only.

The per-pixel `kl_degrain_2x3` / `kl_compensate_2x3` kernels are additionally
defined by a **host launch geometry**, not by their own indexing: they accumulate
weighted block patches (`Degrain1to6_C`-style `>>8` denoise, then an
`Overlaps_C`-style `(px*winOver+256)>>6` overlap-add into a shared tmp, later
`Short2Bytes`-style `>>5`/`>>11`) but tile the plane via `basex =
(blockIdx.x*M+nPatternX)*SPAN_X`, `basey=(blockIdx.y*2+nPatternY)*SPAN_Y` with
`SPAN_X=3, SPAN_Y=2` and dispatch several `(nPatternX,nPatternY,M)` instances
from MV.cpp. Reproducing their exact global-tmp writes therefore requires that
host pattern/grid, plus the window generation (`OverlapWindows`, 9 feathered
windows of `nBlkSize*nBlkSize`), the block-geometry model of §6.1, and the
per-plane `pDst` tmp layout — i.e. the full on-rig host assembly, not a
self-contained kernel. We therefore record these as rig-bound rather than ship a
speculative scalar re-derivation. The arithmetic they need is now *not* missing:
the ALG-VERIFIED combiner core `kt_degrain_patch` + `kt_overlap_out` (§6, above)
is the order-equivalent, verified way to produce identical output once the host
supplies the per-block ref-plane base offsets and windows (see §6.2).
- **Separate KTGMC temporal-filter path** (Kernel.cu, not MV): `kl_init_sad`,
  `kl_calculate_sad`, `kl_copy_boarder1(_v)`, `kl_logic1/2/3`, `kl_box3_v`,
  `kl_box5_v_and_border`, `kl_binomial_temporal_soften_1/2` — several use packed
  `vpixel_t`/`__vabsdiff4` or float `atomicAdd` reductions whose output ordering
  is not bit-deterministic, so they are ported scalar / as flags only.

### 6.2 The one seam between verified kernels and the rig

The degrain/compensate combiner math is verified; the *only* thing left to wire
is turning the MV.cpp `KMPlane`/`GetPointer` reference semantics into the flat
per-block element offsets (`refBaseB/refBaseF`, indexed `k*nBlk+blk`) that
`kt_degrain_patch` consumes, plus the host-side 9-window generation. Concretely:
`KMPlane::SetTarget` stacks `nPel*nPel` sub-pel planes `nPitch*nExtendedHeight`
apart; `GetPointer(nX,nY)` folds the low `log2(nPel)` bits of `nX,nY` into the
plane index, adds `nHPadPel/nVPadPel`, and the block base is
`block.x*nPel + mv[k][blk].x` (block.x = `bx*StepX`, StepX = `nBlkSize-overlap`).
That offset maps directly onto the verified `kt_ref_block_offset` helper. A
concise step-by-step recipe for the follow-up agent / rig lives in
`docs/CODEX_HANDOFF.md`.

So the isolated, per-output deterministic MV kernels are complete
(22 in `ktgmc_motion.cl` + 25 in `ktgmc_simple.cl`), and `kt_calc_all_sad`
extends the set into the first block-level (MV + super-frame) kernel.
Finishing KTGMC from here means assembling the `SearchBatchData` + super-frame
layout on a real rig (see `docs/HOST_CONTRACT.md`) and then porting/validating
the remaining groups above there.

## 7. Verification plan on a real rig

1. Bring up an OpenCL build (see top-level CMakeLists `OpenCL::OpenCL` block;
   install `ocl-icd-opencl-dev` + a device, e.g. `pocl-opencl-icd` or vendor
   runtime) and compile `ktgmc_simple.cl`/`ktgmc_motion.cl`.
2. First run `make lint` (or CTest `ktgmc_cl_lint`): it does a genuine C parse of
   every `.cl` through `lint/oc_shim.h` to catch structural/typo errors. gcc does
   NOT parse a `.cl` file on its own (it treats it as a linker input), so the
   script copies each source to `.c` and uses `-x c`. The shim only emulates
   syntax (vector `.x/.y` members, work-item ids, `atomic_add`, scalar builtins),
   not OpenCL semantics.
3. Add a host runner that applies each kernel to raw planes and diffs against the
   existing `sim/ktgmc_cpu_ref.cpp` output (the same vectors already used by
   `python/run_validation.py`). This validates every kernel not tied to the MV
   search data model. See `docs/HOST_CONTRACT.md` for the exact program builds,
   kernel→grid maps, argument order and — importantly — the `int2`/`int3` host
   buffer-layout probe to run before exercising the MV kernels.
4. For the search/degrain/compensate kernels, build the `MV.cpp` host state
   machine (predictors, meander, level loop) and compare `VECTOR` arrays + output
   frames against the CUDA build of `AviSynthCUDAFilters` on identical inputs
   (or against `mvtools` output where the parameterization matches).
5. Remove each `RIG-VERIFY` marker as its test passes.

## 8. Licensing

`KTGMC` (incl. the MV engine) is **GPL** per the upstream `AviSynthCUDAFilters`
README; `common/KMV.h` type defs come from the same tree. Distribute derived
files under the matching upstream terms.
