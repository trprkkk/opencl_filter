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
  Note: upstream `kl_write_default_mv` sets `.x` twice (a typo for `.sad`); we
  implement the intended default.
- **RIG-VERIFY** (faithful source ports, device run pending): `kt_copy_pad`,
  `kt_pad_frame_h`, `kt_pad_frame_v`, `kt_init_scene_change`.
All search / degrain-block / compensate kernels (block SAD, expanding/hex2
search, `kl_degrain_2x3`, `kl_compensate_2x3`) still need the MV.cpp host state
machine and super-frame layout before they can be assembled and validated.

## 7. Verification plan on a real rig

1. Bring up an OpenCL build (see top-level CMakeLists `OpenCL::OpenCL` block;
   install `ocl-icd-opencl-dev` + a device, e.g. `pocl-opencl-icd` or vendor
   runtime) and compile `ktgmc_simple.cl`/`ktgmc_motion.cl`.
2. Add a host runner that applies each kernel to raw planes and diffs against
   the existing `sim/ktgmc_cpu_ref.cpp` output (the same vectors already used by
   `python/run_validation.py`). This validates every kernel not tied to the MV
   search data model.
3. For the search/degrain/compensate kernels, build the `MV.cpp` host state
   machine (predictors, meander, level loop) and compare `VECTOR` arrays + output
   frames against the CUDA build of `AviSynthCUDAFilters` on identical inputs
   (or against `mvtools` output where the parameterization matches).
4. Remove each `RIG-VERIFY` marker as its test passes.

## 8. Licensing

`KTGMC` (incl. the MV engine) is **GPL** per the upstream `AviSynthCUDAFilters`
README; `common/KMV.h` type defs come from the same tree. Distribute derived
files under the matching upstream terms.
