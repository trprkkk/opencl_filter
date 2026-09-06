# Port plan — KTGMC (and beyond) from CUDA → OpenCL

This document is the engineering map for porting
[`rigaya/AviSynthCUDAFilters`](https://github.com/rigaya/AviSynthCUDAFilters)
(→ `nekopanda/AviSynthCUDAFilters`) to OpenCL. It records the verified source
inventory, the chosen strategy, per-kernel status, and how to extend the repo.

## 1. Source inventory (verified from `master`)

KTGMC is split into three CUDA files plus AVS glue:

| File | Lines | Contents |
|---|---|---|
| `KTGMC/Kernel.cu` | ~3560 | **27** `__global__` kernels: the per-plane/no-MV pixel kernels **and** all the AviSynth `KTGMC_*`/`K*` filter classes (`KTGMC_Bob`, `KBinomialTemporalSoften`, `KRemoveGrain`, `KRepair`, `KVerticalCleaner`, `KGaussResize`, masktools-style `KMakeDiff/KAddDiff/KLogic/KMerge`, `KXpand/Expand`, bob shimmer fixes, VResharpen/Resharpen, LimitOverSharpen, ToFullRange, TweakSearchClip, LosslessProc, ErrorAdjust, `KDoubleWeave/KWeave`, `KCopy`, and the `ResamplingFunction`/`ResamplingProgram` framework). Registers 25 AVS functions. |
| `KTGMC/MVKernel.cu` | ~3540 | **44** device functions/kernels for motion estimation: `KMSuper` (super-sampled pyramid), `KMAnalyse` (SAD, block search, MV vector prediction/refinement, scene-change), and `KMCompensate` motion compensation. |
| `KTGMC/MV.cpp` | ~5550 | Host pipeline + `KMSuper`/`KMAnalyse`/`KMDegrain1/2`/`KMCompensate` classes + AVS registration. This is a port of AviSynth `mvtools` to CUDA. |

Other suite projects (next milestones after KTGMC): `KNNEDI3` (neural-net
upscaler), `KFM` (KDeband, Deblock, CombingAnalyze, DecombeUCF, MergeStatic…),
`AvsCUDA` (AviSynthNeo CUDA plumbing), `GRunT`, `masktools`.

## 2. Strategy

1. **Per-pixel separable kernels first.** Most `Kernel.cu` kernels process a
   plane pixel-by-pixel (or 4 pixels per CUDA thread via `uchar4`/`ushort4`).
   Each output element is an *independent* dot/point/sort operation that never
   reads a sibling channel, so a **scalar translation is bit-identical**. This is
   exactly what Milestone 1 does. Vectorization is a later, pure-performance pass.
2. **Double implementation for correctness.** Every kernel exists as (a) a scalar
   OpenCL `.cl` and (b) a scalar CPU mirror. An independent Python golden is the
   third, cross-language check. `make test` runs CPU-vs-Python bit-for-bit.
3. **Reuse QSVEnc's OpenCL idioms.** `rigaya/QSVEnc/QSVPipeline/rgy_filter_*.cl`
   (e.g. the OpenCL `--vpp-kfm`, `--vpp-degrain` filters) show the house style:
   `rgy_CL*` buffer/host helpers, `#pragma` / build-option handling, and the
   kernel-source-as-string approach. Mirror that for a drop-in feel.
4. **Host glue separated from kernels.** The AviSynth host adaptation and the
   motion-vector pipeline are ported separately from the pixel kernels, so each
   milestone is independently testable without AviSynth.

### Precision rules preserved from CUDA (do not "fix")
- Resample accumulates in `float` (CUDA) — scalar CPU reference uses `double`
  only because that is the *algorithm*; the `.cl` resampler uses `float` to match
  real CUDA bit behaviour. All other kernels are integer-exact.
- Rounding is always `clamp(x, 0, maxval)` then `+0.5` before truncation
  (`(int)(x + 0.5)`), matching CUDA `cast_to(x + 0.5f)`.
- Pixel range: `maxval = 255` (8-bit) or `65535` (16-bit); plane `ComponentSize`
  selects the kernel instantiation, exactly like the `switch (pixelSize)` in the
  CUDA `Proc()` templates.
- Fixed point: `KMerge` uses `(w*32767.0f)` → `32767` scale, `>>15`.
- `MakeDiff`: `a-b+range_half`, `range_half = 1 << (bits-1)`.

## 3. Milestone status map (Kernel.cu simple kernels)

`src/opencl/ktgmc/kernels/ktgmc_simple.cl` currently contains (✔ = validated
bit-for-bit via `make test`, both 8-bit and 16-bit):

| CUDA kernel (Kernel.cu) | OpenCL | Status |
|---|---|---|
| `kl_resample_v` + `ResamplingFunction`/`Program` | `kt_resample_v` | ✔ |
| `kl_resample_h` | `kt_resample_h` | ✔ |
| `kl_makediff` (`MakeDiffOp`) | `kt_makediff mode 0` | ✔ |
| `kl_makediff` (`AddDiffOp`) | `kt_makediff mode 1` | ✔ |
| `kl_box3x3_filter` RG11/RG20 | `kt_rg_box3x3` | ✔ |
| `kl_rg_clip` N=1..4 | `kt_removegrain_clip` | ✔ |
| `kl_repair_clip` N=1..4 | `kt_repair_clip` | ✔ |
| `kl_vertical_cleaner_median` | `kt_vertical_cleaner_median` | ✔ |
| `kl_box5_v_and_border` (Min5/Max5) | `kt_box5_minmax` | ✔ |
| `kl_logic2` (LogicMin/Max) | `kt_logic_minmax` | ✔ |
| `kl_box3_v` (Resharpen) | `kt_vresharpen` | ✔ |
| `kl_resharpen` | `kt_resharpen` | ✔ |
| `kl_limit_over_sharpen` | `kt_limit_over_sharpen` | ✔ |
| `kl_to_full_range` (Y / UV) | `kt_to_full_range` | ✔ |
| `kl_bobshimmerfixes_merge` | `kt_bobshimmerfixes_merge` | ✔ |
| `kl_tweak_search_clip` | `kt_tweak_search_clip` | ✔ |
| `kl_error_adjust` | `kt_error_adjust` | ✔ |
| `kl_lossless_proc` | `kt_lossless_proc` | ✔ |
| `kl_merge` | `kt_merge` | ✔ |
| `kl_binomial_temporal_soften_1` | `kt_temporal_soften_1` | ✔¹ |
| `kl_binomial_temporal_soften_2` | `kt_temporal_soften_2` | ✔¹ |
| `kl_weave` (KDoubleWeave) | `kt_weave` | ✔ |
| `kl_copy` / `kl_elementwise Copy` | `kt_copy` | ✔ |
| `kl_vertical_wiener` (MVKernel.cu, KMSuper "sharp") | `kt_vertical_wiener` | ✔ |
| `kl_horizontal_wiener` (MVKernel.cu, KMSuper "sharp") | `kt_horizontal_wiener` | ✔ |
| `kl_calculate_sad` frame-level (Kernel.cu temporal soften) | `kt_plane_sad` | ✔² |

² `kt_plane_sad` is the frame/plane absolute-diff sum (scene-change gate). The
CUDA block/warp reductions are intrinsic/layout-specific; the *metric* is the
scalar `sum |a-b|` which is validated here. Wiring it into the temporal-soften
`scN` scene flags is host logic shown in the kernel comment.

¹ Scene-change replacement (the CUDA shared-memory `isSC[]` reduction) is exposed
as scalar per-frame flags `scN` here; wiring the full SAD-based
`kl_calculate_sad` reduction is part of the motion stage.

Motion / super-sampling kernels now live in `src/opencl/ktgmc/kernels/ktgmc_motion.cl`
(see `docs/MV_PORT_SPEC.md`). ALG-VERIFIED: the degrain weight helpers
(`dev_degrain_weight`, `dev_norm_weights`), the MV-aux integer kernels
(`kl_write_default_mv`, `kl_scene_change`/`_x2`, `kl_short_to_byte`,
`kl_short_to_byte_or_copy_src`), the coarse→fine MV upsampler
`kl_interpolate_prediction`, the global-MV refinement `kl_mean_global_mv`, and
the per-block search setup `kl_prepare_search` (ANALYZE_SYNC=1), and the MV
I/O trio `kl_load_mv`/`kl_store_mv`/`kl_init_const_vec`.
Source-ported (RIG-VERIFY, device run pending):
`kl_copy_pad`, `kl_pad_frame_h/v`, `kl_init_scene_change`, and
`kl_most_freq_mv` (smallest-mode; bit-exact vs CUDA except on exact mode ties).

TODO (not yet ported): `kl_logic1/kl_logic3`, `kl_calculate_sad` (block-level),
`kl_init_sad`, `kl_copy_boarder1(_v)`, `kl_RB2B_bilinear_filtered(_with_pad)`,
the block-search kernels (`Search`, expanding/hex2), `kl_degrain_2x3`,
`kl_compensate_2x3`, `kl_scene_change*`, `kl_write_default_mv`, and the MV.cpp
host state machine. `GaussianFilter` (KGaussResize) is already implemented as a
second `ResamplingFunction` in the reference; add its `.cl` variants next.

## 4. Motion-compensation stages (the big remaining work)

To get a working deinterlacer you must port the mvtools-equivalent layers:

1. **Super sampling** (`KMSuper`, `MVKernel.cu`): separable upscale to 4× pel,
   levels pyramid. — **in progress**: the `kl_vertical_wiener`/`kl_horizontal_wiener`
   ("sharp") kernels are ported & validated; `ktgmc_motion.cl` adds the frame
   padding / mirror-copy kernels (source port, **RIG-VERIFY**); the RB2B
   sub-pel sampler and the padded super-frame layout still need the host buffer
   semantics from `MV.cpp`. Full data-model + verification plan:
   [`docs/MV_PORT_SPEC.md`](MV_PORT_SPEC.md).
2. **Analysis** (`KMAnalyse`, `kl_calculate_sad`, block search, temporal/vector
   prediction, scene-change detection, `dev_reduce` block reductions → OpenCL
   `barrier`/local reduce). — **frame-level SAD started** (`kt_plane_sad`,
   validated); the block-level SAD (`dev_calc_sad`, warp reductions, `load4pix`)
   and the block *search* (expanding / hex2 diamond) remain and are
   architecture-layout-specific (see note ² above).
3. **Compensation / Degrain** (`KMCompensate`, `kl_compensate_2x3`,
   `kl_degrain_2x3`, `kl_prepare_*`).
4. **Assembly** — the QTGMC AVS script wires individual `KTGMC_*`/`K*` filters
   (Bob → … → Repair/Resharpen → weave). The AviSynth graph lives on the host
   side and is device-independent once each leaf filter runs on OpenCL.

## 5. Beyond KTGMC

- **KNNEDI3**: self-contained neural-net 2× scaler; medium-large.
- **KFM**: pick smallest filter first (KDeband) to prove the plumbing, then the
  rest. KFM is MIT → cleanest to reuse/redistribute.

## 6. AviSynth integration (device glue)

Real plugins need AviSynthNeo host code. That layer (frame `GetFrame`, plane
pitch/read/write pointers, `RegisterFunction`) is **separate** from the kernels
and must be written/run where AviSynthNeo exists (Windows, or Linux AviSynth+).
The kernels here are pure `plane → plane` operations so they can be unit-tested
with no AviSynth. When wiring up, follow the original classes' plane loop
(`PLANAR_Y/U/V`, `uvSamePitch` short-circuit, `logUVx/logUVy` subsampling).

## 7. Licensing

KTGMC-derived files follow the upstream **GPL** (per
`AviSynthCUDAFilters` README); KFM-derived files are **MIT**. Keep provenance
per file/directory. Nothing in this repo should be released without matching the
upstream license of the code it derives from.

## 8. Validation runner details

- `python/run_validation.py` generates deterministic raw planes (8 & 16 bit),
  computes the Python golden, builds `sim/ktgmc_cpu_ref.cpp`, runs it, and
  compares every output **and** the resampling-program `offset`/`coef` tables
  bit-for-bit.
- Add a new kernel by: implement it in `ktgmc_simple.cl`, mirror it in
  `sim/ktgmc_cpu_ref.cpp`, add the Python golden + a comparison entry.
