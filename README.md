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
| KFM | filter family (KDeband, Deblock, CombingAnalyze, …) | **KDeband/KEdgeLevel/KTemporalNR + MergeStatic/KAnalyzeStatic/KNoiseClip + KDeblock done** (core/qp-table/show/max/scale/sharpen-coeff/merge ALG-VERIFIED; sharpen-pair in provisional `kfm_deblock_rig.cl`, `// RIG-VERIFY`) (`src/opencl/kfm/kernels/`) |
| AvsCUDA / GRunT / masktools | CUDA-aware dispatch + helpers | out of scope unless requested |

A faithful KTGMC port is large: `KTGMC/Kernel.cu` (~3,560 lines) + `MVKernel.cu`
(~3,540) + `MV.cpp` (~5,550) implement the whole QTGMC algorithm including
multi-level motion estimation, block search and motion compensation. This repo
therefore progresses in **milestones**; each milestone's kernels are translated
faithfully and validated bit-for-bit against an independent reference.

### Milestone 1 (validated core)

A validated set of the *per-plane, no-motion* KTGMC kernels (the parts that need
no motion vectors), plus the host-side `ResamplingProgram`. All 22 kernels below
are present in `src/opencl/ktgmc/kernels/ktgmc_simple.cl` and cross-validated
bit-for-bit (8-bit and 16-bit) against independent C++ and Python references via
`make test`.

Motion super-sampling (stage 2, from `KTGMC/MVKernel.cu`, the `KMSuper`
"sharp" interpolation path):
- `kt_vertical_wiener` / `kt_horizontal_wiener` — 6-tap Wiener interpolation.

Resample / resize (host FIR `ResamplingProgram`, Mitchell/Catmull-Rom + more):
- `kt_resample_v` / `kt_resample_h` — the vertical & horizontal resamplers used
  by the "Bob" field-interpolation engine and the GaussResize-style path.

masktools-style combining:
- `kt_makediff` / add-diff (mt_makediff / mt_adddiff equivalent),
- `kt_logic_minmax` (KLogic min/max), `kt_merge` (32767 fixed point, KMerge).

Denoise / spatial:
- `kt_rg_box3x3` (RemoveGrain 11/12 & 20), `kt_removegrain_clip` (modes 1–4,
  8-neighbour sort network), `kt_repair_clip` (modes 1–4, 9-neighbour sort
  network), `kt_vertical_cleaner_median` (mode 1), `kt_box5_minmax`
  (Inpand/Expand ×2 vertical).

Bob/weave / sharper / misc:
- `kt_vresharpen`, `kt_resharpen`, `kt_limit_over_sharpen`,
  `kt_to_full_range` (Y & UV), `kt_bobshimmerfixes_merge`,
  `kt_tweak_search_clip`, `kt_error_adjust`, `kt_lossless_proc`,
  `kt_temporal_soften_1/2` (binomial, scene-change flags as scalars),
  `kt_weave` (KDoubleWeave), `kt_copy`.

Each is a faithful scalar OpenCL port (see the per-file notes about why scalar is
mathematically identical). See [`docs/PORT_PLAN.md`](docs/PORT_PLAN.md) for the
full kernel-by-kernel roadmap and how to extend the set.

## Layout

```
docs/PORT_PLAN.md              # port strategy + per-kernel roadmap + licensing
docs/MV_PORT_SPEC.md           # motion-engine data model + kernel inventory + rig test plan
docs/HOST_CONTRACT.md          # on-rig OpenCL host runner spec (builds, grids, buffer layout)
docs/BLOCKSEARCH_MODEL.md      # KTGMC block-search host model (SearchBatch, super-frame, CPU_EMU)
docs/CODEX_HANDOFF.md          # step-by-step recipe for the rig-bound MV remainder
docs/KFM_PORT_SPEC.md          # KFM filter-family map + verified/next status
docs/RIG_HANDOFF_KDEBLOCK.md   # verification handoff spec for the provisional KDeblock kernels
docs/AVSCUDA_PORT_SPEC.md      # AvsCUDA family map + batch status (18/18 ported)
docs/RIG_HANDOFF_AVSCUDA_CONDITIONAL.md  # device-run handoff for the 2 float RIG-VERIFY reductions
docs/PERF_NOTES.md             # static perf pass: work-group contracts, atomics, vectorization
src/opencl/ktgmc/kernels/      # OpenCL kernel sources: KTGMC motion/simple
src/opencl/kfm/kernels/        # OpenCL kernel sources: KFM (deband/edgelevel/temporalnr/mergestatic/filterbase/noiseclip/deblock/deblock_rig .cl)
src/opencl/avscuda/kernels/    # OpenCL kernel sources: AvsCUDA (merge/filters/convert/conditional/conditional_rig/resample)
src/host/                      # OpenCL host harnesses (rig-only; need an ICD, not in make test)
third_party/grunt/             # verbatim upstream GRunT (CPU-only AviSynth plugin, no CUDA to port)
sim/ktgmc_cpu_ref.cpp          # scalar CPU mirror of the kernels (validates logic)
python/run_validation.py       # independent Python golden + cross-check harness
Makefile                       # make test  (no OpenCL required)
```

## Port status

- `ktgmc_simple.cl`: 25 per-plane KTGMC kernels — **bit-for-bit validated**
  (8/16-bit) via CPU + Python references (`make test` PASS). KGaussResize
  needs no new kernels (same `kt_resample_h/v` at fir 8/9); its gaussian
  program builder + fir-8/9 planes/tables are covered by `run_validation.py`.
- `ktgmc_motion.cl`: motion / super-sampling kernels. **ALG-VERIFIED**: the
  degrain weight helpers (`kt_degrain_weight`, `kt_norm_weights`), MV-aux
  kernels (`kt_write_default_mv`, `kt_scene_change`/`_x2`, `kt_short_to_byte`,
  `kt_short_to_byte_or_copy_src`), the coarse→fine MV upsampler
  `kt_interpolate_prediction`, the global-MV refinement `kt_mean_global_mv`,
  the per-block search setup `kt_prepare_search`, the MV I/O quartet
  `kt_load_mv`/`kt_store_mv`/`kt_load_mv_batch`/`kt_init_const_vec`, and both
  reduced-plane builders `kt_rb2b_bilinear_filtered` (separable) and
  `kt_rb2b_bilinear_filtered_with_pad` (CUDA-only fused twin that also fills the
  hpad/vpad border), and the degrain/compensate pixel-combiner core
  `kt_degrain_patch` + `kt_overlap_out` (the full Degrain1to6_C/Overlaps_C/
  Short2Bytes overlap arithmetic, ALG-VERIFIED vs the MV.cpp staging mirror).
  **RIG-VERIFY** (device run pending): frame padding / mirror copy,
  `kt_most_freq_mv` (smallest-mode seed), the block-search pure helpers, and the
  first block-level kernel `kt_calc_all_sad` (per-block SAD; host model in
  `docs/BLOCKSEARCH_MODEL.md`). The remaining search / degrain-block /
  compensate kernels need the MV.cpp host state machine + super-frame sub-pel
  layout (see `docs/MV_PORT_SPEC.md`, `docs/CODEX_HANDOFF.md`).
- `kfm_deband.cl`: **KFM KDeband core** — `kf_deband_reduce_banding`, faithful to
  the authoritative CPU twin `cpu_reduce_banding` (KFM/KDeband.cu, MIT).
  **ALG-VERIFIED** via `python/run_kfm_deband.py` (300 cases: verbatim CPU
  mirror `sim/kfm_deband_ref.cpp` vs independent Python golden; sample modes
  0-2, blur_first, 8/16-bit).
- `kfm_edgelevel.cl`: **KFM KEdgeLevel** — `kf_edgelevel`,
  `kf_edgelevel_repair`, `kf_el_to444`, `kf_el_from444`, faithful to their CPU
  twins in KFM/KDeband.cu (MIT). Edge detection/enhance + visualise (`check`),
  the 3×3 `repair` limiter (interior ALG-VERIFIED; border needs a padded plane
  on the rig), and the 4:2:x↔4:4:4 chroma pack/unpack helpers. Float math is
  IEEE float32 with no FMA contraction. **ALG-VERIFIED** via
  `python/run_kfm_edgelevel.py` (500 cases: CPU mirror
  `sim/kfm_edgelevel_ref.cpp` vs an independent float32-exact Python golden,
  all 8 check/selective/uv combos + repair N=1-4 + to/from444, 8/16-bit).
- `kfm_temporalnr.cl`: **KFM KTemporalNR** — `kf_temporal_nr`, faithful to
  `cpu_temporal_nr` (KFM/KDeband.cu, MIT). Temporal noise reduction: for each
  pixel averages the same-location pixels across `2*dist+1` frames that are
  within `thresh` of the centre (`avg = (float)sum/count + 0.5f`). Column-wise,
  so a scalar port is identical to the CUDA vectorised one. **ALG-VERIFIED** via
  `python/run_kfm_temporalnr.py` (300 cases: CPU mirror
  `sim/kfm_temporalnr_ref.cpp` vs an independent float32-exact Python golden,
  dist 0-6, 8/16-bit). NOTE: OpenCL uses normal fp32 `/` (matches the CPU twin);
  the real CUDA device kernel uses the approximate `__fdividef` (≈ulp-level CUDA
  HW nuance, no OpenCL equivalent) — see spec.
- `kfm_mergestatic.cl`: **KFM MergeStatic.cu kernels** — `kf_compare_frames`
  (KTemporalDiff core, 5-frame max−min), `kf_min_frames` (KAnalyzeStatic
  temporal-min sub-step), `kf_and_coefs` (KAnalyzeStatic combing∧static
  sub-step, float32), `kf_merge_static` (KMergeStatic core,
  `(coef*v30+(128-coef)*v60+64)>>7`). KTemporalDiff and KMergeStatic are
  kernel-complete (only AviSynth host glue remains); KAnalyzeStatic's remaining
  coefficient kernels are in kfm_filterbase.cl. **ALG-VERIFIED** via
  `python/run_kfm_mergestatic.py` (580 cases: CPU mirror
  `sim/kfm_mergestatic_ref.cpp` vs an independent float32-exact Python golden).
- `kfm_filterbase.cl`: **KFM KFMFilterBase coefficient + pad + merge kernels** —
  `kf_calc_combe` (CompareFields combing, `|a+4c+e-3(b+d)|>>2` clamped [0,255];
  interior ALG-VERIFIED, border needs a VPAD-padded plane on the rig),
  `kf_merge_uvcoefs` (fold UV into Y), `kf_extend_coef2` (ExtendCoefs; = the
  CUDA `kl_extend_coef2` device kernel — upstream's CPU fallback differs at the
  extreme rows), `kf_apply_uvcoefs_420` (YV12 Y→UV), the shared in-place mirror
  pads `kf_padv`/`kf_padh` (exact `cpu_padv`/`cpu_padh` twins, race-free; 2D pad
  = padv-then-padh host sequencing, as DeblockPlane does), and the MergeBlock
  blender `kf_merge_block` (`(flag*src60+(128-flag)*src24+64)>>7`, uchar flag at
  both depths, full 0..255 flag sweep; makes KPatchCombe/KFMSwitch
  kernel-complete), plus the CombingAnalyze/CompareFields helpers `kf_average`
  (floor mean), `kf_max` (uint8 max; scalar twin, so any width),
  `kf_merge_uvflags` (in-place `fY|=((fU|fV)<<4)`, mod-256 wrap covered),
  `kf_copy_border` (extreme rows straight through) and `kf_analyze_frame`
  (SHIMA/LSHIMA/MOVE threshold fold over padded taps, all rows verified),
  the padded-frame copies `kf_copy_pad`/`kf_copy_pad_2plane` (lane-swap
  quirk proven to cancel; no upstream CPU twin) and the ExtendBlocks
  ping-pong `kf_max_extend_blocks_h/v` (composed h->v proven equal to the
  in-place CPU twin), and the plain plane utilities `kf_copy` /
  `kf_copy_2plane` (3D grid over the plane pair) / `kf_fill` (runtime value;
  also covers `kl_fill<*,0>`). These plus `kf_min_frames`/`kf_and_coefs`
  complete
  KAnalyzeStatic's kernel set (only the VPAD-pad host assembly remains,
  `// RIG-VERIFY`); `KFMFilterBase.cu` itself is now fully ported.
  **ALG-VERIFIED** via `python/run_kfm_filterbase.py` (3100 cases,
  integer-exact).
- `kfm_noiseclip.cl`: **KFM KNoiseClip** — `kf_noise_clip`, faithful to
  `cpu_noise_clip`/`dev_limitter` (KFM/DecombeUCF.cu, MIT). Self-contained
  **8-bit-only** filter that maps each src pixel against a `noise` pixel into
  band markers `{0,56,128,199,255}`: `out = dev_limitter((src-noise+256)>>1,
  nmin, range)` (128 == equal). Driven by one per-plane kernel, so it is
  kernel-complete (only AviSynth host glue remains). **ALG-VERIFIED** via
  `python/run_kfm_noiseclip.py` (300 cases, integer-exact, nmin/range sweeps).
- `kfm_decombeucf.cl`: **KFM DecombeUCF reductions (7/7 remaining kernels)** —
  `kf_init_uint64`, `kf_calculate_field_diff`, `kf_init_block_sum`,
  `kf_add_block_sum` (BLOCK_SIZE 4/8/16/32), `kf_block_sum_max`,
  `kf_analyze_noise`, `kf_analyze_diff` (KFM/DecombeUCF.cu, MIT): gated 5-tap
  combe sums, per-block sumAbs/sumSig accumulation, and the analyze-noise/
  diff census with uint64 atomics. Together with `kf_noise_clip`, all 8
  DecombeUCF.cu device kernels are ported (only multi-clip host pipelines
  remain). **ALG-VERIFIED** via `python/run_kfm_decombeucf.py` (810 cases,
  integer-exact).
- `kfm_combinganalyze.cl`: **KFM CombingAnalyze, complete (15/15 kernels)** —
  `kf_copy_first`, `kf_combe_to_flag`, `kf_sum_box3x3`, `kf_binary_flag`,
  `kf_bilinear_h/v`, `kf_temporal_soften` (float32, exhaustive t sweep),
  `kf_remove_combe2` (8/16-bit), `kf_clean_super`, `kf_contains_durty_block`
  (+ init), the 8-tap `kf_calc_combe8`/`kf_calc_diff8` helpers, the KFMSuper
  block analyzer `kf_super_analyze` (serial-cell == warp-reduce) and the
  FMCount census `kf_init_fmcount`/`kf_count_cmflags`/`kf_count_cmflags_2planes`
  (group-reduce + atomics; fused == U+V proven). **ALG-VERIFIED** via
  `python/run_kfm_combinganalyze.py` (2731 cases).
- `kfm_deblock.cl`: **KFM KDeblock** — `kf_deblock` (the CUDA `kl_deblock`
  device kernel, KFM/Deblock.cu, MIT): the fixed float32 8×8 DCT →
  hard-threshold (AC coeffs only, DC untouched) → IDCT deblocking stage that
  runs `count = 1<<quality` shifted passes per block and accumulates into a
  16-bit block-parity plane (transcribed from the *device* transform path; the
  upstream CPU fallback's `thresh<=0` identity shortcut is not used). Plus the
  QP-table builder `kf_make_qp_table` (`kl_make_qp_table`, macroblock-QP → block
  QP with optional DC blend + `norm_qscale` 0-3) and the `show==2` visualiser
  `kf_deblock_show` (`kl_deblock_show`, paints enabled/disabled blocks 230/16).
  **ALG-VERIFIED** via `python/run_kfm_deblock.py` (300 cases, core, vs
  `sim/kfm_deblock_ref.cpp` + float32-exact golden) and
  `python/run_kfm_deblock_qp.py` (make_qp_table + deblock_show, 200+200 cases,
  vs `sim/kfm_deblock_qp_ref.cpp`) — bit-exact. Graduated from the rig file:
  the DC-mask dilation `kf_max_vh/v/h`, the ShowQP scaler `kf_scale_qp`, the
  sharpen LUT `kf_sharpen_coeff` and the Bayer accumulator merge
  `kf_merge_deblock` (+`g_ldither`), **ALG-VERIFIED** via
  `python/run_kfm_deblock_aux.py` (1593 cases vs `sim/kfm_deblock_aux_ref.cpp`,
  incl. merge end-to-end + layout proofs and sharpen/show deterministic pins).
  The remaining member — the SharpenFilter pair `kf_sharpen` /
  `kf_show_sharpen_coeff` (manual bilinear replacing the CUDA texture fetch;
  device-run comparison mandatory; deterministic behaviour already
  pinned) — lives separately in `kfm_deblock_rig.cl`
  as faithful `// RIG-VERIFY` transcriptions (bannered PROVISIONAL/UNVERIFIED,
  not covered by `make test`); all 11 Deblock.cu device kernels are now
  transcribed, with the host sequencing and rig proofs remaining open.
  Verification handoff spec for another agent:
  `docs/RIG_HANDOFF_KDEBLOCK.md`. Documented in `docs/KFM_PORT_SPEC.md`.
- `avscuda_merge.cl`: **AvsCUDA Merge/MergeChroma/MergeLuma planar core** —
  `ka_merge` (in-place weighted merge, dual pitch, >>15 SIMD/device scale;
  the scalar-C >>16 fallback is a different weight scale, not a twin),
  `ka_merge_f32`, `ka_average` (exact `average_plane_c` twin) and
  `ka_average_f32` (device `(a+b)*0.5f` vs CPU `(a+b)/2.0f`, proven equal).
  Host dispatch contract (average band / early-outs / YUY2-throws) pinned in
  the file header. **ALG-VERIFIED** via `python/run_avscuda_merge.py`
  (620 cases vs `sim/avscuda_merge_ref.cpp`, incl. band-edge weights).
- `avscuda_filters.cl`: **AvsCUDA Invert** — `ka_invert_plane_u8/u16/f32`
  (word-XOR lane transcription; planar full masks + packed channel masks;
  exact row widths, no word overhang) and `ka_invert_rgb` (RGB24/48
  interleaved per-channel XOR). **ALG-VERIFIED** via
  `python/run_avscuda_filters.py` (750 cases vs
  `sim/avscuda_filters_ref.cpp`). Family map in `docs/AVSCUDA_PORT_SPEC.md`.
- `avscuda_convert.cl`: **AvsCUDA ConvertBits** — ordered-Bayer down-convert
  (`ka_convert_lower_dither_u8/u16`, verbatim tables), truncating down-convert
  (`ka_convert_lower_nodither_u8/u16`), shift up-convert
  (`ka_convert_higher_from_u8/from_u16`), float→int (`ka_convert_from_float_u8/
  u16`, rgy clamp macro spelled out: NaN yields MAX_VAL) and int→float
  (`ka_convert_to_float_from_u8/from_u16`). **ALG-VERIFIED** via
  `python/run_avscuda_convert.py` (1320 cases vs
  `sim/avscuda_convert_ref.cpp`, incl. NaN/Inf edges).
- `avscuda_conditional.cl`: **AvsCUDA Conditional metrics** — counter/hist
  inits, AveragePlane sums, PlaneDifference SAD (u8/u16 x u32/u64) and MinMax
  histograms (u8/u16/f32; NaN index yields 65535). Integer reductions use
  per-item atomics (order-free); u32 wraparound pinned. **ALG-VERIFIED** via
  `python/run_avscuda_conditional.py` (1320 cases vs
  `sim/avscuda_conditional_ref.cpp`). The float sum/SAD reductions are
  faithful `// RIG-VERIFY` transcriptions in `avscuda_conditional_rig.cl`
  (arrival order undefined upstream too; tolerance-based device comparison
  mandatory, not covered by `make test`).
  Verification handoff spec for another agent:
  `docs/RIG_HANDOFF_AVSCUDA_CONDITIONAL.md`.
  OpenCL-side host harness: `src/host/run_avscuda_rig.cpp` (canonical input
  fills + §4.4 launch; build via cmake when OpenCL exists).
- `avscuda_resample.cl`: **AvsCUDA FilteredResizeH/V** — row/unit select
  (`ka_resize_v_pointresize[_f32]`, `ka_resize_h_pointresize_bytes`) and
  separable filters (`ka_resize_v_planar[_f32]`, `ka_resize_h_planar_u8/u16/
  f32`; logical programs, shared-mem/transpose staging elided, unfused f32).
  **ALG-VERIFIED** via `python/run_avscuda_resample.py` (1230 cases vs
  `sim/avscuda_resample_ref.cpp`). AvsCUDA is now fully ported (18/18).
- `nnedi3_pad.cl`: **NNEDI3 batch-1 (pad/copy)** — in-place mirror pads
  `kn_pad_h/v` (group-id idiom, interior origin), plain `kn_copy`, and the
  fused `kn_pad_ref_and_copy_half` (per-vector grid, lane reversal on
  x-mirror). **ALG-VERIFIED** via `python/run_nnedi3_pad.py` (600 cases vs
  `sim/nnedi3_pad_ref.cpp`). Family map in `docs/NNEDI3_PORT_SPEC.md`.

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
make test        # structural .cl lint + CPU-ref + 8/16-bit + motion cross-checks
```

`make test` also runs `make lint`, which does a *genuine* C parse of every
`.cl` (gcc does not parse `.cl` files by themselves — they are copied to `.c`
and parsed through `lint/oc_shim.h`) to catch syntax/typo errors before the
rig's OpenCL compiler is available. This is a structural check only; real
OpenCL semantics are validated on a device.

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
