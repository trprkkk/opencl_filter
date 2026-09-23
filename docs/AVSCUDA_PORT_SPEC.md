# AvsCUDA port spec — `AvsCUDA/` (filters) from CUDA → OpenCL

Upstream: `rigaya/AviSynthCUDAFilters`, directory `AvsCUDA/` (MIT). This family
covers the standalone AviSynth filters (Merge, Invert, Convert, Conditional,
resizer) — no KTGMC/KFM dependencies. Kernel prefix: `ka_`.

## 1. Source inventory (upstream `@68aef6e`, device kernels only)

| Upstream file | Device kernels | Status |
|---|---|---|
| `filters/merge.cu` | `kl_merge_plane`, `kl_average_plane` | ✔ batch-1 (8 scalar kernels, ALG-VERIFIED) |
| `filters/Filters.cu` | `kl_invert_plane`, `kl_invert_rgb` | ✔ batch-1 (same) |
| `filters/Convert.cu` | 5 (`lower_dither`, `lower_no_dither`, `higher`, `from_float`, `to_float`) | ✔ batch-2 (10 scalar kernels, ALG-VERIFIED) |
| `filters/ConditionalFunctions.cu` | 5 (`init_sum`/`sum`/`sad`/`init_hist`/`count_hist`) | ✔ batch-3 (15 kernels ALG-VERIFIED + 2 float // RIG-VERIFY) |
| `filters/Resample.cu` (~2833 lines) | 4 (shared-mem coeff resamplers) | roadmap §4 |

Total: 18 device-kernel templates. Excluded by census: `common/Copy.cu` and
`memcpy_kernel` (host-plumbing memcpys → host API, no port), `kl_draw_text`
(debug OSD, skipped), dead code (see §5).

## 2. Batch-1: merge + invert (`avscuda_merge.cl`, `avscuda_filters.cl`)

Upstream launches over uchar4/ushort4/float4/int/int2 vectors; every lane is an
independent per-pixel op, so the scalar ports are lane-identical. All six
templates are IN-PLACE on the destination plane (dual pitch for merge/average),
transcribed faithfully; the harness verifies values (in-place and out-of-place
are indistinguishable for elementwise ops).

- `ka_merge` (PX uchar/ushort) — `(a*iw + b*w + 16384) >> 15`, C-narrowing
  cast (mod 2^bits, no saturation — `VHelper::cast_to` is a plain cast).
  This is the >>15 SIMD/device scale with `w = (int)(weight*32767+0.5)`,
  `iw = 32767-w`. The scalar-C fallback (`weighted_merge_planar_c`) uses
  `(+32768)>>16` with a *65535-based weight — a different scale, NOT a twin
  (the SSE2/AVX2 fallbacks use this >>15 formula). Valid weights can never
  overflow int32 (`a*iw+b*w <= 65535*32767`).
- `ka_merge_f32` — `a*iwf + b*wf`, unfused; `weighted_merge_planar_c_float`
  is the exact CPU twin (same op order). Host build must disable FP
  contraction for bit-exactness.
- `ka_average` (PX) — `(a+b+1)>>1`; `average_plane_c` is the exact CPU twin.
- `ka_average_f32` — device form `(a+b)*0.5f`; the CPU twin uses `(a+b)/2.0f`
  (provably identical — both correctly-rounded exact halvings; the harness
  runs one form in the mirror, the other in the golden).
- `ka_invert_plane_u8` — byte `(x&3)` of mask0 XORed per pixel; planar path
  passes `0xFFFFFFFF`, packed YUY2/RGB32 paths pass channel masks.
- `ka_invert_plane_u16` — 16-bit lane of the `(mask0,mask1)` sequence;
  planar masks are uniform (all four parts equal), RGB64 packs channels.
- `ka_invert_plane_f32` — `1.0f - x`, masks ignored (launch passes 0,0). The
  C twin's chroma variant (`max = 0`) is under
  `FLOAT_CHROMA_IS_ZERO_CENTERED`, never defined upstream — exact twin.
- `ka_invert_rgb` (PX-generic) — interleaved `ptr[3*x+c] ^= mask_c`, narrowed
  on store; production masks are 0/all-ones per channel.

Host dispatch contract (`merge_plane`, `merge.cu`): weight in (0.4961, 0.5039)
→ average kernel; weight < 0.0039 → src untouched; weight > 0.9961 → plain
copy; else weighted merge. YUY2 has no CUDA path (throws) — host/SIMD only.

Word-overhang note: upstream rounds rows up to whole words, writing up to
3 (u8) / 7 (u16) bytes past rowsize into the pitch padding. The scalar ports
take width = exact elements and write exactly the row — the only divergence,
and only in the overhang bytes (harmless with SIMD-aligned pitches).

Batch-2 (`avscuda_convert.cl`, 10 kernels — the five templates split by
element width per the BitsToType rule) adds the ConvertBits core: ordered
Bayer down-convert (`ka_convert_lower_dither_u8/u16`, verbatim c_dither2/4/6/8
tables, shifts always even in {2,4,6,8}), truncating down-convert
(`ka_convert_lower_nodither_u8/u16` — plain `>>SHIFT`; the `+HALF` rounding
line is commented out upstream, transcribed as-is), shift up-convert
(`ka_convert_higher_from_u8/from_u16`), float→int (`ka_convert_from_float_u8/
u16`, chroma adds HALF; the rgy clamp macro is spelled out so NaN yields
MAX_VAL like upstream — NOT OpenCL's builtin clamp, which yields 0; 16-bit
MAX_VAL is 65280) and int→float (`ka_convert_to_float_from_u8/from_u16`,
`FACTOR = 1.0f/MAX_VAL`, chroma subtracts HALF first; unfused float32).

Batch-3 (`avscuda_conditional.cl`, 15 kernels) covers the Conditional runtime
metrics: counter/histogram zeroing (`ka_init_sum_u32/u64/f32`, `ka_init_hist`),
AveragePlane sums (`ka_sum_pixels_u8/u16` x u32/u64, with the min(v,maxv)
content clamp for u16), PlaneDifference SAD (`ka_sad_u8/u16` x u32/u64, single
pitch) and MinMaxPlane histograms (`ka_count_hist_u8/u16/f32` over int
counters). Integer reductions use one atomic per work-item (arrival-order-free;
u32 wraparound pinned with oversized frames; u64 needs OpenCL 2.0 or
cl_khr_int64_base_atomics). The float hist index spells out max(0,min(t,65535))
so NaN yields 65535 like CUDA. Width-mult-4, the 32/64-bit host threshold and
the average/SAD host divisions are pinned as contracts. The two FLOAT sum/SAD
reductions are faithful // RIG-VERIFY transcriptions in
`avscuda_conditional_rig.cl` (portable __local tree + CAS-loop float atomic;
cross-block arrival order is undefined upstream too, so only a
tolerance-based device comparison can verify them -- no mirror, no pin).

**ALG-VERIFIED** via `python/run_avscuda_merge.py` (620 cases:
`sim/avscuda_merge_ref.cpp` vs independent golden; dual-pitch, host-formula
weights incl. band edges 0.4961/0.5039 and 0/1, crafted value edges, float
edges incl. ±0/subnormals) and `python/run_avscuda_filters.py` (750 cases:
channel/uniform/random masks, non-mult-4 widths, RGB full+partial masks).

## 3. Kernel files

- `src/opencl/avscuda/kernels/avscuda_merge.cl` — the 4 merge/average kernels.
- `src/opencl/avscuda/kernels/avscuda_filters.cl` — the 4 invert kernels.
- `src/opencl/avscuda/kernels/avscuda_convert.cl` — the 10 convert kernels.
- `src/opencl/avscuda/kernels/avscuda_conditional.cl` — the 15 conditional
  kernels; `avscuda_conditional_rig.cl` — the 2 float // RIG-VERIFY kernels.
- `sim/avscuda_merge_ref.cpp`, `sim/avscuda_filters_ref.cpp`,
  `sim/avscuda_convert_ref.cpp` — CPU mirrors (int-token protocol;
  float as bit-pattern ints).
- `python/run_avscuda_merge.py` (620 cases), `python/run_avscuda_filters.py`
  (750), `python/run_avscuda_convert.py` (1320: all producible (shift,bits)
  pairs, NaN/Inf float edges) and `python/run_avscuda_conditional.py`
  (1320: u32-wrap oversized frames, clamped content, NaN hist indices) —
  goldens, wired into `make test`. `sim/avscuda_conditional_ref.cpp` is the
  15-mode mirror (int-token protocol; u64 sums print as one decimal).

## 4. Roadmap (remaining 4 templates)

- **Resample (4)**: the ~2833-line resizer with shared-memory coefficient
  staging. Largest batch; shared-mem → `__local` transcription with the
  rig-bound layout proofs, like the KDeblock texture/sharpen work.

## 5. Excluded (proven by census caller greps @`68aef6e`)

- `common/Copy.cu` dual-pitch `kl_copy` (self-wrapped host plumbing),
  `memcpy_kernel` (MV host plumbing) → host API, no kernel port.
- `kl_draw_text` (debug OSD via TextOut) — skipped.
- Dead: anything with no live caller (same bar as the KFM/KTGMC census:
  `#if 0`-gated, commented-out, or never instantiated).
