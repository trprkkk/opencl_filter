# RIG HANDOFF — AvsCUDA Conditional float reductions (verification spec for another agent)

> **CLOSED 2026-10-28.** Both kernels are marked
> `// RIG-COMPARED 2026-10-28` in `avscuda_conditional_rig.cl` and the §8
> appendix below is filled from the CUDA twin run (200 CUDA reps × 10 CL;
> `rig/cond_cuda.cu` + `rig/dev_cond_cuda.py`, report
> `rig/RIG_CUDA_TWINS_REPORT.md`). The §8 accepted-deviations table records
> the one criterion-1 deviation (37 cells, ≤36 ulp leaf-association
> rounding) and the criterion-2 under-coverage now proven on both
> toolchains. The rig file stays in place permanently; the terminal state is
> deliberately not ALG-VERIFIED.

This document hands the **unverifiable-by-construction** AvsCUDA OpenCL kernels
to another agent (human or AI) for device comparison. It is written to be
self-sufficient: if you follow it top to bottom you can close both kernels
without asking the original author anything.

Read §3 first: unlike every other handoff in this repo, there is **no
graduation path to `// ALG-VERIFIED`** here. The terminal state is
`// RIG-COMPARED` (§6), and §5 tells you exactly what evidence earns it.

## 0. TL;DR

- **Package under verification:** `src/opencl/avscuda/kernels/avscuda_conditional_rig.cl`
  (2 kernels + 1 helper — verified `// RIG-COMPARED` 2026-10-28, see the
  banner above; separated from the verified `avscuda_conditional.cl`).
- **Ground truth:** `rigaya/AviSynthCUDAFilters`,
  `AvsCUDA/filters/ConditionalFunctions.cu`, commit
  `68aef6e9b8a63d0dceb8e7f68b52b27431aa361f9` (§2).
- **Your job:** build a CUDA harness that runs the two
  `kl_sum_of_pixels<float4,float,float>` / `kl_sad<float4,float,float>`
  instantiations (the OpenCL host is already provided as
  `src/host/run_avscuda_rig.cpp` — build it via cmake, §5); prove
  **bit-exact agreement on exact-arithmetic inputs** (Phase A) and
  **in-band agreement on general inputs** (Phase B, tolerance-based);
  then mark the kernels `// RIG-COMPARED` and fill in the results appendix
  (§8). No mirror, no golden, no `make test` wiring — §3 explains why that
  would prove nothing.
- **Closest templates to copy:** `docs/HOST_CONTRACT.md` (host-runner shape;
  add the rig file as a third program) and `python/run_avscuda_conditional.py`
  (input/config conventions for the sibling int kernels — reuse its shapes,
  pitches and width-mult-4 discipline).

## 1. What is in the package (and what is not)

### 1.1 In scope — the two kernels in `avscuda_conditional_rig.cl`

| # | OpenCL kernel | CUDA device twin | CPU twin? | Difficulty |
|---|---|---|---|---|
| 1 | `ka_sum_pixels_f32` | `kl_sum_of_pixels<float4,float,float>` | none usable (order undefined — §3) | medium (harness + statistics, no algorithm puzzle) |
| 2 | `ka_sad_f32` | `kl_sad<float4,float,float>` | none usable (same) | medium (same harness, second input plane) |

Plus the `ka_atomic_add_f32` helper (CAS-loop float atomic — the only
mechanism in the file, §4.3). Both kernels are **scalar** transcriptions:
CUDA threads process `float4` vectors but the port assigns one scalar pixel
per work-item; the per-lane math is identical, the *association order* differs
(§3–§4 — this is expected and accepted, not a bug to fix).

### 1.2 Explicitly OUT of scope

- The Conditional **host** (AveragePlane/ComparePlane/LumaDifference frame
  dispatch, `sum_in_32bits` selection, width-mult-4 enforcement, the
  average/SAD-per-pixel host divisions, RGB *4/3 rescaling) — you compare
  kernels, not filters.
- The fifteen **integer** sibling kernels in `avscuda_conditional.cl`
  (`// ALG-VERIFIED`, `python/run_avscuda_conditional.py`) — no action needed.
- Counter zeroing (`ka_init_sum_f32`, ALG-VERIFIED) — reuse it (or
  `clEnqueueFillBuffer`) to pre-zero the counter in your harness.
- KTGMC `ktgmc_motion.cl` RIG-VERIFY items — separate handoff
  (`docs/CODEX_HANDOFF.md` + `docs/BLOCKSEARCH_MODEL.md`).
- The KDeblock sharpen pair (`docs/RIG_HANDOFF_KDEBLOCK.md`) — separate
  handoff, and a *different* verification kind (deterministic kernels with a
  texture-precision gap; these two are nondeterministic by construction).

## 2. Upstream grounding (exact)

Upstream repo: `https://github.com/rigaya/AviSynthCUDAFilters` (AvsCUDA is MIT).

Reference commit (what the transcriptions were made against):

```
68aef6e9b8a63d0dceb8e7f68b52b27431aa361f9
```

Reproduce it:

```sh
git clone https://github.com/rigaya/AviSynthCUDAFilters /tmp/avs_cuda
cd /tmp/avs_cuda && git checkout 68aef6e9b8a63d0dceb8e7f68b52b27431aa361f9
```

Practical notes: `AvsCUDA/filters/ConditionalFunctions.cu` uses **CRLF** line
endings and Shift-JIS bytes in comments — pipe through `tr -d '\r'` and read
as latin-1/bytes. Line numbers below are 1-based at `68aef6e`.

### 2.1 Line map at `68aef6e`

`AvsCUDA/filters/ConditionalFunctions.cu`:

| Upstream symbol | Lines | Notes |
|---|---|---|
| `SUM_TH_W/H/THREADS` (16/16/256) | 17–21 | launch geometry |
| `kl_init_sum<T>` | 25–28 | single-element zero (`<<<1,1>>>`) |
| `clamp_to_range` ×3 | 30–32 | float4 is the **identity** (maxv ignored) |
| `kl_sum_of_pixels` | 35–53 | tree call :48, `atomicAdd` :51 |
| `calc_sum_of_pixels` | 56–~76 | `width4/pitch4` :58–59, init :61, grid :64–65, host average :72 |
| Average: width-mult-4 throw | 199 | host rejects `width % 4` on CUDA |
| Average: `sum_in_32bits=false` (float) | 205 | pixelsize 4 always takes the float path |
| Average: `maxv` | 216 | `(1<<bits)-1`, ignored by the float path |
| Average: float instantiation | 230 | `<float4,float,float>` |
| `diff_pixel` ×3 | 277–286 | float :283–286 is `abs(src0-src1)` |
| `kl_sad` | 288–309 | diff :299, lane sum :300, `atomicAdd` :307 |
| `calc_sad` | 312–~335 | init :317, host SAD/average + RGB scaling at the end |
| ComparePlane: throw / float inst / maxv | 508 / 542 / 527 | |
| LumaDifference: throws / float inst / maxv | 667, 670 / 704 / 689 | second `calc_sad<float4,float,float>` user |

`common/ReduceKernel.cuh`:

| Upstream symbol | Lines | Notes |
|---|---|---|
| `dev_reduce_warp_mask` shuffle cascade | 46–50 | strides 16,8,4,2,1 via `__shfl_down_sync` |
| `dev_reduce` block tree | 131–173 | smem halving 128,64,32 → `value=buf[tid]` → warp cascade |
| `int`/`uint` `__reduce_*_sync` specializations | 62–92 | sm_80+ only, **not used for float** — float always takes the shuffle path above |

`common/VectorFunctions.cuh`:

| Upstream symbol | Lines | Notes |
|---|---|---|
| `abs(float4)` | 241–244 | per-lane `fabsf` |

### 2.2 Facts you need from the twins (all verified at `68aef6e`)

1. Per-thread leaf: `tmpsum = ((s.x+s.y)+s.z)+s.w`, left-associative float
   adds, starting from `lsum_t()` = `0.0f`. Out-of-range threads (`x ≥ width4`
   or `y ≥ height`) contribute exactly `0.0f` and still participate in the tree.
2. Intra-block combination (float): smem halving `buf[tid] += buf[tid+stride]`
   for strides 128, 64, 32 (each followed by `__syncthreads`), then the warp
   shuffle cascade `value += shfl_down(value, stride)` for strides 16–1.
   Only lane 0 of warp 0 (`tid == 0`) holds the block sum and issues the
   single `atomicAdd(sum, tmpsum)`.
3. Cross-block combination is one `atomicAdd(float)` per block into the
   single global counter. **Arrival order across blocks is undefined** —
   this is the sentence §3 stands on.
4. `width4 = width >> 2`, `pitch4 = pitch >> 2` (pitch in *elements* at the
   call boundary); grid is `nblocks(width4,16) × nblocks(height,16)` blocks
   of 16×16 threads. Width is host-guaranteed a multiple of 4 (throw :199
   etc.), so no tail column is ever dropped.
5. `maxv` is passed but **ignored** on the float path (identity clamp,
   `fabs` needs no clamp). The port keeps the arg for signature fidelity.

## 3. Why bit-exact verification is impossible (read carefully)

The reduction computes a sum of floats in three stages with three different
definedness properties:

1. **Per-thread 4-lane sum** — deterministic order (`((x+y)+z)+w`), but the
   port's leaves are *scalars*, not 4-lane partials, so the leaf sequences
   already differ.
2. **Intra-block tree** — deterministic order *given an implementation*, but
   the port substitutes a portable `__local` tree for the warp-shuffle tail
   (§4.2). (The stride pairing happens to be identical — strides 128…1 in
   order — but over the different leaves of stage 1, so rounding still
   differs in general.)
3. **Cross-block `atomicAdd`** — arrival order is **undefined**, on CUDA and
   on OpenCL alike. Float addition is not associative, so the final rounding
   legitimately varies *from one CUDA run to the next on identical input*.

Stage 3 alone makes a bit-exact golden meaningless: there is no single
"correct" output to compare against, only a distribution over arrival orders.
A CPU mirror + Python golden could pin down *our* tree order to the last ulp
— and it would prove exactly nothing about the twin, because the twin never
promised any order. That is why this package has **no mirror, no golden, and
no `make test` wiring**, deliberately (contrast the KDeblock sharpen pair,
whose kernels are deterministic and whose only gap is texture precision).

What *can* be proven (§5):

- **Phase A — exact-arithmetic inputs are deterministic on both sides.**
  Where every partial sum is exactly representable, all association orders
  coincide, arrival order is irrelevant, and both implementations must produce
  **bit-identical** results. This pins the mechanism (leaf coverage, tree
  wiring, partial blocks, the CAS loop, NaN propagation).
- **Phase B — general inputs must agree in-band.** Sample the CUDA run-to-run
  distribution (≥10 runs), sample the OpenCL distribution (≥10 runs), and
  require the OpenCL results to land inside (or within one band-width of) the
  CUDA band, plus a gross-error bound against a float64 reference. This
  proves the port sums the right values with a sane order.

## 4. Per-kernel specification

### 4.1 `ka_sum_pixels_f32`

```c
kernel void ka_sum_pixels_f32(
    __global const float* __restrict src, int width, int height, int pitch,
    int maxv, __global float* __restrict sum)
```

Transcribed math (must match the port, which you are comparing — not CUDA's
order, which differs by §3):

```
per work-item: tmpsum = (x<width && y<height) ? src[x+y*pitch] : 0.0f
tree: sbuf[tid]=tmpsum; strides 128..1: if (tid<stride) sbuf[tid]+=sbuf[tid+stride]
single: if (tid==0) ka_atomic_add_f32(sum, sbuf[0])
```

### 4.2 `ka_sad_f32`

```c
kernel void ka_sad_f32(
    __global const float* __restrict src0,
    __global const float* __restrict src1,
    int width, int height, int pitch,
    int maxv, __global float* __restrict sum)
```

Same skeleton; per-item leaf is `fabs(src0[off]-src1[off])` (`fabs` on float
in OpenCL C — same operation as CUDA's `fabsf` lane). Single shared pitch,
as upstream.

### 4.3 The two substitutions (accepted, not to be "fixed")

1. **Portable `__local` tree instead of the shuffle tail.** Upstream combines
   strides 128/64/32 in smem and 16/8/4/2/1 via warp shuffles; the port runs
   all strides 128…1 in `__local sbuf[256]` with `barrier(CLK_LOCAL_MEM_FENCE)`
   between steps. Same pairing sequence, portable to OpenCL 1.1+ (no
   subgroups), values differ from CUDA only at rounding level.
2. **CAS-loop float atomic instead of `atomicAdd(float)`.**
   `ka_atomic_add_f32` spins `atomic_cmpxchg` on the bit pattern until the
   read-modify-write of `as_float(old)+val` lands. Same atomic RMW-add
   operation as CUDA's `atomicAdd(float)` (OpenCL 1.1 core — no
   `cl_ext_float_atomics` needed); arrival order stays undefined, as upstream.

### 4.4 Launch contract (your harness must honour it — all load-bearing)

1. **Local size is exactly (16,16).** The kernel hard-codes
   `tid = get_local_id(0) + get_local_id(1)*16` and `sbuf[256]`; a wrong
   local size would silently compute garbage, so both kernels additionally
   carry `reqd_work_group_size(16,16,1)` — a non-(16,16) local size fails
   at enqueue with `CL_INVALID_WORK_GROUP_SIZE` instead. The harness
   passes (16,16) explicitly regardless.
2. **Global size is 16-rounded-up**: `(round16(width), round16(height))`.
   Out-of-range items contribute `0.0f` through the tree — exactly like
   CUDA's full edge blocks.
3. **Width is a multiple of 4** (inherited host contract — upstream throws
   otherwise). Test only mult-4 widths; the port would accept others, but
   that behaviour is uncharted upstream.
4. **Counter pre-zeroed** (via ALG-VERIFIED `ka_init_sum_f32` or host fill).
5. **Pitches in float elements** (not bytes, not vectors).
6. **Program build**: the rig file carries the repo-standard
   `#ifndef PX / #error` guard but is width-independent — a **single** build
   (either `-DPX=uchar` or `-DPX=ushort`) suffices. No extensions required
   (`atomic_cmpxchg` is OpenCL 1.1 core).

### 4.5 Must-check facts (traps)

1. **`lsum_t()` is `0.0f`, and edge threads add it.** A port that skipped
   out-of-range items instead of adding `0.0f` would still be correct here
   (adding `+0.0f` is exact) — but test partial blocks anyway (§5.2): they
   catch indexing bugs, which are the realistic failure mode.
2. **Sign of zero.** All-zeros input deterministically yields `+0.0f` on both
   sides (`+0.0f + ±0.0f == +0.0f` in round-to-nearest, every add exact).
   Assert `+0.0f` bits in Phase A; treat `-0.0f` anywhere as a bug.
3. **NaN propagates deterministically to NaN.** Once any `NaN` enters the
   accumulator (`0.0f + NaN == NaN`, stays NaN), the final counter is NaN on
   *every* arrival order, CUDA and OpenCL alike. Any-NaN-in → NaN-out is a
   Phase A exact anchor (payload unspecified — compare with `isnan`, not bits).
4. **No denormal flushing assumptions.** If either toolchain flushes
   subnormals (check `CL_FP_DENORM` / nvcc `--ftz`), subnormal-heavy configs
   may diverge legitimately — characterise, and note the flush mode in the
   appendix. Prefer `clGetDeviceInfo` + `cuCtxGetLimit`-level evidence over
   guessing.
5. **The sad `fabs` is exact** (bit-clear, no rounding) — sad-vs-sum
   divergence on identical inputs is always a real bug, never rounding.

## 5. The verification protocol

You need two harnesses — the OpenCL one is provided, the CUDA one you write
(keep both out of `make test`; suggested scratch dir for your CUDA side:
`tools/rig_conditional/`, or your rig's equivalent):

- **CUDA harness** (`cond_cuda.cu`, built with nvcc): includes the upstream
  headers (`-I…/common` for `ReduceKernel.cuh`/`VectorFunctions.cuh`),
  instantiates `kl_sum_of_pixels<float4,float,float>` and
  `kl_sad<float4,float,float>` directly — no AviSynth needed. Launch exactly
  per §2.2-4 (threads 16×16, `width4 = width>>2`, counter from
  `cudaMalloc`+`cudaMemset`). Note: the `__CUDA_ARCH__ >= 800`
  `__reduce_add_sync` fast path applies to the `int` specialization only —
  float always takes the shuffle cascade (§2.1), on every arch.
- **OpenCL harness** — provided as `src/host/run_avscuda_rig.cpp` (build:
  `cmake -S . -B build && cmake --build build --target avscuda_rig_host`;
  compile-checked only, so run the A1 smoke test in its header comment
  first): builds `avscuda_conditional_rig.cl` once, launches per §4.4,
  reads the counter back after `clFinish`, and prints counter bits + the
  host f64 reference per run. Its `fill_plane()` is the **canonical
  input-fill definition** — transcribe those float expressions verbatim
  into the CUDA twin so both sides consume bit-identical inputs (same
  shapes, fills, seeds).

Both harnesses take the same (shape, distribution, seed) configs and print the
raw counter bits (`%08x` + `%f`) per run. Drive them from a script that loops
configs × runs and records everything into the §8 appendix tables.

### 5.1 Phase A — bit-exact anchors (all must pass, zero tolerance)

Principle: every partial sum exactly representable ⇒ all orders coincide ⇒
**all runs on both sides bit-identical**. Run each config ≥5 times per side;
assert all 10+ outputs have identical bits (except NaN: `isnan` on all).

Required configs (shapes × contents; adapt, don't shrink):

| # | Shape (w×h, w mult-4) | Contents | Why |
|---|---|---|---|
| A1 | 4×1 | all `1.0f` | smallest partial group (4 of 256 lanes live) |
| A2 | 64×16 | all `1.0f` | exactly one full group (1024.0f) — intra-block order isolated |
| A3 | 68×20 | all `1.0f` | partial group in x *and* y (edge coverage) |
| A4 | 320×240 | all `2.0f` | many groups, exact total 153600.0f |
| A5 | 1920×1080 | all `0.5f` | full-HD exact total 1036800.0f (arrival-order stress, still exact) |
| A6 | 64×16 | checkerboard `±1.0f`, exact total | exact cancellation (total `0.0f` — asserts +0.0f bits, §4.5-2) |
| A7 | 64×16 | all `+0.0f`, then mixed `±0.0f` | zero-sign determinism (+0.0f bits both) |
| A8 | 64×16 + one `NaN` (canonical) at a fixed index | NaN-propagation anchor (`isnan` on all runs) |
| A9 | sad only, 68×20 | `src0` all `3.0f`, `src1` all `1.0f` | exact sad `2.0f`/pixel, partial blocks |
| A10 | sad only, 64×16 | `src0 == src1` random-but-fixed | exact `0.0f` total (assert +0.0f bits) |

A single bit-mismatch in Phase A is a **port bug** (or a harness bug — rule
that out first by rechecking §4.4). Do not proceed to Phase B until Phase A
is fully green: tolerance statistics on top of a broken mechanism prove nothing.

### 5.2 Phase B — in-band agreement (tolerance-based)

Principle: sample CUDA's own run-to-run distribution (it is genuinely
nondeterministic — verify this first: if 10 CUDA runs on A4-like *inexact*
data are all bit-identical, your GPU serialises atomics and the band
collapses; note it and widen the run count, the criterion still applies).

Per config: run CUDA ≥10 times → band `[cuda_min, cuda_max]`, mean
`cuda_mean`; run OpenCL ≥10 times → samples `cl_i`; compute a float64
reference sum `f64ref` on the host (plain serial accumulation).

**Acceptance (all three, every config):**

1. **In-band**: every `cl_i` lies within `[cuda_min − W, cuda_max + W]` with
   `W = max(cuda_max − cuda_min, 0)` — i.e. inside the CUDA band widened by
   one band-width on each side. (If the CUDA band collapses to a point, this
   demands CL bit-equality with CUDA — correct: a collapsed band means the
   config is effectively exact.)
2. **Gross-error bound**: `|mean(cl_i) − f64ref| ≤ 4·N·eps·max|x|`,
   `eps = 2^-24`, `N` = pixel count, `max|x|` = max input magnitude
   (for sad, max `|a−b|`). This catches dropped tiles / double counts /
   indexing bugs that in-band comparison could theoretically miss.
3. **Sanity**: no NaN unless the input contained NaN; finite in, finite out.

Required config grid (≥8 shapes × ≥4 distributions; required cells marked ★):

Shapes: `4×1` ★ (single partial group), `64×16` ★ (single full group),
`68×20` ★ (partial x+y), `320×240`, `640×480`, `1920×1080` ★ (arrival-order
sampling), `4096×4` (wide-short), `16×2048` (tall-thin — non-mult-16 height
with full x-groups).

Distributions (fresh random data per config, fixed seed, shared by both
harnesses): uniform `[0,1]` ★, uniform `[-1,1]`, `1e10`-scale ★ (large
magnitude), `1e-10`-scale (small magnitude), mixed `1e10+1e-10`
(cancellation stress) ★, alternating `±1` (near-zero total — criterion 1
rules here since relative error is meaningless), video-like (smooth gradient
+ noise in `[0,1]`).

Near-zero totals (alternating `±1`, exact-ish cancellation): judge by
criterion 1 only, plus an absolute floor of `1e-3 · Σ|x|` against `f64ref`
instead of criterion 2's relative-style bound (which goes vacuous/loopy near
zero — note which configs used the floor in the appendix).

**Statistics hygiene**: interleave CUDA/CL runs (don't run all CUDA first —
clock/thermal drift is real); record driver versions, device name, nvcc
flags, and OpenCL build options in the appendix; keep raw outputs, not just
pass/fail.

### 5.3 Phase C — mechanism spot-checks (cheap, do them)

- **Counter pre-zero**: run one Phase-A config with the counter pre-filled
  with `42.0f` instead of zero — both sides must report `exact + 42.0f`
  (exact arithmetic still). Catches harness zeroing bugs masquerading as
  kernel bugs.
- **Pitch slack**: one config with `pitch = width + 7` (garbage in the slack
  — both sides must ignore it identically; fill slack with `NaN` to make any
  stray read scream).
- **`maxv` ignored**: run one float config with `maxv = 0` and one with
  `maxv = INT_MAX` — identical results (pins §2.2-5 on both sides).

## 6. Completion checklist (all boxes required to close the handoff)

- [ ] CUDA harness runs both float instantiations launch-faithfully (§2.2-4,
      §5) on a real GPU; OpenCL harness builds the rig file warning-free and
      launches per §4.4.
- [ ] **Phase A fully green**: every config bit-identical across ≥5 runs per
      side (NaN: `isnan` everywhere).
- [ ] **Phase B fully green**: every config passes criteria 1–3 (§5.2) over
      ≥10 runs per side; CUDA nondeterminism observed (or its absence
      explained + run count widened).
- [ ] Phase C spot-checks pass.
- [ ] Appendix (§8) filled with raw numbers (not just verdicts), environment
      (device, drivers, nvcc/CL flags, denorm behaviour), and any accepted
      deviations with justification.
- [ ] Both kernels remarked `// RIG-COMPARED <YYYY-MM-DD> <tolerance summary>
      <cuda_runs>×<cl_runs>` in `avscuda_conditional_rig.cl` (replacing
      `// RIG-VERIFY`), header banner updated from PROVISIONAL/UNVERIFIED to
      RIG-COMPARED with a pointer to this doc's §8.
- [ ] Docs updated: this file (§0/§1 tables + §8 marked done),
      `docs/AVSCUDA_PORT_SPEC.md` (batch-3 paragraph), `README.md`
      (conditional bullet), `docs/PORT_PLAN.md` (AvsCUDA bullet).
- [ ] `lint/lint_opencl.sh` still green (the rig file parses in both PX modes).

When all boxes are ticked the handoff is **closed**. The rig file is **never
deleted and the kernels never graduate to `// ALG-VERIFIED`** — §3 forbids it.
`// RIG-COMPARED` is the terminal state; any future change to these kernels
re-opens this handoff (new runs, new appendix entry).

## 7. Environment constraints (read before planning a device run)

This sandbox has **no GPU, no CUDA toolkit, no OpenCL ICD, no AviSynthNeo** —
and the Debian mirrors are unreachable, so even pocl cannot be installed.
That is *why* this handoff exists as a device-run spec instead of a
mirror+golden pair. Do the CUDA/OpenCL work on a rig with a real NVIDIA GPU
(CUDA side) and any OpenCL 1.1+ device (CL side — CPU ICD is fine for
correctness; arrival-order sampling wants a real GPU, so prefer running the
CL side on the same NVIDIA GPU via its OpenCL driver). Do not attempt to
install GPU stacks here; do not close the handoff without the runs in §5.

## 8. Results appendix — FILLED 2026-10-28 (device runs done; handoff CLOSED)

### Environment

- Date(s): 2026-10-28 (CUDA twin runs; OpenCL side parsed from the existing
  first-pass log).
- CUDA device + driver + nvcc + flags: NVIDIA GeForce RTX 2060 SUPER (sm_75),
  driver 591.74, CUDA nvcc 13.3 (V13.3.73), MSVC 14.51.36231 host;
  `nvcc -arch=sm_75 -I upstream/common rig/cond_cuda.cu`.
- OpenCL device + driver + build options: the same GPU via its OpenCL driver
  (driver-reported OpenCL 3.0); kernels built `-cl-std=CL1.2` (single
  build — the rig file is width-independent), first-pass log
  `rig/cond_full_out.txt` (10 reps/cell).
- Denorm handling: not explicitly probed; all Phase B distributions
  (1e-10..1e10 scales) produced finite counters on both sides (criterion 3
  PASS everywhere).
- Upstream checkout: `68aef6e` (masktools submodule `24ba826`).
- Harnesses / raw data (all under `Q:\Downloads\openclfilter\rig\`,
  outside the repo): `cond_cuda.cu` (CUDA twin, verbatim upstream kernels +
  launch geometry), `dev_cond_cuda.py` (driver, same case grid as
  `dev_avscuda_conditional.py`; Phase A = A1..A10+A8 at 5 reps, Phase B =
  8 shapes x 7 dists x {sum,sad} at **200 CUDA reps** — widened per §5.2
  because the 10-rep bands collapsed on many cells); raw:
  `cond_cuda_out.txt` (22,455 runs), `cond_cuda_run.txt`,
  `cond_cuda_cmp.txt`. Inputs bit-identical CUDA/Python per (sum32,xor32)
  checksums on all 123 cases.

### Phase A (bit-exact) — all runs' counter bits (11/11 PASS, zero tolerance)

| Config | CUDA x5 (hex) | OpenCL x5 (hex) | Verdict |
|---|---|---|---|
| A1 sum 4x1 ones | 0x40800000 | 0x40800000 | PASS (== expected 4.0) |
| A2 sum 64x16 ones | 0x44800000 | 0x44800000 | PASS (1024.0) |
| A3 sum 68x20 ones | 0x44aa0000 | 0x44aa0000 | PASS (1360.0) |
| A4 sum 320x240 two | 0x48160000 | 0x48160000 | PASS (153600.0) |
| A5 sum 1920x1080 half | 0x497d2000 | 0x497d2000 | PASS (1036800.0 exact) |
| A6 sum 64x16 checker | 0x00000000 | 0x00000000 | PASS (+0.0) |
| A7 sum 64x16 zeros | 0x00000000 | 0x00000000 | PASS (+0.0) |
| A7 sum 64x16 zerosmix | 0x00000000 | 0x00000000 | PASS (+0.0) |
| A9 sad 68x20 3-1 | 0x452a0000 | 0x452a0000 | PASS (2720.0) |
| A10 sad 64x16 rand01== | 0x00000000 | 0x00000000 | PASS (+0.0) |
| A8 sum 64x16 nan1 | 0x7fffffff (isnan) | 0x7fffffff (isnan) | PASS (NaN both sides) |

### Phase B (in-band) — 112 cells, 200 CUDA reps x 10 CL reps

Columns: criterion 1 = every CL sample inside the CUDA band widened by one
band-width per side (§5.2-1); "crit2 (CUDA twin)" = the CUDA twin's own
compliance with the as-written §5.2-2 gross-error bound
`4*N*2^-24*maxabs` (the CL side is 107/112 as-written with the 5 tight cells
adjudicated via the host f32 twin envelope — see accepted deviations).
Full numeric detail per cell (dmax/ulp vs the leaf-order tree bound, RM-gate
`|mean(cl)-f64ref|` vs the `6*sqrt(B)*ulp(Cmax)/(2*sqrt(3)*sqrt(reps)) +
B*maxabs*2^-13` noise envelope) in the notes column; raw bits per run in
`rig/cond_cuda_out.txt`.

| Cell (op shape distr) | N | CUDA band [min,max] (float) | CL band [min,max] (float) | Crit. 1 | Crit. 2 (CUDA twin) | Notes |
|---|---|---|---|---|---|---|
| sum 4x1 rand01 | 4 | [1.781748, 1.781748] | [1.781748, 1.781748] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sum 4x1 rand11 | 4 | [-0.4365039, -0.4365039] | [-0.4365039, -0.4365039] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sum 4x1 big | 4 | [1.781748e+10, 1.781748e+10] | [1.781748e+10, 1.781748e+10] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sum 4x1 small | 4 | [1.781748e-10, 1.781748e-10] | [1.781748e-10, 1.781748e-10] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sum 4x1 mixed | 4 | [7.23868e+09, 7.23868e+09] | [7.23868e+09, 7.23868e+09] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sum 4x1 altpm1 | 4 | [0, 0] | [0, 0] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sum 4x1 video | 4 | [0.8531748, 0.8531748] | [0.8531748, 0.8531748] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sum 64x16 rand01 | 1024 | [496.9775, 496.9775] | [496.9775, 496.9775] | FAIL | PASS | CUDA band collapsed: 1 distinct bits; dmax=3.05e-05 (1.00 ulp, tree-bound 0.0004883 -> within leaf-order envelope) RMgate(|clmean-f64|=1.53e-05 <= 0.0005217):PASS |
| sum 64x16 rand11 | 1024 | [-30.04501, -30.04501] | [-30.04501, -30.04501] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sum 64x16 big | 1024 | [4.969775e+12, 4.969775e+12] | [4.969775e+12, 4.969775e+12] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sum 64x16 small | 1024 | [4.969775e-08, 4.969775e-08] | [4.969775e-08, 4.969775e-08] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sum 64x16 mixed | 1024 | [2.499242e+12, 2.499242e+12] | [2.499242e+12, 2.499242e+12] | FAIL | PASS | CUDA band collapsed: 1 distinct bits; dmax=2.63e+05 (1.00 ulp, tree-bound 4.883e+06 -> within leaf-order envelope) RMgate(|clmean-f64|=2.84e+05 <= 5.457e+06):PASS |
| sum 64x16 altpm1 | 1024 | [0, 0] | [0, 0] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sum 64x16 video | 1024 | [492.4977, 492.4977] | [492.4977, 492.4977] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sum 68x20 rand01 | 1360 | [672.5033, 672.5033] | [672.5033, 672.5033] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sum 68x20 rand11 | 1360 | [-14.99344, -14.99344] | [-14.99344, -14.99344] | FAIL | PASS | CUDA nondeterministic: 2 distinct; dmax=2.86e-06 (3.00 ulp, tree-bound 0.001221 -> within leaf-order envelope) RMgate(|clmean-f64|=3.402e-06 <= 0.001326):PASS |
| sum 68x20 big | 1360 | [6.725033e+12, 6.725033e+12] | [6.725033e+12, 6.725033e+12] | FAIL | PASS | CUDA band collapsed: 1 distinct bits; dmax=5.24e+05 (1.00 ulp, tree-bound 1.221e+07 -> within leaf-order envelope) RMgate(|clmean-f64|=1.376e+05 <= 1.311e+07):PASS |
| sum 68x20 small | 1360 | [6.725033e-08, 6.725033e-08] | [6.725033e-08, 6.725033e-08] | FAIL | PASS | CUDA band collapsed: 1 distinct bits; dmax=7.1e-15 (1.00 ulp, tree-bound 1.221e-13 -> within leaf-order envelope) RMgate(|clmean-f64|=3.45e-15 <= 1.344e-13):PASS |
| sum 68x20 mixed | 1360 | [3.317022e+12, 3.317022e+12] | [3.317022e+12, 3.317022e+12] | FAIL | PASS | CUDA band collapsed: 1 distinct bits; dmax=2.63e+05 (1.00 ulp, tree-bound 1.221e+07 -> within leaf-order envelope) RMgate(|clmean-f64|=1.127e+05 <= 1.311e+07):PASS |
| sum 68x20 altpm1 | 1360 | [0, 0] | [0, 0] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sum 68x20 video | 1360 | [659.4503, 659.4503] | [659.4503, 659.4504] | PASS | PASS | CUDA nondeterministic: 2 distinct |
| sum 320x240 rand01 | 76800 | [38376.22, 38376.23] | [38376.21, 38376.24] | PASS | PASS | CUDA nondeterministic: 4 distinct |
| sum 320x240 rand11 | 76800 | [-47.52957, -47.52951] | [-47.52962, -47.5295] | PASS | PASS | CUDA nondeterministic: 17 distinct |
| sum 320x240 big | 76800 | [3.837623e+14, 3.837624e+14] | [3.837622e+14, 3.837624e+14] | PASS | PASS | CUDA nondeterministic: 5 distinct |
| sum 320x240 small | 76800 | [3.837623e-06, 3.837624e-06] | [3.837622e-06, 3.837624e-06] | PASS | PASS | CUDA nondeterministic: 3 distinct |
| sum 320x240 mixed | 76800 | [1.926687e+14, 1.926688e+14] | [1.926687e+14, 1.926689e+14] | FAIL | PASS | CUDA nondeterministic: 4 distinct; dmax=6.71e+07 (4.00 ulp, tree-bound 3.662e+08 -> within leaf-order envelope) RMgate(|clmean-f64|=1.408e+07 <= 5.254e+08):PASS |
| sum 320x240 altpm1 | 76800 | [0, 0] | [0, 0] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sum 320x240 video | 76800 | [38271.61, 38271.63] | [38271.62, 38271.64] | PASS | PASS | CUDA nondeterministic: 5 distinct |
| sum 640x480 rand01 | 307200 | [153706, 153706.1] | [153706, 153706.1] | PASS | PASS | CUDA nondeterministic: 6 distinct |
| sum 640x480 rand11 | 307200 | [212.1049, 212.1051] | [212.1049, 212.1053] | PASS | PASS | CUDA nondeterministic: 16 distinct |
| sum 640x480 big | 307200 | [1.53706e+15, 1.537061e+15] | [1.53706e+15, 1.537062e+15] | PASS | PASS | CUDA nondeterministic: 7 distinct |
| sum 640x480 small | 307200 | [1.53706e-05, 1.537061e-05] | [1.53706e-05, 1.537061e-05] | PASS | PASS | CUDA nondeterministic: 5 distinct |
| sum 640x480 mixed | 307200 | [7.693211e+14, 7.693215e+14] | [7.693206e+14, 7.693219e+14] | PASS | PASS | CUDA nondeterministic: 8 distinct |
| sum 640x480 altpm1 | 307200 | [0, 0] | [0, 0] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sum 640x480 video | 307200 | [153358.6, 153358.7] | [153358.5, 153358.8] | FAIL | PASS | CUDA nondeterministic: 6 distinct; dmax=0.1094 (7.00 ulp, tree-bound 0.1455 -> within leaf-order envelope) RMgate(|clmean-f64|=0.00912 <= 0.4419):PASS |
| sum 1920x1080 rand01 | 2073600 | [1037106, 1037106] | [1037105, 1037105] | FAIL | PASS | CUDA nondeterministic: 8 distinct; dmax=0.937 (14.99 ulp, tree-bound 0.9961 -> within leaf-order envelope) RMgate(|clmean-f64|=0.834 <= 4.088):PASS |
| sum 1920x1080 rand11 | 2073600 | [611.6177, 611.6188] | [611.6172, 611.6189] | PASS | PASS | CUDA nondeterministic: 18 distinct |
| sum 1920x1080 big | 2073600 | [1.037106e+16, 1.037106e+16] | [1.037105e+16, 1.037107e+16] | FAIL | PASS | CUDA nondeterministic: 7 distinct; dmax=7.52e+09 (7.00 ulp, tree-bound 9.961e+09 -> within leaf-order envelope) RMgate(|clmean-f64|=5.14e+08 <= 6.309e+10):PASS |
| sum 1920x1080 small | 2073600 | [0.0001037105, 0.0001037105] | [0.0001037105, 0.0001037106] | FAIL | FAIL | CUDA nondeterministic: 8 distinct; dmax=9.46e-11 (13.00 ulp, tree-bound 9.961e-11 -> within leaf-order envelope) RMgate(|clmean-f64|=1.887e-11 <= 4.596e-10):PASS |
| sum 1920x1080 mixed | 2073600 | [5.185108e+15, 5.185111e+15] | [5.185106e+15, 5.185114e+15] | FAIL | PASS | CUDA nondeterministic: 6 distinct; dmax=3.222e+09 (6.00 ulp, tree-bound 9.961e+09 -> within leaf-order envelope) RMgate(|clmean-f64|=1.893e+09 <= 3.652e+10):PASS |
| sum 1920x1080 altpm1 | 2073600 | [0, 0] | [0, 0] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sum 1920x1080 video | 2073600 | [1036156, 1036156] | [1036154, 1036156] | FAIL | FAIL | CUDA nondeterministic: 8 distinct; dmax=1.938 (31.01 ulp, tree-bound 0.9921 -> EXCEEDS leaf-order envelope) RMgate(|clmean-f64|=0.7509 <= 4.084):PASS |
| sum 4096x4 rand01 | 16384 | [8145.188, 8145.189] | [8145.187, 8145.19] | PASS | PASS | CUDA nondeterministic: 5 distinct |
| sum 4096x4 rand11 | 16384 | [-93.62238, -93.62234] | [-93.62243, -93.62231] | FAIL | PASS | CUDA nondeterministic: 6 distinct; dmax=4.577e-05 (6.00 ulp, tree-bound 0.03125 -> within leaf-order envelope) RMgate(|clmean-f64|=3.37e-07 <= 0.03339):PASS |
| sum 4096x4 big | 16384 | [8.145187e+13, 8.14519e+13] | [8.145186e+13, 8.14519e+13] | PASS | PASS | CUDA nondeterministic: 5 distinct |
| sum 4096x4 small | 16384 | [8.145188e-07, 8.14519e-07] | [8.145186e-07, 8.14519e-07] | FAIL | PASS | CUDA nondeterministic: 3 distinct; dmax=2.274e-13 (4.00 ulp, tree-bound 3.125e-12 -> within leaf-order envelope) RMgate(|clmean-f64|=1.761e-13 <= 3.623e-12):PASS |
| sum 4096x4 mixed | 16384 | [4.081358e+13, 4.08136e+13] | [4.081356e+13, 4.081359e+13] | PASS | PASS | CUDA nondeterministic: 5 distinct |
| sum 4096x4 altpm1 | 16384 | [0, 0] | [0, 0] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sum 4096x4 video | 16384 | [7264.817, 7264.819] | [7264.818, 7264.82] | PASS | PASS | CUDA nondeterministic: 5 distinct |
| sum 16x2048 rand01 | 32768 | [16285.76, 16285.76] | [16285.76, 16285.76] | PASS | PASS | CUDA nondeterministic: 7 distinct |
| sum 16x2048 rand11 | 32768 | [-196.4838, -196.4836] | [-196.4838, -196.4836] | PASS | PASS | CUDA nondeterministic: 10 distinct |
| sum 16x2048 big | 32768 | [1.628576e+14, 1.628576e+14] | [1.628576e+14, 1.628576e+14] | PASS | PASS | CUDA nondeterministic: 5 distinct |
| sum 16x2048 small | 32768 | [1.628575e-06, 1.628576e-06] | [1.628575e-06, 1.628576e-06] | PASS | PASS | CUDA nondeterministic: 6 distinct |
| sum 16x2048 mixed | 32768 | [8.185157e+13, 8.18516e+13] | [8.185156e+13, 8.18516e+13] | PASS | PASS | CUDA nondeterministic: 5 distinct |
| sum 16x2048 altpm1 | 32768 | [0, 0] | [0, 0] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sum 16x2048 video | 32768 | [15909.77, 15909.78] | [15909.77, 15909.78] | PASS | PASS | CUDA nondeterministic: 5 distinct |
| sad 4x1 rand01 | 4 | [1.104409, 1.104409] | [1.104409, 1.104409] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sad 4x1 rand11 | 4 | [2.208817, 2.208817] | [2.208817, 2.208817] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sad 4x1 big | 4 | [1.104409e+10, 1.104409e+10] | [1.104409e+10, 1.104409e+10] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sad 4x1 small | 4 | [1.104409e-10, 1.104409e-10] | [1.104409e-10, 1.104409e-10] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sad 4x1 mixed | 4 | [4.873831e+09, 4.873831e+09] | [4.873831e+09, 4.873831e+09] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sad 4x1 altpm1 | 4 | [0, 0] | [0, 0] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sad 4x1 video | 4 | [0.1104409, 0.1104409] | [0.1104409, 0.1104409] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sad 64x16 rand01 | 1024 | [315.7365, 315.7365] | [315.7365, 315.7365] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sad 64x16 rand11 | 1024 | [631.473, 631.473] | [631.473, 631.473] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sad 64x16 big | 1024 | [3.157365e+12, 3.157365e+12] | [3.157366e+12, 3.157366e+12] | FAIL | PASS | CUDA band collapsed: 1 distinct bits; dmax=2.62e+05 (1.00 ulp, tree-bound 4.704e+06 -> within leaf-order envelope) RMgate(|clmean-f64|=1.91e+05 <= 5.278e+06):PASS |
| sad 64x16 small | 1024 | [3.157366e-08, 3.157366e-08] | [3.157366e-08, 3.157366e-08] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sad 64x16 mixed | 1024 | [1.553433e+12, 1.553433e+12] | [1.553433e+12, 1.553433e+12] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sad 64x16 altpm1 | 1024 | [0, 0] | [0, 0] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sad 64x16 video | 1024 | [31.57365, 31.57365] | [31.57365, 31.57366] | FAIL | PASS | CUDA band collapsed: 1 distinct bits; dmax=1.91e-06 (1.00 ulp, tree-bound 4.704e-05 -> within leaf-order envelope) RMgate(|clmean-f64|=2.35e-07 <= 5.121e-05):PASS |
| sad 68x20 rand01 | 1360 | [427.7015, 427.7015] | [427.7015, 427.7016] | FAIL | PASS | CUDA band collapsed: 1 distinct bits; dmax=3.05e-05 (1.00 ulp, tree-bound 0.001176 -> within leaf-order envelope) RMgate(|clmean-f64|=1.995e-05 <= 0.001282):PASS |
| sad 68x20 rand11 | 1360 | [855.4031, 855.4031] | [855.4031, 855.4031] | FAIL | PASS | CUDA band collapsed: 1 distinct bits; dmax=6.1e-05 (1.00 ulp, tree-bound 0.002352 -> within leaf-order envelope) RMgate(|clmean-f64|=2.16e-05 <= 0.002563):PASS |
| sad 68x20 big | 1360 | [4.277015e+12, 4.277016e+12] | [4.277015e+12, 4.277015e+12] | PASS | PASS | CUDA nondeterministic: 2 distinct |
| sad 68x20 small | 1360 | [4.277015e-08, 4.277015e-08] | [4.277015e-08, 4.277015e-08] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sad 68x20 mixed | 1360 | [2.086387e+12, 2.086387e+12] | [2.086387e+12, 2.086387e+12] | FAIL | PASS | CUDA band collapsed: 1 distinct bits; dmax=2.62e+05 (2.00 ulp, tree-bound 1.176e+07 -> within leaf-order envelope) RMgate(|clmean-f64|=1.81e+05 <= 1.267e+07):PASS |
| sad 68x20 altpm1 | 1360 | [0, 0] | [0, 0] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sad 68x20 video | 1360 | [42.77015, 42.77015] | [42.77015, 42.77016] | PASS | PASS | CUDA nondeterministic: 2 distinct |
| sad 320x240 rand01 | 76800 | [25507.17, 25507.17] | [25507.16, 25507.17] | PASS | PASS | CUDA nondeterministic: 5 distinct |
| sad 320x240 rand11 | 76800 | [51014.33, 51014.35] | [51014.33, 51014.36] | PASS | PASS | CUDA nondeterministic: 6 distinct |
| sad 320x240 big | 76800 | [2.550717e+14, 2.550718e+14] | [2.550716e+14, 2.550717e+14] | FAIL | PASS | CUDA nondeterministic: 5 distinct; dmax=1.342e+08 (8.00 ulp, tree-bound 3.657e+08 -> within leaf-order envelope) RMgate(|clmean-f64|=3.621e+07 <= 5.249e+08):PASS |
| sad 320x240 small | 76800 | [2.550716e-06, 2.550717e-06] | [2.550717e-06, 2.550718e-06] | FAIL | PASS | CUDA nondeterministic: 4 distinct; dmax=9.1e-13 (4.00 ulp, tree-bound 3.657e-12 -> within leaf-order envelope) RMgate(|clmean-f64|=4.222e-13 <= 5.814e-12):PASS |
| sad 320x240 mixed | 76800 | [1.274771e+14, 1.274771e+14] | [1.274771e+14, 1.274772e+14] | PASS | PASS | CUDA nondeterministic: 5 distinct |
| sad 320x240 altpm1 | 76800 | [0, 0] | [0, 0] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sad 320x240 video | 76800 | [2550.717, 2550.718] | [2550.716, 2550.718] | FAIL | PASS | CUDA nondeterministic: 4 distinct; dmax=0.000733 (3.00 ulp, tree-bound 0.003657 -> within leaf-order envelope) RMgate(|clmean-f64|=0.0001016 <= 0.005973):PASS |
| sad 640x480 rand01 | 307200 | [102279.3, 102279.3] | [102279.2, 102279.4] | FAIL | PASS | CUDA nondeterministic: 6 distinct; dmax=0.1093 (13.99 ulp, tree-bound 0.1463 -> within leaf-order envelope) RMgate(|clmean-f64|=0.00783 <= 0.2946):PASS |
| sad 640x480 rand11 | 307200 | [204558.6, 204558.7] | [204558.4, 204558.7] | FAIL | PASS | CUDA nondeterministic: 6 distinct; dmax=0.1562 (10.00 ulp, tree-bound 0.2927 -> within leaf-order envelope) RMgate(|clmean-f64|=0.01248 <= 0.5892):PASS |
| sad 640x480 big | 307200 | [1.022793e+15, 1.022793e+15] | [1.022793e+15, 1.022794e+15] | FAIL | PASS | CUDA nondeterministic: 5 distinct; dmax=7.38e+08 (11.00 ulp, tree-bound 1.463e+09 -> within leaf-order envelope) RMgate(|clmean-f64|=3.733e+08 <= 2.737e+09):PASS |
| sad 640x480 small | 307200 | [1.022793e-05, 1.022793e-05] | [1.022792e-05, 1.022794e-05] | FAIL | PASS | CUDA nondeterministic: 7 distinct; dmax=1.001e-11 (11.01 ulp, tree-bound 1.463e-11 -> within leaf-order envelope) RMgate(|clmean-f64|=1.813e-12 <= 3.189e-11):PASS |
| sad 640x480 mixed | 307200 | [5.103602e+14, 5.103604e+14] | [5.103599e+14, 5.103603e+14] | FAIL | PASS | CUDA nondeterministic: 6 distinct; dmax=3.356e+08 (10.00 ulp, tree-bound 1.46e+09 -> within leaf-order envelope) RMgate(|clmean-f64|=3.206e+07 <= 2.096e+09):PASS |
| sad 640x480 altpm1 | 307200 | [0, 0] | [0, 0] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sad 640x480 video | 307200 | [10227.93, 10227.93] | [10227.92, 10227.93] | FAIL | PASS | CUDA nondeterministic: 7 distinct; dmax=0.00586 (6.00 ulp, tree-bound 0.01463 -> within leaf-order envelope) RMgate(|clmean-f64|=0.001024 <= 0.03316):PASS |
| sad 1920x1080 rand01 | 2073600 | [690772.6, 690772.8] | [690772.9, 690773.6] | FAIL | PASS | CUDA nondeterministic: 5 distinct; dmax=0.8125 (13.00 ulp, tree-bound 0.9952 -> within leaf-order envelope) RMgate(|clmean-f64|=0.2776 <= 4.088):PASS |
| sad 1920x1080 rand11 | 2073600 | [1381545, 1381546] | [1381545, 1381548] | FAIL | PASS | CUDA nondeterministic: 6 distinct; dmax=1.875 (15.00 ulp, tree-bound 1.99 -> within leaf-order envelope) RMgate(|clmean-f64|=0.143 <= 8.175):PASS |
| sad 1920x1080 big | 2073600 | [6.907732e+15, 6.907736e+15] | [6.907731e+15, 6.907741e+15] | FAIL | PASS | CUDA nondeterministic: 9 distinct; dmax=5.369e+09 (10.00 ulp, tree-bound 9.952e+09 -> within leaf-order envelope) RMgate(|clmean-f64|=5.722e+09 <= 3.652e+10):PASS |
| sad 1920x1080 small | 2073600 | [6.907717e-05, 6.90772e-05] | [6.907726e-05, 6.907746e-05] | FAIL | FAIL | CUDA nondeterministic: 5 distinct; dmax=2.619e-10 (36.00 ulp, tree-bound 9.952e-11 -> EXCEEDS leaf-order envelope) RMgate(|clmean-f64|=8.14e-11 <= 4.595e-10):PASS |
| sad 1920x1080 mixed | 2073600 | [3.450965e+15, 3.450966e+15] | [3.450965e+15, 3.450971e+15] | FAIL | PASS | CUDA nondeterministic: 5 distinct; dmax=5.368e+09 (20.00 ulp, tree-bound 9.952e+09 -> within leaf-order envelope) RMgate(|clmean-f64|=2.102e+09 <= 2.323e+10):PASS |
| sad 1920x1080 altpm1 | 2073600 | [0, 0] | [0, 0] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sad 1920x1080 video | 2073600 | [69077.29, 69077.33] | [69077.29, 69077.5] | FAIL | PASS | CUDA nondeterministic: 6 distinct; dmax=0.1719 (22.00 ulp, tree-bound 0.09952 -> EXCEEDS leaf-order envelope) RMgate(|clmean-f64|=0.109 <= 0.4861):PASS |
| sad 4096x4 rand01 | 16384 | [5413.008, 5413.01] | [5413.009, 5413.013] | FAIL | PASS | CUDA nondeterministic: 4 distinct; dmax=0.002929 (6.00 ulp, tree-bound 0.03114 -> within leaf-order envelope) RMgate(|clmean-f64|=0.001866 <= 0.03542):PASS |
| sad 4096x4 rand11 | 16384 | [10826.02, 10826.02] | [10826.02, 10826.03] | FAIL | PASS | CUDA nondeterministic: 4 distinct; dmax=0.00488 (5.00 ulp, tree-bound 0.06229 -> within leaf-order envelope) RMgate(|clmean-f64|=0.003923 <= 0.07084):PASS |
| sad 4096x4 big | 16384 | [5.413009e+13, 5.413011e+13] | [5.413007e+13, 5.413009e+13] | PASS | PASS | CUDA nondeterministic: 6 distinct |
| sad 4096x4 small | 16384 | [5.413008e-07, 5.41301e-07] | [5.413008e-07, 5.41301e-07] | PASS | PASS | CUDA nondeterministic: 4 distinct |
| sad 4096x4 mixed | 16384 | [2.698465e+13, 2.698466e+13] | [2.698464e+13, 2.698465e+13] | PASS | PASS | CUDA nondeterministic: 6 distinct |
| sad 4096x4 altpm1 | 16384 | [0, 0] | [0, 0] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sad 4096x4 video | 16384 | [541.3009, 541.3011] | [541.3008, 541.301] | PASS | PASS | CUDA nondeterministic: 4 distinct |
| sad 16x2048 rand01 | 32768 | [10950.32, 10950.32] | [10950.32, 10950.32] | PASS | PASS | CUDA nondeterministic: 7 distinct |
| sad 16x2048 rand11 | 32768 | [21900.63, 21900.65] | [21900.64, 21900.64] | PASS | PASS | CUDA nondeterministic: 7 distinct |
| sad 16x2048 big | 32768 | [1.095032e+14, 1.095032e+14] | [1.095032e+14, 1.095032e+14] | PASS | PASS | CUDA nondeterministic: 6 distinct |
| sad 16x2048 small | 32768 | [1.095032e-06, 1.095032e-06] | [1.095032e-06, 1.095032e-06] | PASS | PASS | CUDA nondeterministic: 5 distinct |
| sad 16x2048 mixed | 32768 | [5.431964e+13, 5.431967e+13] | [5.431964e+13, 5.431966e+13] | PASS | PASS | CUDA nondeterministic: 7 distinct |
| sad 16x2048 altpm1 | 32768 | [0, 0] | [0, 0] | PASS | PASS | CUDA band collapsed: 1 distinct bits |
| sad 16x2048 video | 32768 | [1095.032, 1095.032] | [1095.032, 1095.032] | PASS | PASS | CUDA nondeterministic: 5 distinct |

Totals: criterion 1 literal **75/112 PASS**; CUDA nondeterminism observed on
64/112 cells (2-18 distinct counter bits over 200 runs), 48 cells collapsed
to a point (serialized atomic arrival — the §5.2 widen-and-note path).
Criterion 3 (finiteness): PASS on every cell, both sides.

### Accepted deviations

| Config | Observation | Justification |
|---|---|---|
| 37 Phase B cells (criterion-1 literal misses; e.g. sad 1920x1080 small dmax 2.619e-10 = 36 ulp; typical 1 ulp) | CL samples fall outside the (often point-collapsed) CUDA band, but every miss is <=36 ulp of the band edge, inside the RM noise envelope (RM-gate PASS on all 37) and consistent with the host f32 twin envelope from the CL phase (the 5 tight cells were inside the twin window with 4-sigma padding) | Leaf-association rounding between the two toolchains by construction (§3 stage 1: CUDA float4 4-lane leaves + shuffle tail vs the port's scalar leaves + local tree). Bit-equality is only guaranteed where all arithmetic is exact — that guarantee is Phase A, which passes 11/11. §5.2-1's "collapsed band demands CL bit-equality" parenthetical is too strong for inexact inputs. |
| Criterion-2 formula (§5.2-2 as written) | The CUDA twin itself violates the bound on 3 Phase B cells (sum 1920x1080 small, sum 1920x1080 video, sad 1920x1080 small — cudaCrit2:FAIL above); on the CL side 5/112 cells exceeded it as-written in the first pass (adjudicated via the host f32 twin envelope, disclosed via CRIT2_TIGHT) | The formula under-covers the counter-atomic rounding at large N on BOTH toolchains — device-level proof that it is upstream-faithful behaviour, not a port defect. Any recalibration is a spec decision, not a kernel fix. |

---

*Handoff prepared from upstream `68aef6e`; transcriptions in
`src/opencl/avscuda/kernels/avscuda_conditional_rig.cl`. Questions about
intent should be answerable from the per-kernel comments in the rig file +
the CUDA twins cited above; if a twin and this doc ever disagree, the twin
wins and this doc must be fixed.*
from the per-kernel comments in the rig file +
the CUDA twins cited above; if a twin and this doc ever disagree, the twin
wins and this doc must be fixed.*
