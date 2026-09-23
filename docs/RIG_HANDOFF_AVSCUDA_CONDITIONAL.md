# RIG HANDOFF — AvsCUDA Conditional float reductions (verification spec for another agent)

This document hands the **unverifiable-by-construction** AvsCUDA OpenCL kernels
to another agent (human or AI) for device comparison. It is written to be
self-sufficient: if you follow it top to bottom you can close both kernels
without asking the original author anything.

Read §3 first: unlike every other handoff in this repo, there is **no
graduation path to `// ALG-VERIFIED`** here. The terminal state is
`// RIG-COMPARED` (§6), and §5 tells you exactly what evidence earns it.

## 0. TL;DR

- **Package under verification:** `src/opencl/avscuda/kernels/avscuda_conditional_rig.cl`
  (2 kernels + 1 helper, both `// RIG-VERIFY` — deliberately separated from
  the verified `avscuda_conditional.cl`).
- **Ground truth:** `rigaya/AviSynthCUDAFilters`,
  `AvsCUDA/filters/ConditionalFunctions.cu`, commit
  `68aef6e9b8a63d0dceb8e7f68b52b27431aa361f9` (§2).
- **Your job:** build a CUDA harness that runs the two
  `kl_sum_of_pixels<float4,float,float>` / `kl_sad<float4,float,float>`
  instantiations and an OpenCL host that runs the two transcriptions (§5);
  prove **bit-exact agreement on exact-arithmetic inputs** (Phase A) and
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
   `tid = get_local_id(0) + get_local_id(1)*16` and `sbuf[256]`; any other
   local size silently computes garbage.
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

You need two harnesses (names advisory, keep them out of `make test` —
suggested scratch dir `tools/rig_conditional/`, or your rig's equivalent):

- **CUDA harness** (`cond_cuda.cu`, built with nvcc): includes the upstream
  headers (`-I…/common` for `ReduceKernel.cuh`/`VectorFunctions.cuh`),
  instantiates `kl_sum_of_pixels<float4,float,float>` and
  `kl_sad<float4,float,float>` directly — no AviSynth needed. Launch exactly
  per §2.2-4 (threads 16×16, `width4 = width>>2`, counter from
  `cudaMalloc`+`cudaMemset`). Note: the `__CUDA_ARCH__ >= 800`
  `__reduce_add_sync` fast path applies to the `int` specialization only —
  float always takes the shuffle cascade (§2.1), on every arch.
- **OpenCL harness** (`cond_cl.cpp`, following `docs/HOST_CONTRACT.md` §1–§3
  with the rig file as a third program): builds
  `avscuda_conditional_rig.cl` once, launches per §4.4, reads the counter
  back after `clFinish`.

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

## 8. Results appendix (fill in on the rig — template)

### Environment

- Date(s):
- CUDA device + driver + nvcc version + compile flags:
- OpenCL device(s) + driver + build options:
- Denorm handling observed (both sides):
- Upstream checkout (must be `68aef6e`, or note drift + re-verify §2.1):

### Phase A (bit-exact) — all runs' counter bits

| Config | CUDA ×5 (hex) | OpenCL ×5 (hex) | Verdict |
|---|---|---|---|
| A1 | | | |
| … | | | |

### Phase B (in-band) — bands, means, f64ref

| Config (shape × distr) | N | CUDA band [min,max] (hex + float) | CL samples (min/mean/max) | f64ref | Crit. 1/2/3 |
|---|---|---|---|---|---|
| | | | | | |

### Accepted deviations (if none, write "none")

| Config | Observation | Justification |
|---|---|---|
| | | |

---

*Handoff prepared from upstream `68aef6e`; transcriptions in
`src/opencl/avscuda/kernels/avscuda_conditional_rig.cl`. Questions about
intent should be answerable from the per-kernel comments in the rig file +
the CUDA twins cited above; if a twin and this doc ever disagree, the twin
wins and this doc must be fixed.*
