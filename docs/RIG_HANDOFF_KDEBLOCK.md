# RIG HANDOFF — KDeblock provisional kernels (verification spec for another agent)

This document hands the **unverified** KDeblock-family OpenCL kernels to another
agent (human or AI) for verification. It is written to be self-sufficient: if
you follow it top to bottom you can verify all eight kernels without asking the
original author anything.

## 0. TL;DR

- **Package under verification:** `src/opencl/kfm/kernels/kfm_deblock_rig.cl`
  (8 kernels + 2 tables at handoff, all `// RIG-VERIFY`; 6 kernels + 2 tables
  have since graduated — only the 2 sharpen kernels remain (deterministic
  behaviour pinned by mirror+golden; device run still required) — deliberately separated from the
  verified `kfm_deblock.cl`).
- **Ground truth:** `rigaya/AviSynthCUDAFilters`, `KFM/Deblock.cu`, commit
  `cceb8da0e623e6bf5eea2cf655b06d4428e1600b` (§2).
- **Your job:** for each kernel, build the repo-standard verification pair —
  a scalar CPU mirror in `sim/` + an independent Python golden in `python/` —
  wire it into `make test`, prove **bit-exact** agreement, then *graduate* the
  kernel: move it to `kfm_deblock.cl`, mark `// ALG-VERIFIED`, update docs.
- **Closest template to copy:** `sim/kfm_deblock_qp_ref.cpp` +
  `python/run_kfm_deblock_qp.py` (same family, same harness style).
- **Suggested order (easy first):** `kf_scale_qp` → `kf_sharpen_coeff` →
  `kf_max_h` → `kf_max_v` → `kf_max_vh` → `kf_merge_deblock` →
  `kf_show_sharpen_coeff` → `kf_sharpen` (last two need a device comparison).

## 1. What is in the package (and what is not)

### 1.1 In scope — the eight kernels in `kfm_deblock_rig.cl`

| # | OpenCL kernel | CUDA device twin | Exact CPU twin? | Difficulty |
|---|---|---|---|---|
| 1 | `kf_merge_deblock` | `kl_merge_deblock` | `cpu_merge_deblock` (exact) | hard (float32 + layout proof) — **GRADUATED** |
| 2 | `kf_max_vh` | `kl_max_vh` | none (device-only upstream) | medium (pad contract) — **GRADUATED** |
| 3 | `kf_max_v` | `kl_max_v` | `cpu_max_v` (exact) | easy-medium — **GRADUATED** |
| 4 | `kf_max_h` | `kl_max_h` | `cpu_max_h` (exact) | easy — **GRADUATED** |
| 5 | `kf_scale_qp` | `kl_scale_qp` | `cpu_scale_qp` (exact) | easy — **GRADUATED** |
| 6 | `kf_sharpen_coeff` | `kl_sharpen_coeff` | `cpu_sharpen_coeff` (exact) | easy — **GRADUATED** |
| 7 | `kf_sharpen` | `kl_sharpen` | `cpu_sharpen` (differs: guarded borders, no quirk) | hard (texture gap + quirk — device run required) — **PINNED** (mirror+golden; not graduated) |
| 8 | `kf_show_sharpen_coeff` | `kl_show_sharpen_coeff` | `cpu_show_sharpen_coeff` (exact for the manual form) | medium (mirror+golden pins it; device run closes the texture gap) — **PINNED** (not graduated) |

Plus two file-scope tables (`g_ldither`, `g_sharpen_coeff`), the
`kf_sharpen_bilinear` helper, and a private copy
of the `kf_norm_qscale` helper (duplicated from `kfm_deblock.cl` so the rig
file is standalone — keep the copies in sync if you touch it).

All eight are **scalar** transcriptions: CUDA processes 4-wide vectors
(`uchar4`/`ushort4`) but every lane is independent, so the scalar port is
lane-identical by construction. Your verification must still prove it.
Kernels 7–8 additionally replace the CUDA **texture fetch** with manual
float32 bilinear (`kf_sharpen_bilinear`, CPU-twin expression) — see §4.5/§4.6
for why that substitution is structural-exact but rounding-inexact.

### 1.2 Explicitly OUT of scope

- The SharpenFilter **host** (coeff-frame allocation, `TextureObject` setup,
  GaussResize unsharp clip, per-plane dispatch) and the KDeblock **host
  assembly** below — you verify kernels, not filters.
- Pad kernels `kl_padv`/`kl_padh` (live in KFMFilterBase, not Deblock.cu) —
  since ported as `kf_padv`/`kf_padh` in `kfm_filterbase.cl`, `// ALG-VERIFIED`;
  no action needed here.
- The KDeblock **host assembly** (DeblockPlane: pad → qp-table → deblock →
  merge sequencing, AviSynth glue). You verify kernels, not the filter.
- KTGMC `ktgmc_motion.cl` RIG-VERIFY items (pad/mirror, `kt_most_freq_mv`,
  block-search helpers, `kt_calc_all_sad`) — a separate handoff, see
  `docs/CODEX_HANDOFF.md` + `docs/BLOCKSEARCH_MODEL.md`.

## 2. Upstream grounding (exact)

Upstream repo: `https://github.com/rigaya/AviSynthCUDAFilters` (KFM is MIT).

Reference commit (what the transcriptions were made against):

```
cceb8da0e623e6bf5eea2cf655b06d4428e1600b
```

Reproduce it:

```sh
git clone https://github.com/rigaya/AviSynthCUDAFilters /tmp/avs_cuda
cd /tmp/avs_cuda && git checkout cceb8da0e623e6bf5eea2cf655b06d4428e1600b
```

Practical notes: `KFM/Deblock.cu` uses **CRLF** line endings and Shift-JIS
bytes in comments — pipe through `tr -d '\r'` and read as latin-1/bytes.
Line numbers below are 1-based at `cceb8da`. (All eight kernels + both tables
were re-checked **semantically identical** at upstream HEAD `8e086bb`; only
brace style drifted, so line numbers there differ but bodies do not.)

### 2.1 Line map at `cceb8da`, `KFM/Deblock.cu`

| Upstream symbol | Lines | Notes |
|---|---|---|
| `g_ldither[8][2]` (`uchar4`) | 639–650 | Bayer dither contents |
| `kl_merge_deblock` | 652–670 | device kernel |
| `cpu_merge_deblock` | 672–690 | exact CPU twin — transcribe this |
| `kl_max_vh` | 749–763 | device-only (no CPU twin) |
| `kl_max_v` | 766–779 | device (`uchar4`) |
| `cpu_max_v` | 781–793 | exact CPU twin |
| `kl_max_h` | 795–808 | device (scalar `uint8_t`) |
| `cpu_max_h` | 810–823 | exact CPU twin |
| `d_sharpen_coeff[]` / `g_sharpen_coeff[]` | 1166–1173 / 1175–1182 | 30-entry LUT (device/host copies) |
| `kl_sharpen_coeff` | 1184–1194 | device kernel |
| `cpu_sharpen_coeff` | 1196–1205 | exact CPU twin |
| `kl_sharpen` | 1208–1241 | device kernel (texture coeff!) |
| `cpu_sharpen` | 1244–1312 | CPU twin (manual bilinear, guarded borders) |
| `kl_show_sharpen_coeff` | 1315–1325 | device kernel (texture coeff!) |
| `cpu_show_sharpen_coeff` | 1328–1347 | CPU twin (manual bilinear — exact for our form) |
| texture setup (`SharpenFilter`) | ~1431 | Clamp/Linear/NormalizedFloat over the qp-sized uchar coeff plane |
| `width % 8 == 0` check + `coeffvi` | ~1491–1493 | `coeffvi = qpclip dims`; taps always in bounds |
| `kl_scale_qp` | 1838–1846 | device kernel (ShowQP filter) |
| `cpu_scale_qp` | 1848–1856 | exact CPU twin |
| `norm_qscale` | (find by name) | `__host__ __device__` helper, QP scale types 0–3 |
| merge call site (`DeblockPlane`) | ~1650–1662 | grid, pitches, base offsets, shift/maxv derivation |
| `kl_max_*<5>` call sites (`QPForDeblock`) | ~927–951 | always `RADIUS=5`; pad+8+8*pitch pointers |
| `kl_scale_qp` call site (`ShowQP`) | ~1944–1949 | |
| `kl_deblock` store | ~419–424 in body 357–426 | `out[(blockIdx.x*4+tx) + ((bh*off_z+blockIdx.y)*8+y)*out_pitch]`, `out` is `ushort2*` |

### 2.2 Vector-helper semantics (`common/VectorFunctions.cuh`)

You need exactly three facts (all verified in the header):

1. `to_int(ushort4)` → per-lane `(int)`; `to_float(int4)` → per-lane
   `(float)`. Plain widening, no saturation.
2. `min(float4, float)` is **per-component** `fmin`.
3. `VHelper<uchar4>::cast_to(float4)` / `VHelper<ushort4>::cast_to(float4)` are
   **C-style truncation casts per lane** (`(uchar)a.x`, …): round-toward-zero,
   **no lower clamp**, wrap mod 256/65536 on overflow. The OpenCL `(PX)v`
   C-cast is the faithful equivalent.

## 3. The repo's verification method (ALG-VERIFY)

Every `// ALG-VERIFIED` kernel in this repo was proven the same way, without
any GPU:

1. **`sim/<family>_<thing>_ref.cpp`** — scalar CPU mirror, a verbatim
   transcription of the CUDA twin (device kernel preferred; CPU twin when the
   device path is identical). Compiled with **`-ffp-contract=off`** (this is
   load-bearing: it forbids FMA contraction so the mirror matches OpenCL's
   default float semantics).
2. **`python/run_<thing>.py`** — an *independent* Python golden, written from
   the CUDA math (not by copying the mirror), emulating float32 per operation
   where float is involved (`struct.pack/unpack('f', …)` after every op).
3. **`Makefile`** wires the runner into `make test`; the runner prints
   `...: PASS (N cases)` and exits non-zero on any mismatch. Bit-exact means
   **zero differing elements across all randomized cases**, 8-bit and 16-bit
   (`-DPX=uchar` / `-DPX=ushort`) where the kernel is bit-depth-generic.
4. `lint/lint_opencl.sh` must stay green for the touched `.cl` files (it is a
   genuine C parse via `lint/oc_shim.h`, catching typos — not semantics).

Float rules (from the KDeblock/KEdgeLevel/KTemporalNR precedents):

- Mirror built `-O2 -std=c++17 -ffp-contract=off`.
- Golden emulates **float32 per operation**; integer→float conversions are
  exact for the magnitudes here (≤ 262140 for the merge sum).
- No FMA anywhere; expression order must match the transcription exactly
  (e.g. `(float)sum * (1/(1<<shift)) + (float)d * (1/64)` — the division
  `1.0f/(float)(1<<shift)` is itself a float32 op).

## 4. Per-kernel specification

### 4.1 `kf_merge_deblock` — **GRADUATED**

Signature (grid: 2D `(vis_width, vis_height)` **pixels**):

```c
kernel void kf_merge_deblock(
    __global const ushort* tmp, int tmp_pitch_u4, int tmp_ipitch_rows,
    __global PX* out, int out_pitch,
    int vis_width, int vis_height, int shift, float maxv)
```

Exact per-pixel math (must match CUDA lane-for-lane):

```
X   = x >> 2                      // ushort4 column
L   = x & 3                       // lane inside the ushort4
sum = Σ_{k=0..3} (int)tmp[((X + (tmp_ipitch_rows*k + y)*tmp_pitch_u4) << 2) + L]
v   = (float)sum * (1.0f/(float)(1<<shift)) + (float)g_ldither[y&7][X&1][L] * (1.0f/64.0f)
v   = fmin(v, maxv)
out[x + y*out_pitch] = (PX)v      // C-truncation, NO lower clamp
```

Traps (all of these have bitten before — check each deliberately):

1. **Dither middle index is over ushort4 columns**: `g_ldither[y&7][X&1][L]`
   with `X = x>>2`. Indexing `[y&7][x&1][…]` (scalar pixels) is WRONG.
2. **`tmp` base is pre-offset by the host**: the CUDA call passes
   `tmpOut + 2 + 8*pitch` (i.e. +2 ushort4 = +8 ushorts horizontally, +8 rows
   vertically). Your test harness must apply the same pre-offset
   (+8 ushorts, +8 rows) to the accumulator base before calling the mirror.
3. **Pitch units**: `tmp_pitch_u4` is in **ushort4** units
   (= accumulator ushort-pitch >> 2); `tmp_ipitch_rows` is **rows per parity
   slice** (= bh*8). `out_pitch` is in pixels.
4. **`shift`/`maxv` derivation** (from the call site ~1650–1662):
   `deblockShift = max(0, quality+bits-10)`, `shift = quality+6-deblockShift`,
   `maxv = (float)((1<<bits)-1)`. Sweep `quality` 1..6 and `bits` 8/10/12/16
   in tests (PX=uchar covers bits=8; PX=ushort covers 10–16 with maxv set
   accordingly).
5. **Width contract**: `vis_width` must be a multiple of 4
   (pass `width & ~3`); CUDA covers `width>>2` vector lanes and never writes
   the trailing `width%4` columns.
6. **No lower clamp**: values are ≥ 0 by construction; do not add
   `max(v, 0)` — it would still be bit-identical in practice, but it is not
   what upstream computes, so the transcription (correctly) omits it.

Layout proof obligation (the reason this kernel is RIG-VERIFY, §5):
CUDA `kl_deblock` stores packed `ushort2` at `blockIdx.x*4` (`out_pitch` in
ushort2) while `kf_deblock` (§`kfm_deblock.cl`, ALG-VERIFIED) writes scalar
`ushort` at `bx*8` (`out_pitch` in ushort). These are claimed to be the same
bytes when the byte stride matches. Your verification should include an
**end-to-end layout test**: run the `kf_deblock` CPU mirror
(`sim/kfm_deblock_ref.cpp`) to produce an accumulator, feed it to your merge
mirror with `tmp_pitch_u4 = acc_pitch_ushort >> 2`, and compare against a
golden that models the CUDA packed indexing directly. If they agree, the
reconciliation is proven, not just reasoned.

### 4.2 `kf_max_vh` / `kf_max_v` / `kf_max_h` — **GRADUATED**

```c
kernel void kf_max_vh(v,h)(
    __global uchar* dst, __global const uchar* src,
    int width, int height, int pitch, int radius)
```

- `kf_max_vh`: `dst = max(src)` over the `(2R+1)²` box `[x±R]×[y±R]`.
- `kf_max_v`: `dst = max(src)` over the vertical `[y±R]` column segment.
- `kf_max_h`: `dst = max(src)` over the horizontal `[x±R]` row segment.
- Upstream always instantiates `RADIUS=5`; the OpenCL radius is a kernel arg.
  Verify at least `radius = 5` (plus 1–2 other radii for generality).
- **Edge contract (load-bearing):** reads span `±radius` around every pixel,
  so `src`/`dst` must point at the **interior of a plane padded by ≥ radius**
  — the CUDA host passes `pad+8+8*pitch` with an 8 px margin (call sites
  ~927–951). Your harness must build the padded plane (8 px/side, contents
  arbitrary — fill random to prove margin-independence of the interior
  result… careful: the result *does* depend on margin contents within radius;
  so fill margins deterministically and feed the same padded plane to mirror
  and golden).
- `kf_max_v` equivalence note: CUDA is `uchar4`-vector with width in uchar4
  units; the OpenCL grid is pixels. Lanes are independent ⇒ identical; the
  golden should model the scalar form and the mirror likewise (both trivially
  transcribed from `cpu_max_v`).
- `kl_max_vh` has **no CPU twin** (device-only upstream; the `if (true)`
  branch at ~927). Transcribe the device kernel directly. Cross-check bonus:
  box-max is separable, so `max_vh(R) ≡ max_h(R) ∘ max_v(R)` — assert this
  identity in the golden as a second independent check.
- All integer, 8-bit only (`uchar` plane) — no PX/16-bit variant needed.

### 4.3 `kf_scale_qp` — **GRADUATED**

```c
kernel void kf_scale_qp(int width, int height,
    __global uchar* dst, int dst_pitch,
    __global const uchar* src, int src_pitch, int scale_type)
```

- Per pixel: `dst = (uchar)kf_norm_qscale(src, scale_type)` with
  type 0: `q<<2`, 1: `q<<1`, 2: `q`, 3: `63-q+2`.
- **Wrap semantics:** the int→uchar conversion wraps **mod 256**, exactly like
  the CUDA assignment in `cpu_scale_qp`. Test with full-range inputs 0–255
  (type 0 with `q > 63` exercises the wrap — upstream QP inputs are ≤ 51 so
  wrap never fires in production, but the transcription must still match).
- Integer, 8-bit only. Sweep `scale_type` 0–3 (and, for robustness, an
  out-of-range type such as 4/7 — both twins fall through to `return qscale`).

### 4.4 `kf_sharpen_coeff` — **GRADUATED**

```c
kernel void kf_sharpen_coeff(__global uchar* dst, int width, int height, int pitch,
    __global const ushort* qp, int qp_pitch)
```

- Per block: `q = qp[idx] >> 3; dst = (q >= 25) ? 255 : g_sharpen_coeff[q]`.
- `qp` is `uint16_t` (`ushort`); `q` ranges 0–8191, guarded by the `>= 25`
  saturation (so LUT entries 25–29 are unreachable but must still be
  byte-correct — verify the table separately, §6).
- Integer; sweep `qp` over 0–65535 with emphasis on boundaries
  (`q` = 24/25 i.e. `qp` = 199/200/207).

### 4.5 `kf_sharpen` — **PINNED**, device run required (do last, after §4.6)

```c
kernel void kf_sharpen(
    __global PX* __restrict dst, int width, int height, int pitch,
    __global const PX* __restrict src, int src_pitch,
    __global const uchar* __restrict coeff, int coeff_pitch,
    __global const PX* __restrict unsharp)
```

Device-faithful transcription of `kl_sharpen` with ONE substitution: the
texture fetch `tex2D<float>(coeff, x/8+0.5, y/8+0.5)` is replaced by manual
float32 bilinear (`kf_sharpen_bilinear`, the CPU twins' verbatim expression).
Everything else follows the DEVICE kernel, including its quirks:

```
s = src[x + y*src_pitch]; l = h = s
(l,h) = min/max of s over the 8 neighbour taps (device order, edge-clamped):
    (max(x-1,0),        max(y-1,0)), ((x+0),          max(y-1,0)),
    (min(x+1,HEIGHT-1), max(y-1,0)), (max(x-1,0),     (y+0)),       // <- QUIRK
    (min(x+1,HEIGHT-1), (y+0)),      (max(x-1,0),     min(y+1,height-1)),
    ((x+0),             min(y+1,height-1)),
    (min(x+1,HEIGHT-1), min(y+1,height-1))                          // <- QUIRK
c = bilinear(coeff, x/8, y/8) * (1/255)
u = unsharp[x + y*pitch]                       // shares the DST pitch, as CUDA
r = s + (s-u)*c + 0.5; clamp r to [l,h]; dst = (PX)(int)r   // C-truncation
```

Must-check facts:

1. **The `height-1` quirk is INTENTIONAL.** Upstream `kl_sharpen` clamps the
   right-neighbour x by `height-1`, not `width-1`. Do NOT "fix" it — the
   transcription preserves it, and your mirror must too. It diverges from
   `cpu_sharpen` (which guards with `x < width-1`) wherever `x+1 > height-1`.
   Your tests MUST cover `width > height` configs (e.g. 32×8, 64×16) or you
   will never exercise the quirk.
2. **No `c > 0` branch.** The CPU twin skips the window when `c == 0` and the
   device always computes it; both yield `s` there
   (`(int)clamp(s+0.5, l, h) == s` since `l <= s <= h`). The transcription
   follows the device (no branch). Do not add one.
3. **Type/order fidelity:** `(s-u)` is int, converted to float for `* c`;
   `s + …` is int+float → float; `+ 0.5f`; clamp against `(float)l/h`;
   `(int)` truncation; then `PX` conversion. Mirror this order exactly.
4. **Host contract** (your harness must honour it): `width % 8 == 0`
   (enforced by the SharpenFilter ctor); `coeff` is the qp-sized uchar plane
   (`qpw = (width+15)>>3`, i.e. a +1-block margin), so taps `ix+1`/`iy+1`
   are always in bounds; `unsharp` shares dst geometry and pitch (it is the
   GaussResize clip at dst size).
5. **The texture gap (why a device run is mandatory):** CUDA filters with HW
   fixed-point weights; the transcription uses float32 bilinear. Structurally
   identical (Clamp never engages per §4.6-1 — same argument), but the two
   can differ by 1 ulp at rounding boundaries, which after `(int)` truncation
   can flip a pixel by 1. Verification = mirror+golden pin the deterministic
   behaviour (§3) AND a rig compares `.cl` output vs real `kl_sharpen` device
   output: expect exact match almost everywhere with rare ±1 diffs at
   truncation boundaries; characterise, do not hand-wave.

### 4.6 `kf_show_sharpen_coeff` — **PINNED**, device run required

```c
kernel void kf_show_sharpen_coeff(
    __global PX* __restrict dst, int width, int height, int pitch,
    __global const uchar* __restrict coeff, int coeff_pitch)
```

- Per pixel: `dst = (PX)(int)bilinear(coeff, x/8, y/8)` — the unnormalized
  manual bilinear, C-truncated. This is EXACTLY `cpu_show_sharpen_coeff`'s
  expression, so a mirror+golden pair CAN pin this kernel bit-exact (§3);
  vs the device (`(int)(normalized_tex*255)`) the texture-precision gap of
  §4.5-5 applies, closable only by a device run.
- Same coeff host contract as §4.5-4. PX-generic: verify 8-bit and 16-bit
  (`(PX)(int)c` truncates then converts; `c` ∈ [0,255] so no wrap).
- Test emphasis: fractional positions (any `x % 8 != 0`), coeff ramps
  0→255 (exercises every truncation boundary), plus random planes.

## 5. The merge accumulator-layout reconciliation (read carefully)

This is the one non-local proof in the package. Two facts:

1. CUDA `kl_deblock` (Deblock.cu ~419–424) writes its 16×16 `local_out` tile
   through an `ushort2* out` at `(blockIdx.x*4 + tx) + (off_y + y)*out_pitch`,
   `tx ∈ [0,8)`, `out_pitch` in **ushort2** units. The on-chip packing
   (`atomicAdd` of `tmp << ((off_x&1)*16)` into `local_out[…][off_x>>1]`)
   places even pixels in the low half on little-endian — i.e. scalar order.
2. `kf_deblock` (`kfm_deblock.cl`, ALG-VERIFIED) writes scalar `ushort` at
   `(bx*8 + x) + (off_y + y)*out_pitch`, `out_pitch` in **ushort** units.

Claim: (1) and (2) are the **same bytes** whenever the byte stride matches,
because a packed `ushort2` array IS a row-major `ushort` array
(`ushort2[i] == ushort[2i], ushort[2i+1]`). The merge transcription relies on
this: it consumes the accumulator as scalar `ushort` with
`tmp_pitch_u4 = acc_pitch_ushort >> 2`.

Your job: **prove, don't trust** — via the end-to-end layout test in §4.1(6).
If it fails, the transcription's indexing (not the claim) is the first
suspect; if the claim itself fails, the merge needs a re-layout stage and the
handoff author wants to know.

## 6. Table byte-checks (do these first — 5 minutes)

Before any kernel work, diff the two tables against upstream:

- `g_ldither[8][2][4]` in `kfm_deblock_rig.cl` vs Deblock.cu:639–650
  (`uchar4 g_ldither[8][2]`): flatten each `uchar4 {a,b,c,d}` to lanes
  `[0..3]` in order. All 64 bytes must match.
- `g_sharpen_coeff[30]` vs Deblock.cu:1175–1182 (`g_sharpen_coeff[]`, the host
  copy; `d_sharpen_coeff` at 1166–1173 must equal it too — assert that as
  well): all 30 bytes must match, in order:
  `0,0,0,0,0, 0,0,0,0,10, 50,90,120,150,160, 170,180,190,200,210,
  220,230,240,245,250, 255,255,255,255,255`.

A mechanical diff script (parse both files, compare) beats eyeballing.

## 7. Graduation checklist (per kernel — all boxes required)

- [ ] CPU mirror added to `sim/` (new file or extension), transcribed from the
      CUDA twin named in §1.1, compiled `-O2 -std=c++17 -ffp-contract=off`.
- [ ] Independent Python golden added to `python/` (written from the CUDA
      math, not copied from the mirror; float32 emulated per op where used).
- [ ] Runner wired into `Makefile` `test` target; prints `PASS (N cases)`.
- [ ] `make test` fully green (no regressions in other runners).
- [ ] `lint/lint_opencl.sh` green for the touched `.cl` files, both PX modes.
- [ ] Kernel MOVED from `kfm_deblock_rig.cl` to `kfm_deblock.cl` (or, if the
      rig file still holds others, the move keeps the rig file compiling —
      move its tables/helpers only when the last consumer graduates).
- [ ] Kernel comment says `// ALG-VERIFIED` with runner name + case count.
- [ ] Docs updated: `docs/KFM_PORT_SPEC.md`, `README.md`, `docs/PORT_PLAN.md`
      (and this file: mark the kernel graduated).
- [ ] For `kf_merge_deblock` only: the §5 end-to-end layout test passes.
- [ ] For `kf_sharpen` / `kf_show_sharpen_coeff` only: a device run compares
      `.cl` output against real CUDA device output (§4.5-5); the ±1 truncation
      boundary diffs are characterised and accepted. Mirror+golden alone does
      NOT graduate these two (for `kf_show_sharpen_coeff` it pins the
      deterministic behaviour, which is still worth doing first).

When the last kernel graduates and `kfm_deblock_rig.cl` is empty: **delete the
file** and this handoff doc's remaining-kernel sections (keep a one-line
pointer to the git history).

## 8. Environment constraints (read before planning a device run)

This sandbox has **no GPU, no CUDA toolkit, no OpenCL ICD, no AviSynthNeo** —
and the Debian mirrors are unreachable, so even pocl cannot be installed.
That is *why* the mirror+golden method exists: it proves the algorithm
bit-exact without executing any `.cl`. A real device run of the `.cl` itself
(device compile + dispatch + diff vs the CPU mirror) remains the rig's job
even after ALG-VERIFIED — see `docs/HOST_CONTRACT.md` for the host-runner
spec. Do not attempt to install GPU stacks here; do not gate graduation on a
device run you cannot perform.

## 9. Worked starting points (copy-paste)

Closest existing pair (same family, same harness style — imitate the file
layout, the `run_mirror` plumbing, the float32 helpers `F`/`FB`):

- `sim/kfm_deblock_qp_ref.cpp` (text-protocol CPU mirror)
- `python/run_kfm_deblock_qp.py` (independent golden + randomized cases)
- `Makefile` (`test` target wiring)

Suggested new files (names are advisory, not mandatory):

- `sim/kfm_deblock_rig_ref.cpp` — mirrors for all eight (one `main` with a mode
  flag, like the qp mirror's `M`/`S` modes).
- `python/run_kfm_deblock_rig.py` — goldens + randomized cases for all eight.

Case-count guidance (match repo precedent: 200–680 cases per runner):
scale_qp ≥200, sharpen_coeff ≥200, max_h/v/vh ≥200 each (radius sweep incl. 5,
padded planes), merge ≥300 (quality/shift sweep, 8 + 16-bit, plus the §5
end-to-end layout test on at least 20 accumulator configs), show_sharpen ≥200
(fractional positions, coeff ramps, 8 + 16-bit), sharpen ≥300 (incl. `width >
height` quirk configs, unsharp sweeps, 8 + 16-bit) — plus the §4.5-5 device
comparison for the sharpen pair.

---

*Handoff prepared from upstream `cceb8da`, transcriptions verified unchanged
vs upstream HEAD `8e086bb` (brace-style drift only). Questions about intent
should be answerable from the per-kernel comments in `kfm_deblock_rig.cl` +
the CUDA twins cited above; if a twin and this doc ever disagree, the twin
wins and this doc must be fixed.*
