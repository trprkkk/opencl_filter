# Performance notes — static pass over the ported OpenCL kernels (154 total)

**Method:** static source analysis only. This sandbox has no OpenCL device,
so there are no measurements, no occupancy numbers, and no benchmark claims
anywhere in this file. Every statement below is either grep-verifiable
(structure) or standard architecture reasoning (clearly marked as such).
Re-verify every "should" on a rig before acting on it.

## 0. Rules of this pass (verification-grade)

1. **No semantic edits to `// ALG-VERIFIED` files.** An ALG-VERIFIED tag
   certifies file content proven by CPU mirror + Python golden; this pass
   does not touch those bytes, not even for attributes. Recommendations for
   them are documentary only (§1, §7).
2. **Attributes only where a launch contract already requires them.** The
   single `.cl` edit in this pass is `reqd_work_group_size(16,16,1)` on the
   two `// RIG-VERIFY` kernels in `avscuda_conditional_rig.cl` (§6): those
   kernels already computed garbage under any other local size, so the
   attribute converts silent corruption into a clean enqueue error. It
   changes no arithmetic and the kernels stay `// RIG-VERIFY`.
3. **Fidelity first, always.** Every port in this repo transcribes upstream
   arithmetic lane-for-lane (scalar ports of vector CUDA kernels are
   deliberate, §4). Nothing in this pass reorders, retunes, or "optimises"
   any computation.

## 1. Work-group-size requirements (the MUST table)

`grep get_local_id/get_group_id/get_local_size/get_num_groups` over
`src/opencl` hits exactly 4 files. Everything not in this table uses only
`get_global_id` (+ guards) and accepts **any** local size:

| Kernels | File | Requirement | `reqd` decision |
|---|---|---|---|
| `ka_sum_pixels_f32`, `ka_sad_f32` | avscuda_conditional_rig.cl | local exactly (16,16): `tid = lx+ly*16`, `sbuf[256]`, tree from 128 | **APPLIED** `(16,16,1)` — matches upstream SUM_TH 16×16 (§6) |
| `kf_calculate_field_diff`, `kf_block_sum_max`, `kf_analyze_noise`, `kf_analyze_diff` | kfm_decombeucf.cl | local exactly (32,16): `tid = lx+ly*32`, `sbuf[512]`(+lanes), tree from 256 | recommend `(32,16,1)` — NOT applied (file is ALG-VERIFIED) |
| `kf_count_cmflags`, `kf_count_cmflags_2planes` | kfm_combinganalyze.cl | exactly 512 items/group (`sbuf[512*3]`, hard offsets +512/+1024, tree from 256); any 512-item shape is correct, 32×16 documented | recommend `(32,16,1)` — NOT applied (kernels ALG-VERIFIED) |
| `kt_pad_frame_h` | ktgmc_motion.cl | local_x **== hPad (runtime arg)** + exactly 2 groups on x; `x = lid0` is unguarded, so `ls0 > hPad` overwrites interior, `ls0 < hPad` under-fills | **cannot use `reqd`** (value is runtime) — host-side contract, already in HOST_CONTRACT.md §3/§6 |
| `kt_pad_frame_v` | ktgmc_motion.cl | mirror: local_y == vPad + exactly 2 groups on y | same — host-side contract |

Notes:

- Shape latitude: the (16,16)/(32,16) trees only need `tid` to be a
  permutation of `[0,N)` — e.g. local `(256,1)` would also be correct for
  the rig kernels. We pin upstream's 2-D shape deliberately (matches the
  CUDA twin's geometry; squarish groups coalesce better — arch. reasoning).
- `kt_copy_pad` (RIG-VERIFY) looks group-ish but is pure-`get_global_id`
  with an origin-shifted dst pointer — any decomposition; its RIG status is
  about the host origin/grid contract, not local size.
- Global-size over-cover tolerance (round-up launch) is a per-kernel
  property of guards and was **not** exhaustively audited here; file headers
  and HOST_CONTRACT.md §3 are authoritative for grids.

## 2. Per-file characteristics (structural)

Legend: mapping = work-item assignment; mem = dominant global-memory
pattern; **bold** = the file's main perf consideration.

| File (kernels) | Mapping | Mem pattern | Notes |
|---|---|---|---|
| ktgmc_simple.cl (26) | 1 thread/px, 2-D | row-contiguous, coalesced | `int2/int3` MV buffers: host layout hazard, probe first (HOST_CONTRACT §2). `kt_plane_sad`: **per-pixel `atomic_add` into one global** (§3). Rest: pure elementwise, any decomposition. |
| ktgmc_motion.cl (22) | mostly 1:1 1-D/2-D; `kt_degrain_patch` 3-D `(BS,BS,nBlk)`; `kt_mean/most_freq_mv` (1,nRows) | row-contiguous; degrain gathers ref blocks via MV | `kt_scene_change(_x2)`: data-conditional per-item atomics. (1,N) grids are one thread/row — trivially small kernels, fine. Pad kernels: §1. |
| kfm_filterbase.cl (19) | 1:1; `kf_copy(_2plane)` 3-D batch | contiguous | padh/padv border geometry is a RIG seam (host contract). No local use. |
| kfm_deblock.cl (9) | 1 thread/px(+block helpers) | contiguous rows; 8×8 block DCT traffic | Scalar port of uchar4 CUDA (lanes independent ⇒ identical, §4). Host mirror-pad seam (header). |
| kfm_deblock_rig.cl (2, RIG-VERIFY pinned) | 1:1 | contiguous | No local/group-id use — any decomposition. Texture-precision gap is a *correctness* seam (RIG_HANDOFF_KDEBLOCK.md), not perf. |
| kfm_combinganalyze.cl (15) | 1:1 except 2 census trees (§1) | contiguous; census reads 4 combe planes | Trees: `sbuf[512*3]` int = 6 KB local, 2×(1+8+1) barriers; **one global atomic per group per field, skipped when 0** — good shape. |
| kfm_decombeucf.cl (7) | 1:1 except 4 trees (§1) | contiguous; ±2-row padded reads | Local: 2 KB / 2 KB / 8 KB / 4 KB — small. u64 atomics via `cl_khr_int64_base_atomics` (+ 2.0-style spelling): **emulated/slow on some HW, and a portability seam** (file's own RIG-VERIFY note). Per-group single atomic — good shape. |
| kfm_deband/edgelevel/mergestatic/noiseclip/temporalnr (12 total) | 1 thread/px, 2-D | contiguous | Pure elementwise stencils/copies. No constraints. |
| avscuda_merge.cl (4) | 1:1 | contiguous | Scalar ports of uchar4/ushort4/float4 CUDA (§4). |
| avscuda_filters.cl (4) | 1:1, in-place | contiguous | Same scalarisation. No atomics, no local use. |
| avscuda_convert.cl (10) | 1:1 | contiguous | Dither LUT traffic is constant/small. |
| avscuda_conditional.cl (15 int, ALG-VERIFIED) | 1:1 | contiguous | **Per-item `atomic_add` straight into the global counter** (u32/u64 sum/sad — a documented departure from upstream's tree + 1-atomic/block, file header) **and per-item scattered-bin `atomic_add(&hist[idx],1)`** in `ka_count_hist_*` (contention spread over bins — milder). Correct (int assoc/comm) but the highest-contention file in the repo; the tree rewrite is future work (§7), deliberately not done here. |
| avscuda_conditional_rig.cl (2, RIG-VERIFY) | 1:1 + §1 tree | contiguous | Per-group single CAS-loop float atomic — good shape. Portable `__local` tree instead of shuffle: rounding differs from CUDA by design (handoff §3). 1 KB local. |
| avscuda_resample.cl (8) | 1 thread/px, 2-D | H: contiguous row taps, coalesced. **V: strided column taps** (`src[x+(yoff+i)*pitch]`, stride = pitch) | No `__local`/barrier anywhere (header: "no work-group-size constraints"). Coeff row broadcast per y (cacheable). V kernels are the classic transpose/tiling candidate (§7); `filter_size` unbounded ⇒ per-item loop cost scales linearly — host routes `filter_size == 1` elsewhere (file header). |

## 3. Atomics inventory (contention ranking, structural)

Ordered worst-first by atomic traffic per work-item. Integer atomics are
order-exact (no correctness concern); this is purely about contention:

1. **Per-item global add** (every item hammers one counter):
   `avscuda_conditional.cl` int sum/sad ×15, `kt_plane_sad`. Worst shape
   in the repo; correct, but expect serialisation on wide GPUs
   (arch. reasoning — measure before rewriting).
2. **Per-item scattered-bin add**: `ka_count_hist_*` (`atomic_add(&hist[idx],1)`
   over up to 64K bins). Still per-item atomic traffic, but contention is
   spread — milder than 1, worse than 3+.
3. **Conditional per-item add**: `kt_scene_change(_x2)` (only over-threshold
   blocks fire). Data-dependent; fine when sparse.
4. **Per-group single add/max** (tree then one atomic): all §1 tree kernels
   (rig, combing census, decombeucf). Good shape; keep.
5. **u64 atomics**: decombeucf `atomic_add(ulong)` via extension. Function +
   speed both vary by device — the file's own RIG-VERIFY note covers the
   fallback gap.

## 4. Vectorization notes

- **Scalar-by-design ports** (upstream used vector loads as a pure load
  optimisation; lanes independent ⇒ scalar is lane-identical): all of
  avscuda_merge/filters/convert, kfm_deblock, ktgmc_simple planes,
  conditional int + rig (`float4` → scalar px). This was the right fidelity
  call (PORT_PLAN: "Vectorization is a later, pure-performance pass") and
  this pass does not revisit it. If a device run ever shows these as
  bandwidth-bound, widen loads to `uchar4/float4` **without** changing lane
  arithmetic — then re-prove via the existing mirror+golden (they already
  cover the scalar form; the vector form must be shown lane-identical).
- **Structural vectors** (part of the data model, not an optimisation):
  ktgmc `int2/int3` motion vectors. The perf angle is nil; the interop
  angle is the `int3` padding hazard (HOST_CONTRACT.md §2 — probe on rig).
- **No implicit vectorization dependencies**: no kernel relies on the
  compiler auto-vectorizing scalar code for *correctness* (all loop bounds
  are scalar-exact). Any auto-vectorization is pure upside.

## 5. Local-memory / barrier inventory

`__local`/`barrier` appear (as code, not comments) in exactly 3 files —
all §1 tree reductions, all small:

| File | Max `__local` per group | Barriers per launch |
|---|---|---|
| avscuda_conditional_rig.cl | 1 KB (`float[256]`) | 8 + initial fill barrier |
| kfm_combinganalyze.cl | 6 KB (`int[512*3]`) | per census pass: 1 + 8 + 1 |
| kfm_decombeucf.cl | 8 KB (`int[4*512]`, analyze_noise) | 1 + 9 per tree kernel |

All ≤ 8 KB — no local-memory occupancy pressure on any real device
(arch. reasoning: minimum `CL_DEVICE_LOCAL_MEM_SIZE` is 16–32 KB).
No `__local` anywhere else was attempted or is needed: the stencil/copy
kernels are all single-pass elementwise.

## 6. Changes applied in this pass

One, and only one, `.cl` edit:

- `src/opencl/avscuda/kernels/avscuda_conditional_rig.cl`: added
  `__attribute__((reqd_work_group_size(16, 16, 1)))` to
  `ka_sum_pixels_f32` and `ka_sad_f32`, + a header comment noting the
  enforcement. Rationale: §1 — the kernels already required (16,16);
  without the attribute a wrong local size silently computed garbage,
  with it the enqueue fails cleanly. Arithmetic untouched; kernels stay
  `// RIG-VERIFY`; `lint` green (the C shim drops `__attribute__`, so the
  parse check is unaffected — real OpenCL C accepts this spelling per
  §6.8 of the spec).

Everything else in this file is documentary. In particular the
`reqd(32,16,1)` recommendations for the KFM tree kernels were NOT applied
(ALG-VERIFIED bytes stay stable — §0 rule 1).

## 7. Future work (needs a device; explicitly NOT done here)

1. Measure first: per-item-atomic kernels (§3.1) and strided V-resample
   (§2) are the only structural hotspots worth profiling.
2. Int-conditional tree rewrite (match upstream: `__local` tree + one
   atomic/block) — then re-run mirror+golden (int assoc ⇒ same proof).
3. V-resample tiling (transpose or `__local` column tile) — arithmetic
   identical, memory pattern only; re-prove via existing golden.
4. `reqd(32,16,1)` on the KFM tree kernels — attribute-only; needs a
   repo decision on whether attribute edits preserve ALG-VERIFIED (§0.1).
5. Subgroup-shuffle variant of the rig tree (matches CUDA pairing AND
   leaves scalar?) — portability tradeoff; keep the portable tree as the
   reference regardless.
6. Vector-load widening (§4) only if bandwidth-bound, lane-identical only.

---
*Pass conducted 2026-09-24: 18 files / 154 kernels inventoried
(`grep ^kernel/__kernel`), local-id users enumerated (§1), atomics ranked
(§3). No `.cl` semantics changed.*
