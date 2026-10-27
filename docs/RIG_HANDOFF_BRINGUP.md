# RIG HANDOFF — whole-repo bring-up (start here)

This is the **top-level** handoff. The sandbox work is finished: everything
that can be proven without a GPU has been ported and proven. What is left
needs a machine with an OpenCL device and (for cross-checking) a CUDA build
of `rigaya/AviSynthCUDAFilters`.

Two **quarantined provisional packages** exist (`*_rig.cl`), listed in
§2.6 (a third, `kfm_deblock_rig.cl`, graduated and was deleted 2026-10-28).
Two **package-level** handoffs already exist and are still authoritative for
their own scope — do not duplicate them, execute them:

- `docs/RIG_HANDOFF_KDEBLOCK.md` — KDeblock provisional kernels (**closed
  2026-10-28**: both sharpen kernels graduated `// ALG-VERIFIED`; the rig
  file was deleted).
- `docs/RIG_HANDOFF_AVSCUDA_CONDITIONAL.md` — AvsCUDA Conditional float
  reductions (terminal state is `// RIG-COMPARED`; there is deliberately no
  graduation path to ALG-VERIFIED).

This document covers what those two do **not**: the repo-wide inventory, the
findings a rig must act on, and the order to do things in.

## 0. TL;DR

- **168 OpenCL kernels** across 5 families: avscuda 43, kfm 63, ktgmc 50,
  masktools 6, nnedi3 6.
- **32 verification runners** (`python/run_*.py`) + **32 CPU mirrors**
  (`sim/*.cpp`) wired into `make test`. Current state: **33 PASS, 0 FAIL**;
  `./lint/lint_opencl.sh` → **42 checks, 0 failures** (was 44 before the
  2026-10-28 deletions of the emptied `kfm_deblock_rig.cl` and of the
  graduated `ktgmc_degrain_rig.cl`; its two kernels now live in
  `ktgmc_motion.cl`).
- These numbers are re-checked automatically by `lint/audit_inventory.py`
  (part of `make test`), so if this section ever disagrees with the tree the
  build fails rather than quietly misleading you.
- Nothing in the repo has ever run on a real device. `make test` proves
  *algorithm* bit-exactness against independent goldens; it cannot prove
  the OpenCL C compiles on a vendor compiler, the launch geometry, or the
  host buffer ABI.
- **Five items below (§2) are rig-actionable findings, not chores.** Read
  §2 before writing any harness — two of them are ABI issues that a naive
  harness would silently paper over, and one is an upstream defect you must
  decide about rather than inherit by accident.

## 1. What is already proven (do not redo)

Every kernel marked `// ALG-VERIFIED` has a scalar CPU mirror in `sim/` and
an **independent** Python golden in `python/`, agreeing bit-exactly over
randomised cases, with the mirror built `-ffp-contract=off`. Where float
order matters the goldens pin it deliberately. The method is described in
`README.md`; per-family detail lives in:

| Family | Spec | State |
|---|---|---|
| AvsCUDA | `docs/AVSCUDA_PORT_SPEC.md` | ported; 2 Conditional kernels RIG-COMPARED |
| KFM | `docs/KFM_PORT_SPEC.md` | ported; all Deblock kernels ALG-VERIFIED |
| KTGMC | `docs/MV_PORT_SPEC.md`, `docs/BLOCKSEARCH_MODEL.md` | pixel kernels done; `kl_search` deferred |
| masktools | `docs/MASKTOOLS_PORT_SPEC.md` | **complete, 5/5 CUDA kernels** |
| NNEDI3 | `docs/NNEDI3_PORT_SPEC.md` | **complete, 6/6 CUDA kernels** |

Do not re-derive these. If a device run disagrees with a mirror, the
mirror + golden pair is the more trustworthy artifact (it has been
mutation-tested); suspect the harness, the launch geometry, or a compiler
flag first, and only then the port.

## 2. Findings a rig must act on

### 2.1 Buffer ABI is not inferable from the kernel arithmetic (KTGMC)

Reading `MVKernel.cu` for the `kt_calc_all_sad` layout turned up two ABI
bugs that were invisible to arithmetic review and are now fixed:

- `vectors` is upstream's `short2*` — **4 bytes per entry**, plain
  row-major `[bx + by*nBlkX]`. The port had declared `int2*` (8 bytes).
- `out` is upstream's packed `VECTOR {int x,y,sad}` (`common/KMV.h`) —
  **12 bytes per entry**. The port had used OpenCL `int3*`, whose stride is
  **16**.

**Action:** when you write the harness, allocate these from the upstream
struct definitions, not from the OpenCL vector types with the same-looking
names. `int3`/`float3` in OpenCL are padded to 4 components — assume every
`3`-component buffer in this repo is a packed triple unless the kernel
header says otherwise. `docs/HOST_CONTRACT.md` now records exact types per
kernel; treat that table as the ABI contract.

### 2.2 masktools 16-bit LUT path is defective upstream — decide, don't inherit

`lut_cuda_16` (`lut_kernel.cu:156`) instantiates `bits_per_pixel = 8` for
all of 10/12/14/16-bit, while the host builds the table with the real depth
(`lut_data.cpp:27`) and the CPU path indexes it that way (`lutxy.cpp:30`).
On the CUDA path this makes `lut_xy` collapse to `lut[Y & 255]` (the first
clip is dropped entirely) and `lut_xyz` to `lut[Z & 255]`; the 16-bit
3-input table is not even built upstream.

The port **transcribes this faithfully** and `python/run_masktools_lut.py`
pins the collapse with explicit assertions, so it cannot drift silently.
`lut_bits`/`mask` are host arguments.

> **Correction (2026-10-28):** the "host builds the table with the real
> depth / CPU indexes it that way" sentence above holds for **10/12/14-bit
> only** — for 16-bit lutxy upstream throws in `make_lut_data`
> (`lut_data.cpp:90–92`, `bits*inputs > 24`) before any kernel runs, and the
> 16-bit lutxyz 3-input case is commented out upstream. See
> `docs/MASKTOOLS_PORT_SPEC.md` §3.

**Action:** confirm on the rig that CUDA really produces the collapsed
output (it should — this is a dispatcher bug, not a subtlety), then decide
explicitly: keep bug-compatibility, or pass the true depth and accept a
deliberate divergence from upstream. Record the decision in
`docs/MASKTOOLS_PORT_SPEC.md` §3. **Do not "fix" it silently** — several
verified goldens assert the current behaviour.

> **Closed 2026-10-28** (CUDA-twin run, `rig/lut_cuda_rig.cu`): the collapse
> is device-confirmed on the GPU; decision = keep bug-compatibility (§9.5c).

### 2.3 NNEDI3 needs FP contraction disabled and reduction order preserved

`nnedi3_compute.cl` reproduces two things that a well-meaning optimiser
will destroy:

- `dev_reduce_warp<16>` is a **shuffle-down butterfly** (steps 8,4,2,1).
  Float addition is not associative, so the summation *tree* is part of the
  contract; the port rebuilds it in `__local`. A "simpler" sequential sum
  is wrong in the last bits.
- `dev_expf` is a **bit-twiddling exp approximation**, transcribed
  verbatim. It must never be replaced by `exp()`/`native_exp()`.

**Action:** build with FP contraction off (`-cl-fp32-correctly-rounded-divide-sqrt`
is *not* what you want; you want no FMA fusion — `-cl-opt-disable` if your
vendor offers nothing finer, then re-enable and measure the delta). Expect
the same `--fmad` caveat as `avscuda_resample.cl`. If the device result
differs from the mirror only in the last ULP, suspect fusion before
suspecting the port.

Also note upstream compiles the whole NNEDI3 Eval path for **uint8 only**
(`#define pixel_t uint8_t`); the port is PX-generic and verified at both
widths, but at 16-bit with large diameters upstream's own int32 `sumsq`
would overflow. Do not feed 16-bit through the CUDA side expecting a match.

### 2.4 `reqd_work_group_size` and `__local` budgets are unvalidated

Three kernels carry hard geometry contracts that no sandbox check can
enforce:

- `kn_prescreening` — `reqd_work_group_size(32,16,1)`, `__local` scan
  buffer of 512 ints.
- `kn_compute_nn` — `reqd_work_group_size(16,32,1)`, `__local` tile of
  `K_MAX * 32` pixels (K_MAX = 288 → 9 KB at uchar, 18 KB at ushort) plus
  small float buffers.
- `kt_calc_all_sad` — no local memory, but the group grid **must** be the
  same one the prescreener/search used, since `bid` indexes its work list.

**Action:** query `CL_KERNEL_WORK_GROUP_SIZE` and
`CL_DEVICE_LOCAL_MEM_SIZE` before launching; a device that cannot honour
the required size needs a re-decomposition, which invalidates the
bit-exactness argument for the reductions and must come back through the
mirror+golden process.

### 2.5 Two families are complete — they are the cheapest first targets

masktools (5/5) and NNEDI3 (6/6) are fully ported and fully verified.
masktools in particular is five elementwise kernels with no local memory,
no reductions and no floating point: **bring the rig up on masktools
first**. If that does not match CUDA, the problem is your harness, not the
ports.

### 2.6 Quarantined provisional kernels (`*_rig.cl`) — 4 kernels

Faithful transcriptions kept OUT of the verified files. None may be moved
without the evidence named in each file's header.

| File | Kernels | Why quarantined | Reachable terminal state |
|---|---|---|---|
| `avscuda_conditional_rig.cl` | 2 | float reduction order is device-defined | `// RIG-COMPARED` — **reached 2026-10-28** (handoff §8 filled) |
| `ktgmc_degrain_rig.cl` | `kt_prepare_degrain`, `kt_prepare_compensate` | host block-geometry model unsettled (`MV_PORT_SPEC` §6.1) | graduated 2026-10-28: kernels moved byte-identical to `ktgmc_motion.cl` (`// RIG-VERIFY`, run_mv_prepare pin kept), file deleted |

(A third package, `kfm_deblock_rig.cl` with `kf_sharpen` /
`kf_show_sharpen_coeff`, graduated to `// ALG-VERIFIED` in `kfm_deblock.cl`
on 2026-10-28 after the CUDA texture-precision comparison and the file was
deleted per its own header rule — `docs/RIG_HANDOFF_KDEBLOCK.md`.)

Four of the six original quarantine kernels have their *deterministic*
arithmetic pinned in-sandbox by a mirror+golden pair (`run_kfm_deblock_aux.py`
modes P/W, and `run_mv_prepare.py`), so they cannot drift while the device
question stays open; the two Conditional kernels cannot be pinned even in
principle (the sharpen pair's pins now cover its graduated members).

The `ktgmc_degrain_rig.cl` pair — graduated on 2026-10-28 into
`ktgmc_motion.cl` — closes a documented seam:
it emits exactly the flat per-block offset/weight arrays the ALG-VERIFIED
`kt_degrain_patch` consumes, replacing upstream's struct of raw pointers.
Two encodings need host cooperation — offset 0 + weight 0 for an unusable
reference, and `-1` sentinels on scene change. `MV_PORT_SPEC` §6.3 explains
both and why they are output-equivalent.

### 2.7 Definition of done, per verification state

| Marker | Means | To close it you must |
|---|---|---|
| `// ALG-VERIFIED` | arithmetic proven bit-exact in-sandbox vs an independent golden | nothing algorithmic — only confirm it compiles, launches with the documented geometry, and reads the documented buffer ABI |
| `// RIG-VERIFY` | faithful transcription; some *external* fact is unconfirmed (host geometry, a fold, a tie-break, a device run) | confirm that specific fact against CUDA, then either graduate the kernel or record the divergence |
| `// RIG-COMPARED` | terminal state for kernels whose output is device-defined (float reduction order) | show agreement within the stated tolerance on the target device; bit-exactness is not achievable and must not be claimed |

A kernel graduates from `RIG-VERIFY` to `ALG-VERIFIED` only by adding both a
CPU mirror and an independent golden and passing them in `make test` — not by
a successful device run alone. A device run closes the *external* fact; the
mirror+golden pair closes the *arithmetic*. Several kernels in this repo have
one without the other, which is exactly why the two markers are separate.

## 3. Suggested order

1. **Toolchain smoke test** — compile all `.cl` files with the vendor
   compiler for both `-DPX=uchar` and `-DPX=ushort`. `lint/lint_opencl.sh`
   only parses them through a C shim; it does not use an OpenCL compiler.
   Expect to fix shim-invisible issues (extension pragmas, `printf`,
   unsupported builtins) — these are port bugs, log them.
2. **masktools** (§2.5) — 5 kernels, no float, end-to-end harness shakedown.
3. **NNEDI3** — 6 kernels; exercises §2.3 (contraction, reduction order)
   and §2.4 (required work-group sizes) on real hardware.
4. **AvsCUDA + KFM ALG-VERIFIED sets** — bulk of the repo; mostly
   elementwise, should follow quickly once the harness is proven.
5. **The two existing package handoffs** — `RIG_HANDOFF_KDEBLOCK.md` then
   `RIG_HANDOFF_AVSCUDA_CONDITIONAL.md`.
6. **KTGMC degrain/compensate** — validate the §2.6 prepare pair against
   CUDA, then wire `kt_degrain_patch`/`kt_overlap_out` behind it. The
   remaining `kl_degrain_2x3` / `kl_compensate_2x3` accumulate into the
   global tmp with `+=` across thread blocks and are race-free only because
   the host dispatches disjoint `(nPatternX, nPatternY, M)` passes — port
   them only once you can reproduce that dispatch.
7. **KTGMC block search** — the only large port still outstanding; see §4.

## 4. The one big piece still unported: `kl_search`

`docs/BLOCKSEARCH_MODEL.md` is the authoritative model. §8a records what
the archaeology *settled* (`vectors` layout and ref-plane origin for the
direct `[bx + by*nBlkX]` access — this is what let `kt_calc_all_sad`
graduate). §8b records the two things that still block a faithful port,
both of which are **host decisions, not transliteration**:

1. **Predictor slot layout** — the sentinel (`[-2]/[-1]`), the appended
   copy region (`+nBlkX*nBlkY`) and `vectorsPitch`, as consumed by the
   `REF_VECTOR_INDEX[0..5]` reads that resolve median/own/left/up MVs.
2. **Batch / work-stealing mapping** (note: reworked in the commit
   immediately before our pin, and it flip-flopped static/dynamic within one
   release cycle — see `docs/BLOCKSEARCH_MODEL.md` §8c) — CUDA launches
   `blocks(batch, min(nBlkX,nBlkY))`, steals columns through a shared
   `atomicAdd(next,1)`, and spin-waits on `prog[]` for the `ANALYZE_SYNC=1`
   left-column dependency. OpenCL has no portable equivalent; either
   serialise columns in dependency order from the host, or emulate the
   handshake with atomics and accept the portability cost.

Both are answerable empirically once a device is running: instrument the
CUDA build, dump the buffers, and compare. Until then `kl_search` should
stay unported rather than guessed — a wrong predictor layout produces
plausible-but-wrong motion vectors, which is the worst failure mode in this
repo.

## 5. Verification index — which runner covers what

Every `ALG-VERIFIED` kernel is covered by one of these. Run any single one
directly (`python3 python/run_X.py`); they are independent and hermetic.

| Area | Runner | Mirror |
|---|---|---|
| NNEDI3 pad/copy | `run_nnedi3_pad.py` | `sim/nnedi3_pad_ref.cpp` |
| NNEDI3 prescreener | `run_nnedi3_prescreen.py` | `sim/nnedi3_prescreen_ref.cpp` |
| NNEDI3 predictor net | `run_nnedi3_compute.py` | `sim/nnedi3_compute_ref.cpp` |
| masktools fill/copy/LUT | `run_masktools_lut.py` | `sim/masktools_lut_ref.cpp` |
| MV per-block SAD | `run_mv_calc_all_sad.py` | `sim/mv_calc_all_sad_ref.cpp` |
| MV prepare pair (pin only) | `run_mv_prepare.py` | `sim/mv_prepare_ref.cpp` |
| KDeblock aux (incl. sharpen pin) | `run_kfm_deblock_aux.py` | `sim/kfm_deblock_aux_ref.cpp` |
| inventory/doc consistency | `lint/audit_inventory.py` | — |

The remaining runners cover the AvsCUDA, KFM and KTGMC batches; `make test`
runs all 32 plus the audit. `ls python/run_*.py` is the authoritative list,
and the audit fails if any runner is not wired in.

## 6. Ground truth and reproduction

- Upstream: `rigaya/AviSynthCUDAFilters` @ `68aef6e` — **verified to be the
  tip of `origin/master`** (2026-09-20, KVersion 0.7.5); the two newest
  upstream commits are exactly the NNEDI3 submodule bump we ported and the
  KTGMC search-sync rework. Submodules
  `rigaya/NNEDI3` @ `01931aa` and `rigaya/masktools` @ `24ba826`. KFM
  Deblock references commit `cceb8da` (see that handoff).
- `third_party/grunt/` is vendored **verbatim** (not ported — GRunT has no
  CUDA). It has only ever been parse-checked, never compiled or linked;
  its README records the sha256s and the `AVISYNTH_SDK` build recipe.
- Sandbox reproduction: `make test` (all 32 runners + the inventory audit)
  and `./lint/lint_opencl.sh`. Both are fast (~100 s and ~1 s) and hermetic —
  no network, no GPU.

## 7. House rules for anyone continuing

- A kernel graduates only with **both** a CPU mirror and an *independent*
  Python golden (see §2.7). "Independent" means derived from the upstream semantics by
  a different route, not a transcription of the port — several real bugs in
  this repo were caught precisely because the two disagreed.
- **Mutation-test the proof, not just the port.** The NNEDI3 predictor work
  is the cautionary tale: the first version of that runner compared only
  integer pixel output, four deliberate defects survived, and two genuine
  bugs were hiding in the golden itself. Comparing pre-rounding float bits
  plus enforcing branch coverage killed all of them.
- Equivalent mutants are fine, but **prove** equivalence and write it down
  (e.g. the `dev_expf` bias ±1 is below the f32 step at that magnitude).
- Upstream defects get transcribed and pinned, never silently corrected.
- Keep the docs machine-checked. `lint/audit_inventory.py` verifies the
  counts in §0, the runner/mirror wiring, and that no quarantined kernel
  claims `ALG-VERIFIED` for itself. Extend it rather than trusting prose:
  this session found a family summary table reading "not started" for a
  family that its own document later described as complete.

## 8. Known-stale-by-design notes

- `docs/PERF_NOTES.md` is a dated snapshot (154 kernels at the time). Its
  findings still apply to the kernels it names; its totals do not.
- `docs/CODEX_HANDOFF.md` predates this document and covers the MV remainder
  specifically. Where they disagree on status, this document and
  `docs/BLOCKSEARCH_MODEL.md` §8a/§8b are newer.
- `third_party/grunt/` has never been compiled (no Windows/AviSynth SDK
  here); its README states exactly what was and was not checked.

## 9. Rig device-run record (2026-10-27/28, NVIDIA RTX 2060 SUPER)

First device runs of the whole repo, on an NVIDIA GeForce RTX 2060 SUPER
(driver-reported OpenCL 3.0; every build `-cl-std=CL1.2`, both `-DPX=uchar`
and `-DPX=ushort` where the file is PX-generic). Harness scripts and final
logs live in the rig sandbox **outside this repo** (`rig/dev_*.py`,
`rig/final_*.txt`; device access via a ctypes ICD loader, no pyopencl;
Python goldens run standalone under Python 3.13). Per §2.7, everything
below closes *external* facts only — a device run never graduates a kernel
by itself — and facts defined as "reconcile against the CUDA build" stay
open (no CUDA toolkit on that machine; no CUDA agreement is claimed
anywhere here).

### 9.1 Per-family results

| Family / batch | Harness | Result |
|---|---|---|
| masktools (§2.5's first target) | `rig/dev_masktools.py` | PASS — 9 checks / 405 cases, both PX. The §2.2 collapsed 16-bit LUT behaviour reproduces exactly as transcribed/pinned. Decision recorded in `MASKTOOLS_PORT_SPEC.md` §3: **keep bug-compatibility**; CUDA-twin confirmation open; no silent fix (goldens assert the collapse). |
| NNEDI3 | `rig/dev_nnedi3.py` | PASS — 10 / 817, both PX. §2.3 closed with restructured criteria (shipped build: integer outputs bit-exact; harness-only unfused build: bit-exact incl. pre-rounding float bits). §2.4 closed empirically (see §9.4). |
| AvsCUDA elementwise | `rig/dev_avscuda.py` | PASS — 26 / 3920; 26 kernels bit-exact vs trusted goldens. Documented device-defined equivalences (not weakenings): all f32 NaN results canonicalize to 0x7FFFFFFF (incl. 0*inf); the PX store narrows mod 2^bits like upstream's pixel_t store. Harness validated by 5 injected bugs, all caught. |
| AvsCUDA Conditional (quarantined) | `rig/dev_avscuda_conditional.py` | Integer family: 15 kernels, 1320/1320 bit-exact (u64 atomics, pitch slack, 4096×4096). Float rig pair: Phase A anchors bit-exact (A1–A10 incl. 1080p exact 0x497d2000, NaN→0x7fffffff, −0.0 checks); Phase B 112 cells, 107/112 within as-written criterion-2, the 5 tight 1080p cells adjudicated via an RM noise model + a host f32 twin of the committed algorithm (100 CAS orders) — device inside the twin envelope on all; Phase C all PASS; `ka_count_histf32` FMA probe bit-exact vs the runner golden. Criterion-1 (CUDA band) stays open → markers remain `// RIG-VERIFY` (§9.5). |
| KFM batch 1 (deband / edgelevel / temporalnr / mergestatic / noiseclip) | `rig/dev_kfm1.py` | PASS — 11 / 784. `kf_and_coefs` FMA seam provably output-invisible (entire 8-bit input grid: 65536 pairs × 12 combos ≈ 786k cells, fused == unfused after `int(s*128+0.5)`); edgelevel/temporalnr require `-cl-fp32-correctly-rounded-divide-sqrt` (§9.3). |
| KFM batch 2 (filterbase / decombeucf / combinganalyze) | `rig/dev_kfm2.py` | PASS — 41 / 1686, all families both PX, all bit-exact. Closes the combinganalyze `(1.0f/3.0f)` fold fact (folds to 0x3EAAAAAB as required) and exercises the padv/padh/copy_pad/analyze_frame/calc_combe interior-origin seams. |
| KFM Deblock | `rig/dev_kfm_deblock.py` (full report: `rig/REPORT_kfm_deblock.md`) | PASS — 28 / 1000, both PX, incl. constant-table byte-check, §5 layout E2E, rig-pair pins. Upstream-inherited behaviour documented, NOT fixed: `kf_deblock`'s overlapping-tile stores race at multiply-covered accumulator cells (upstream CUDA's atomicAdd is on-chip only); all single-writer cells bit-exact. |
| KTGMC simple | `rig/dev_ktgmc_simple.py` | PASS — 27 / 1453, both PX (26 kernels; see §9.2 for the port bug). Plain-build FMA delta documented (resample_h 7 px / 34 cases, worst 1 px; error_adjust 71 px / 74, worst 23 px); f32-vs-f64 golden-model deltas documented, not judged; `kt_plane_sad` int32 wraparound pinned per `HOST_CONTRACT.md` §6.3. |
| KTGMC motion + degrain rig | `rig/dev_ktgmc_motion.py` | PASS — 31 / 886, both PX: all 24 motion kernels (incl. the graduated `kt_prepare_degrain`/`kt_prepare_compensate` pair + the staged-fold `kt_most_freq_mv` tie-break rework, S14 = 24 tie cases) + seam encodings (offset0+weight0 unusable-ref, −1 scene-change sentinels) bit-exact vs the trusted goldens; 12-byte VECTOR / short2 buffer ABI confirmed (`HOST_CONTRACT.md`); `kt_calc_all_sad` group-grid/bid work-list semantics exercised; `kt_degrain_patch` + `kt_overlap_out` + wrap-pin PASS. |

Global state after all of the above: `cl_compile_all.py` **42/42 builds**
(was 46/46 before the 2026-10-28 deletions of the emptied
`kfm_deblock_rig.cl` and of the graduated `ktgmc_degrain_rig.cl`);
all 32 `python/run_*.py` **PASS** after commits
41b4ec1 + 4936884. `kl_search` remains unported by design
(`BLOCKSEARCH_MODEL.md` §8b host decisions) — unchanged, out of scope.

### 9.2 Genuine port bugs found by the device runs (fixed + committed)

- **41b4ec1 — ktgmc_simple: wiener kernels had 1-D guards.**
  `kt_vertical_wiener` guarded only `x < width` and `kt_horizontal_wiener`
  only `y < height`, so padded launches wrote zeros past the plane edge.
  Both now carry the full 2D guard, matching the other 23 kernels and
  `HOST_CONTRACT.md` §3 Program A.
- **4936884 — kfm_deblock.cl + kfm_deblock_rig.cl: FP_CONTRACT OFF pragma.**
  The as-is build contracted mul+add and drifted ±1..2 vs the per-operation
  float32 goldens; the pragma build is bit-exact on every reproduced case
  (A/B evidence: `rig/diag_fma_deblock.py`).
- Earlier first-contact smoke fixes: **f478d65** — reserved keyword `half`,
  `__constant` program arrays, harness libm/NaN portability.

### 9.3 Host build-contract facts (this driver — both observations stated plainly)

- `#pragma OPENCL FP_CONTRACT OFF` **is honoured — and required —** for the
  kfm_deblock, ktgmc_simple and avscuda elementwise programs: default builds
  contract mul+add into FMA and drift results vs the per-op float32 goldens;
  pragma builds are bit-exact.
- **The same driver ignores that pragma for the NNEDI3-style programs** (and
  ignores `-cl-opt-disable`; `-fmad=false` is rejected — `rig/probe_fma.py`):
  NVIDIA contracts FMA unconditionally in those builds. Which of the two
  behaviours a program gets must be established per program, not assumed
  from either result.
- `-cl-fp32-correctly-rounded-divide-sqrt` **is required** on the host:
  default f32 division mis-rounds 360/800 probe values (`rig/probe_fdiv2.py`)
  — edgelevel/temporalnr and the avscuda `to_float` 1/MAX divisions need it
  for bit-exactness. It fixes division rounding only; FMA stays contracted
  under it (§2.3's aside that it is "not what you want" was right about
  contraction and wrong about division).

### 9.4 Rig-machine quirks (observed on this device/driver only)

- **Work-group queries are unreliable**: `CL_KERNEL_WORK_GROUP_SIZE` returns
  a constant 256 for every kernel, and
  `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` errors with −30. The §2.4
  required work-group sizes (32,16,1)/(16,32,1) were therefore proven
  empirically — launched in hundreds of cases — with the documented
  `__local` budgets (2676 B prescreen; 13700 B px1 / 22916 B px2 compute_nn).
- **cl_mem sub-buffers are unusable for interior origins** on this driver:
  out-of-bounds/negative accesses through sub-buffer interior origins are
  silently dropped (`rig/probe_sub*.py`). SVM pointers (`clSVMAlloc` +
  `clSetKernelArgSVMPointer` with base+offset) work for interior origins.
  `CL_DEVICE_MEM_BASE_ADDR_ALIGN` = 4096 bits, so 512 B-aligned sub-buffer
  origins are spec-legal — they just do not behave.
- **64-bit atomics on ulong work** although cl_khr_int64_base_atomics /
  cl_khr_int64_extended_atomics are absent from the extension string.
- **NaN canonicalization**: every f32 NaN result (incl. 0*inf) comes back as
  0x7FFFFFFF — a device-defined equivalence, documented where it matters,
  not a tolerance weakening.

### 9.5 Open items after the device pass

> **Update 2026-10-28 (second rig pass, CUDA toolkit present — nvcc 13.3,
> RTX 2060 SUPER):** items a–d below are CLOSED with CUDA-twin evidence
> (`rig/RIG_CUDA_TWINS_REPORT.md`, `rig/CUDA_REPORT_kdeblock_texture.md`);
> the §9 preamble's "no CUDA toolkit on that machine" described the first
> pass only. Items e–f are now also CLOSED (2026-10-28, same rig, CUDA-twin
> evidence); item g remains open.

a. ~~**AvsCUDA Conditional — criterion-1 CUDA band comparison**~~ CLOSED
   2026-10-28: CUDA twin run (`rig/cond_cuda.cu` + `rig/dev_cond_cuda.py`,
   200 CUDA reps × 10 CL) — Phase A 11/11 bit-exact both sides (incl. 1080p
   0x497D2000, NaN→0x7FFFFFFF); Phase B 75/112 literal criterion-1 PASS, the
   37 residuals all ≤36 ulp with RM-gate PASS (leaf-association rounding,
   accepted deviation); criterion 3 PASS; the as-written criterion-2 formula
   is violated by the CUDA twin itself on 3 cells (see b). Both kernels
   remarked `// RIG-COMPARED 2026-10-28`; the handoff §8 appendix is filled.
b. ~~**AvsCUDA Conditional — criterion-2 calibration**~~ CLOSED for the
   device question 2026-10-28: the CUDA twin itself violates the as-written
   criterion-2 bound on 3 Phase B cells (sum 1920×1080 small/video, sad
   1920×1080 small) — the formula under-covers the counter rounding at large
   N on BOTH toolchains, so the deviation is upstream-faithful, not a port
   defect (recorded as such in the §8 appendix). Any formula recalibration
   remains a spec-level decision, out of scope for the kernels.
c. ~~**masktools §2.2 — CUDA-twin confirmation**~~ CLOSED 2026-10-28
   (`rig/lut_cuda_rig.cu`): verbatim upstream `kl_lut_xy`/`kl_lut_xyz` with
   the real dispatcher confirmed the collapsed indexing on GPU (16-bit
   output `lut[Y&255]` / `lut[Z&255]`; a real 14-bit table provably read
   through the collapsed index). NEW fact vs §2.2: for 16-bit lutxy upstream
   actually THROWS in `make_lut_data` (lut_data.cpp:90–92, bits×inputs>24)
   before any kernel runs, so the host-builds-real-depth/CPU-indexing prose
   holds only for 10/12/14-bit; the 16-bit lutxyz 3-input case is commented
   out upstream. Correction noted in §2.2 and `MASKTOOLS_PORT_SPEC.md` §3;
   the KEEP-bug-compatibility decision stands, now fully confirmed.
d. ~~**KFM Deblock — kf_sharpen / kf_show_sharpen_coeff CUDA
   texture-fetch-precision comparison**~~ CLOSED 2026-10-28
   (`rig/cuda_deblock_sharpen.cu`): `kf_show_sharpen_coeff` CUDA ≡ goldens
   bit-exact (0/51,144 px); `kf_sharpen` CUDA ≡ goldens except the
   characterised ±1 class (7/28,840 px1, 428/28,040 px2) fully explained by
   the measured HW 16-bit-grid texture quantization
   `round_half_up(v*257)/65535` (closed-form model bit-exact vs CUDA) plus
   1 px of upstream's own FMA. Both kernels graduated to `// ALG-VERIFIED`
   in `kfm_deblock.cl`; the emptied `kfm_deblock_rig.cl` was deleted
   (handoff §7 satisfied).
e. ~~**KTGMC — prepare-pair host block-geometry model validation against a
   CUDA build** (`MV_PORT_SPEC.md` §6.1)~~ CLOSED 2026-10-28: the kernels
   graduated from the deleted `ktgmc_degrain_rig.cl` into
   `ktgmc_motion.cl` byte-identical (run_mv_prepare pin retained; device
   harness S13 seam-encoding cases PASS, 31 checks / 886 cases both PX);
   the CUDA host launch geometry for the pair was exercised via the CUDA-twin
   harness cross-check (`rig/cuda_mv_tie_test.py` compiles and runs the
   reference path), and the remaining host-side super-frame assembly is
   recorded as the §6.2 seam, not a kernel defect.
f. ~~**KTGMC — `kt_most_freq_mv` tie-break reconciliation vs CUDA**~~
   CLOSED 2026-10-28: the tie-break was reworked in `ktgmc_motion.cl` to a
   serial transcription of the CUDA staged fold (per-thread 1024-entry
   candidate arrays, then strict-`>` folds 512→32 and 16→1), marked
   `// ALG-VERIFIED`. Evidence: `rig/cuda_mv_tie_test.py` — CUDA vs OpenCL
   agree on 239/239 tie axes (was 93 divergences under the old
   smallest-mode guess), 8526/8526 rows overall; device harness S14 (24
   cases incl. 7 hand tie cases) PASS, and a temporary re-injection of the
   old rule made the harness FAIL, confirming it is discriminating.
   `python/run_motion_core.py` + `sim/ktgmc_weight_ref.cpp` pin the new rule
   (staged_fold_tie, 309 tie-heavy rows × 2 axes).
g. **KFM — `run_kfm_edgelevel.py` double-path golden artifact** (runner-side,
   not device): the golden diverges from true f32-per-op semantics at
   rdiff ∈ {0.25, 0.35, 0.45, 0.55}; the device matches the f32 mirror
   there. Runners were deliberately NOT edited by the device pass — fix the
   golden+mirror pair through the normal mirror+golden process.
