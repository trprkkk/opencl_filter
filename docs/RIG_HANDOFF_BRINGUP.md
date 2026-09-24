# RIG HANDOFF — whole-repo bring-up (start here)

This is the **top-level** handoff. The sandbox work is finished: everything
that can be proven without a GPU has been ported and proven. What is left
needs a machine with an OpenCL device and (for cross-checking) a CUDA build
of `rigaya/AviSynthCUDAFilters`.

Three **quarantined provisional packages** exist (`*_rig.cl`), listed in
§2.6. Two **package-level** handoffs already exist and are still authoritative for
their own scope — do not duplicate them, execute them:

- `docs/RIG_HANDOFF_KDEBLOCK.md` — KDeblock provisional kernels (2 sharpen
  kernels remain; terminal state `// ALG-VERIFIED` is reachable).
- `docs/RIG_HANDOFF_AVSCUDA_CONDITIONAL.md` — AvsCUDA Conditional float
  reductions (terminal state is `// RIG-COMPARED`; there is deliberately no
  graduation path to ALG-VERIFIED).

This document covers what those two do **not**: the repo-wide inventory, the
findings a rig must act on, and the order to do things in.

## 0. TL;DR

- **166 OpenCL kernels** across 5 families: avscuda 43, kfm 63, ktgmc 48,
  masktools 6, nnedi3 6.
- **31 verification runners** (`python/run_*.py`) + **31 CPU mirrors**
  (`sim/*.cpp`) wired into `make test`. Current state: **32 PASS, 0 FAIL**;
  `./lint/lint_opencl.sh` → **44 checks, 0 failures**.
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
| AvsCUDA | `docs/AVSCUDA_PORT_SPEC.md` | ported; 2 Conditional kernels are RIG-only |
| KFM | `docs/KFM_PORT_SPEC.md` | ported; 2 sharpen kernels RIG-only |
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

**Action:** confirm on the rig that CUDA really produces the collapsed
output (it should — this is a dispatcher bug, not a subtlety), then decide
explicitly: keep bug-compatibility, or pass the true depth and accept a
deliberate divergence from upstream. Record the decision in
`docs/MASKTOOLS_PORT_SPEC.md` §3. **Do not "fix" it silently** — several
verified goldens assert the current behaviour.

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

### 2.6 Quarantined provisional kernels (`*_rig.cl`) — 6 kernels

Faithful transcriptions kept OUT of the verified files. None may be moved
without the evidence named in each file's header.

| File | Kernels | Why quarantined | Reachable terminal state |
|---|---|---|---|
| `avscuda_conditional_rig.cl` | 2 | float reduction order is device-defined | `// RIG-COMPARED` only |
| `kfm_deblock_rig.cl` | `kf_sharpen`, `kf_show_sharpen_coeff` | CUDA texture-fetch precision | `// ALG-VERIFIED` |
| `ktgmc_degrain_rig.cl` | `kt_prepare_degrain`, `kt_prepare_compensate` | host block-geometry model unsettled (`MV_PORT_SPEC` §6.1) | `// ALG-VERIFIED` |

Four of the six have their *deterministic* arithmetic pinned in-sandbox by
a mirror+golden pair (`run_kfm_deblock_aux.py` modes P/W, and
`run_mv_prepare.py`), so they cannot drift while the device question stays
open; the two Conditional kernels cannot be pinned even in principle.

The `ktgmc_degrain_rig.cl` pair is the one that closes a documented seam:
it emits exactly the flat per-block offset/weight arrays the ALG-VERIFIED
`kt_degrain_patch` consumes, replacing upstream's struct of raw pointers.
Two encodings need host cooperation — offset 0 + weight 0 for an unusable
reference, and `-1` sentinels on scene change. `MV_PORT_SPEC` §6.3 explains
both and why they are output-equivalent.

### 2.5 Two families are complete — they are the cheapest first targets

masktools (5/5) and NNEDI3 (6/6) are fully ported and fully verified.
masktools in particular is five elementwise kernels with no local memory,
no reductions and no floating point: **bring the rig up on masktools
first**. If that does not match CUDA, the problem is your harness, not the
ports.

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
2. **Batch / work-stealing mapping** — CUDA launches
   `blocks(batch, min(nBlkX,nBlkY))`, steals columns through a shared
   `next` counter, and spin-waits on `prog[]` for the `ANALYZE_SYNC=1`
   left-column dependency. OpenCL has no portable equivalent; either
   serialise columns in dependency order from the host, or emulate the
   handshake with atomics and accept the portability cost.

Both are answerable empirically once a device is running: instrument the
CUDA build, dump the buffers, and compare. Until then `kl_search` should
stay unported rather than guessed — a wrong predictor layout produces
plausible-but-wrong motion vectors, which is the worst failure mode in this
repo.

## 5. Ground truth and reproduction

- Upstream: `rigaya/AviSynthCUDAFilters` @ `68aef6e`, submodules
  `rigaya/NNEDI3` @ `01931aa` and `rigaya/masktools` @ `24ba826`. KFM
  Deblock references commit `cceb8da` (see that handoff).
- `third_party/grunt/` is vendored **verbatim** (not ported — GRunT has no
  CUDA). It has only ever been parse-checked, never compiled or linked;
  its README records the sha256s and the `AVISYNTH_SDK` build recipe.
- Sandbox reproduction: `make test` (all 31 runners) and
  `./lint/lint_opencl.sh`. Both are fast (~90 s and ~1 s) and hermetic —
  no network, no GPU.

## 6. House rules for anyone continuing

- A kernel graduates only with **both** a CPU mirror and an *independent*
  Python golden. "Independent" means derived from the upstream semantics by
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
