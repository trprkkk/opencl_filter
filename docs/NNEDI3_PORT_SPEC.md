# NNEDI3 port spec (KNNEDI3 device kernels → OpenCL)

Upstream: `rigaya/AviSynthCUDAFilters` @ `68aef6e`, submodule
`rigaya/NNEDI3` @ `01931aa` (`NNEDI3/nnedi3/nnedi3_kernel.cu`, GPL).
All line numbers below are 1-based at that commit. The file holds exactly
6 `__global__` kernels (771 lines); the rest of the submodule is host,
weights, and CPU (ASM/intrinsic) paths, all out of scope.

> **Provenance note.** `01931aa` (2026-09-20) is the tip our pin points at.
> That commit — and the parent repo commit `68aef6e` that bumped the
> submodule to it — only changed **host SIMD** code ("SIMD定数を配列化して
> ロード時のAVX命令実行を防ぐ": array-ify SIMD constants to avoid executing
> AVX at load time). The CUDA file this port transcribes,
> `nnedi3/nnedi3_kernel.cu`, was last touched in 2025-05-07 (`091e594`,
> a Linux-build fix), so the device semantics captured here are current and
> were not affected by the recent activity.

## 1. Census (6/6 kernels)

| # | CUDA kernel | Lines | OpenCL port | Batch | Status |
|---|---|---|---|---|---|
| 1 | `kl_pad_h` | 40–55 | `kn_pad_h` (`nnedi3_pad.cl`) | 1 | ALG-VERIFIED (600-case runner, §3) |
| 2 | `kl_pad_v` | 57–71 | `kn_pad_v` (`nnedi3_pad.cl`) | 1 | ALG-VERIFIED |
| 3 | `kl_copy` | 73–81 | `kn_copy` (`nnedi3_pad.cl`) | 1 | ALG-VERIFIED |
| 4 | `kl_pad_ref_and_copy_half` | 88–131 | `kn_pad_ref_and_copy_half` (`nnedi3_pad.cl`) | 1 | ALG-VERIFIED |
| 5 | `kl_prescreening` | 135–238 | `kn_prescreening` (`nnedi3_prescreen.cl`) | 2 | ALG-VERIFIED (66-case runner + 7 mutants, §4) |
| 6 | `kl_compute_nn` | 331–475 | `kn_compute_nn` (`nnedi3_compute.cl`) | 3 | ALG-VERIFIED (28-case runner + mutants, §5) |

Launch wrappers: `CopyPadCUDA` (:491), `BitBltCUDA` (:512),
`PadRefAndCopyHalfCUDA` (:520, hpad=32/vpad=3). Caller passes the
INTERIOR origin (`refptr += vpad*refpitch + hpad*pixelsize`, nnedi3.cpp:1765).

## 2. Port conventions (this family)

- `PX` = uchar/ushort per the repo-standard dual build. Upstream's pad/copy
  launch wrappers hardcode `#define pixel_t uint8_t` and ignore `pixelsize`;
  the port stays PX-generic and both widths are verified (values pass
  through untouched; the runner covers 0..255 and 0..65535).
- Pitches in PIXELS (elements). Upstream's `*pitch4`/byte pitches are
  divided by 4/pixelsize at the host boundary.
- `vpixel_t` (uchar4/ushort4) scalarised: `kn_copy` is a 1:1 pixel copy
  (width in pixels = upstream width4×4); `kn_pad_ref_and_copy_half` keeps
  upstream's per-vector grid with 4 scalar loads/stores and an explicit
  lane reversal on x-mirror (upstream's `swap(x,w)+swap(y,z)`).
- Interior-origin pointers kept verbatim (`kn_pad_h/v` `ptr`, pad_ref
  `ref`): negative indices address the margins, exactly as upstream.
- Group-id idioms kept verbatim: `kn_pad_h` needs exactly 2 groups on x
  with local_x == hPad (runtime value — no `reqd` possible, same as
  `kt_pad_frame_h`); `kn_pad_v` mirrored. Production pads dwarf nothing
  here: reads are interior-only iff pad < dim (pad_h/v) or pad ≤ dim
  (pad_ref single reflection) — the runner constrains dims accordingly
  and production (full frames, pads 32/3) satisfies it trivially.

## 3. Batch-1 (pad/copy): what was verified

- `src/opencl/nnedi3/kernels/nnedi3_pad.cl` (4 kernels),
  `sim/nnedi3_pad_ref.cpp` (modes A/B/C/D), `python/run_nnedi3_pad.py`
  (600 cases: padh 150 over hPad ≤ 33 incl. production 32, padv 150 over
  vPad ≤ 8 incl. production 3, copy 100, padref 200 over hpad4 ≤ 8 /
  vpad ≤ 6 incl. production 8/3; slack pitches everywhere).
- In-place semantics: pad_h/v read interior-only and write margin-only,
  so the parallel update equals any serial order — the mirror applies it
  to a buffer copy, the golden to its own; both compared full-buffer.
- Liveness note (no verification impact): upstream's CopyPadCUDA/BitBltCUDA
  callers sit inside `#if 0` (nnedi3.cpp:1782–1788); the live path is
  PadRef → prescreening → compute_nn. pad_h/v + copy are ported anyway
  (one `#if` flip from live).

## 4. Batch-2 (prescreening): what was verified

`src/opencl/nnedi3/kernels/nnedi3_prescreen.cl` + `sim/nnedi3_prescreen_ref.cpp`
+ `python/run_nnedi3_prescreen.py` (66 cases, both depths, frames that
straddle the 32×16 group grid in both axes, slack pitches, all four
`range_mode` val_min/val_max pairs).

Resolved questions:

- **No atomics.** The compaction is a block-wide *inclusive* add-scan of
  each item's reject count over `tid` (`dev_scan`, ReduceKernel.cuh:448,
  warp-shuffle + shared fan-in), then `idx -= num` for the exclusive base.
  The port substitutes a Hillis-Steele scan in `__local`: integer addition
  is associative and the tid order is unchanged, so `workNN` comes out
  bit-identical, not merely equivalent. `numblocks[bid]` is written by the
  last item (`tid == 511`), whose post-increment `idx` is the group total.
- **Fixed geometry is a hard contract**: `reqd_work_group_size(32,16,1)`.
  Out-of-range items still participate (upstream inits `result` to
  `{1,1,1,1}` so they reject nothing yet keep the scan dense).
- **Host pointer offsets** (EvalCUDA:651): prescreening gets
  `ref - refpitch - 8` (pixels), pitches in 4-pixel vectors.
- **Arithmetic split**: exact int32 for the 48-tap neighbourhood dot;
  unfused f32 for scale+bias → `t/(|t|+1)` squash → 4 accumulations →
  bias. Same `--fmad` caveat as `avscuda_resample.cl` (documented in the
  kernel header).

Independence and adequacy of the proof:

- The golden does **not** transcribe upstream's asymmetric lane split
  (`x==0` takes 2 taps from lanes z,w; `x<4` takes 4; `x==4` takes taps
  14,15 from lanes x,y). It derives the equivalent flat form — row `y`
  consumes the 16 pixels from `xbase*4 + 2`, tap `j` at weight
  `(j + y*16)` — so agreement also proves the split was read correctly.
- Mutation-tested (7 deliberate mirror defects, all caught): lane swap in
  the `x==4` taps; bicubic 19→18; scan order reversed; `result <= 0` →
  `< 0`; `workNN.y` group-relative → absolute; dropped `num < 4` bicubic
  guard; `numblocks` off-by-one.
- The `<= 0` boundary needed dedicated cases: random weights never land on
  exactly zero, and the first `<`-mutant survived. Six cases now zero the
  output layer and set the four biases to `{+0.0, -0.0, +denorm, -denorm}`,
  which pins the comparison (and `-0.0 <= 0` rejecting) exactly.

## 5. Batch-3 (compute_nn): what was verified

`src/opencl/nnedi3/kernels/nnedi3_compute.cl` + `sim/nnedi3_compute_ref.cpp`
+ `python/run_nnedi3_compute.py` (28 cases = 14 shapes x both PX widths,
covering every upstream READ policy, both QUAL values, the NN ladder
16..256, and work-list lengths nb in {0,1,2,15,31,32,33,40,64,65} so the
`b` loop and its `b+ty >= nb` tail are exercised).

**The 70 template instantiations collapse to one kernel.** Upstream
instantiates QUAL{1,2} x NN{16,32,64,128,256} x READ{8x6,16x6,32x6,48x6,
8x4,16x4,32x4}. All seven ReadPixelNxM policies were shown to stage the
identical logical tile — `B[ty][k] == src[(k % xdia) + (k / xdia)*pitch]`,
`K = xdia*ydia` — differing only in which thread loads which element, so
the port takes qual/nn/xdia/ydia as runtime arguments with one strided
loader and a `__local` tile sized for the largest policy (K <= 288).

**Order-significant float reductions.** `dev_reduce_warp<16>` is a
shuffle-down butterfly (steps 8,4,2,1), so lane 0 gets a specific addition
tree; a sequential sum differs in the last bits. The port rebuilds the
same tree in `__local` (lanes 8..15 read outside their 16-lane row
upstream, but those partials never flow back into lane 0, so clamping the
read is equivalent). `dev_expf` is transcribed verbatim — it must not be
replaced by `exp`/`native_exp`. Upstream's `(float)(1.0/(double)K)` is
spelled `1.0f/(float)K` to avoid needing fp64; not a blanket identity, so
it was checked exhaustively over every reachable K and qual.

### How the proof was made adequate (three real defects, all in the test)

The first version compared only the written pixels and **four mutants
survived**, because the integer output rounds ULP-level differences away —
including the two claims this port rests on (the reduction tree and
`dev_expf`). Fixes:

1. The mirror now also emits the **bit pattern of the pre-rounding float**
   `result * (1/qual)` per written pixel, and the golden compares it. This
   immediately exposed two genuine golden bugs: (a) the int reduction was
   going through the float tree helper, silently rounding `sumsq` above
   2^24, and (b) the golden multiplied by the raw `rng.uniform` doubles
   while the mirror receives f32 bit patterns — both now fixed (ints
   reduce exactly; weights are rounded with `F()` at generation). C's
   int-to-float conversion before a multiply is likewise applied
   explicitly.
2. The runner **asserts branch coverage** and fails on any unreached
   branch: `wsum > 1e-10` both ways, `var_ <= FLT_EPSILON` both ways,
   clamping at both ends, and `dev_expf` both saturated and free. Two
   dedicated regimes force them (a flat plane for zero variance; a -200
   bias with the scale term zeroed so `dev_expf` saturates low and `wsum`
   falls under the threshold).
3. Re-run mutants: sequential sum, reversed tree steps, `dev_expf` clamp
   +-80 -> +-81, `(5*v)/w` re-association, float-accumulated `sumsq`,
   dropped squash, dropped rounding, transposed tile, wrong weight
   striding — **all now caught**.

Two mutants are provably **equivalent**, not gaps, and were left alone:

- `dev_expf` bias `1064866805.0f` -> `...804.0f`: at 2^30 the f32 step is
  128, so both literals are the same float.
- `var_ <= FLT_EPSILON` -> `<`: for integer tiles the computed variance is
  either exactly 0 (flat tile) or >= ~1/K^2, never within a ULP of
  1.19e-7, so the boundary is unreachable. (`wsum > 1e-10` -> `>=` is the
  same story; the branch itself is covered, only the exact-equality point
  is unreachable.)

### Upstream notes worth keeping

- The whole Eval path is compiled for **uint8 only** (`#define pixel_t
  uint8_t`, :482). The port is PX-generic and verified at both widths, but
  16-bit is beyond upstream's reach; at 16-bit with large diameters
  upstream's int32 `sumsq` would overflow, so the runner keeps pixel
  magnitudes inside int32 for both accumulators.
- Host preconditions: `ref` offset by
  `-(((ydia>>1)-1)*refpitch + ((xdia>>1)-1))`, weights `wf = &ws[NN*K]`,
  pitches `weights1pitch/2` and `/4`, and the SAME group grid the
  prescreener used (bid indexes its work list).

## 6. Family status

All 6 NNEDI3 device kernels are ported and ALG-VERIFIED. Remaining work is
rig-side only: build the three `.cl` files with a real OpenCL compiler,
check the `reqd_work_group_size` and `__local` budgets on the target
device (the compute tile is K_MAX*32 pixels), and compare against CUDA
output with rounding tolerance where FP contraction may differ.
