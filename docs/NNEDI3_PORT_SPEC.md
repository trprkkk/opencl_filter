# NNEDI3 port spec (KNNEDI3 device kernels → OpenCL)

Upstream: `rigaya/AviSynthCUDAFilters` @ `68aef6e`, submodule
`rigaya/NNEDI3` @ `01931aa` (`NNEDI3/nnedi3/nnedi3_kernel.cu`, GPL).
All line numbers below are 1-based at that commit. The file holds exactly
6 `__global__` kernels (771 lines); the rest of the submodule is host,
weights, and CPU (ASM/intrinsic) paths, all out of scope.

## 1. Census (6/6 kernels)

| # | CUDA kernel | Lines | OpenCL port | Batch | Status |
|---|---|---|---|---|---|
| 1 | `kl_pad_h` | 40–55 | `kn_pad_h` (`nnedi3_pad.cl`) | 1 | ALG-VERIFIED (600-case runner, §3) |
| 2 | `kl_pad_v` | 57–71 | `kn_pad_v` (`nnedi3_pad.cl`) | 1 | ALG-VERIFIED |
| 3 | `kl_copy` | 73–81 | `kn_copy` (`nnedi3_pad.cl`) | 1 | ALG-VERIFIED |
| 4 | `kl_pad_ref_and_copy_half` | 88–131 | `kn_pad_ref_and_copy_half` (`nnedi3_pad.cl`) | 1 | ALG-VERIFIED |
| 5 | `kl_prescreening` | 135–~330 | `kn_prescreening` (`nnedi3_prescreen.cl`, planned) | 2 | not started |
| 6 | `kl_compute_nn` | 332–~470 | `kn_compute_nn` (`nnedi3_compute.cl`, planned) | 3 | not started |

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

## 4. Batch-2/3 preview (not started)

- `kl_prescreening`: shared-mem staging (`sws[64]`, `swf[7]`), fixed
  `PRE_BLOCK_W/H` (32/16) geometry, writes `workNN` + `numblocks`
  (atomics TBD — read the tail before transcribing).
- `kl_compute_nn`: block-level (`bid`, `workoff`), shared `B` tile +
  float `avg`, weights as kernel args (`short2`/`float2` + pitches —
  synthetic weights suffice for verification; the 13.5 MB `binary1.bin`
  is never needed in-repo). Device math is f32 + int (no exp/tanh/double
  in the kernel), so the AvsCUDA `-ffp-contract=off` bit-exact method
  applies — but every float op must be audited for FMA-fusion sensitivity
  against nvcc defaults first.
