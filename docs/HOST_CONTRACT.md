# On-rig OpenCL host bring-up contract

This documents exactly what a device host runner (`src/host/run_opencl.cpp`) must
set up to compile and exercise the kernels in
`src/opencl/ktgmc/kernels/ktgmc_simple.cl` and `ktgmc_motion.cl`, and what must
be cross-checked on the rig. It is written against the committed `.cl`
signatures (see the file headers; grep `^kernel void`).

## 0. Why this exists

The sandbox that produced the kernel ports has **no OpenCL implementation**, so
the `.cl` files could only be validated two other ways:

- **CPU mirrors + independent Python goldens** (`make test`) prove the *integer
  algorithm* bit-for-bit. This covers every kernel marked `ALG-VERIFIED`.
- **`make lint`** does a genuine *C structural parse* of every `.cl` (via
  `lint/oc_shim.h`) to catch syntax/typo errors only.

A real OpenCL frontend/device is still required to (a) confirm each kernel
*compiles* as OpenCL C and (b) confirm runtime results. This file is the
specification for that host program. §1–§3 must be reproduced; §4–§6 are notes
and known risks to resolve on the rig.

## 1. Program construction

Build **two programs** from the two `.cl` files, and build each **twice** (the
sources are `-DPX` template units): `-DPX=uchar -DPX_MAX=255` and
`-DPX=ushort -DPX_MAX=65535`. If any build fails, print the
`clGetProgramBuildInfo(CL_PROGRAM_BUILD_LOG)` and treat as a kernel-port bug.

- Program A: `ktgmc_simple.cl`
- Program B: `ktgmc_motion.cl`

Get a context/queue on one device (prefer `CL_DEVICE_TYPE_GPU`; fall back to CPU
for bring-up). OpenCL 1.2 API is sufficient.

## 2. Buffer element types — the one real hazard

The MV kernels address buffers typed `int2`, `int3` (motion vectors), and plain
`int`/`PX`. **Host interop layout of `int2`/`int3` is implementation-defined in
OpenCL** (a 3-component vector may or may not be padded to 16 bytes), so a host
program that reads/writes these buffers by copying `count * sizeof(struct {int
x,y[,z];})` may misalign on some runtimes.

**Do this first on the rig** with a throwaway kernel compiled into program B:

```c
kernel void kt_meta_type_sizes(__global int* out){
  int i = (int)get_global_id(0);
  if (i == 0) out[0] = (int)sizeof(int2);
  if (i == 1) out[1] = (int)sizeof(int3);
  if (i == 2) out[2] = (int)sizeof(int4);
}
```
Read back `out[0..2]`. Expect sizes `8 / 12 (or 16) / 16`. The intended logical
layout is one contiguous `(x,y)` per `int2` element and `(x,y,sad)` per `int3`
element. If `sizeof(int3)==16` on the runtime, keep the *host* reference array
packed (12-byte stride) and copy element-by-element, or switch the affected
kernels to three separate `int` arrays — do not guess. The validators in
`sim/*.cpp` and `python/*.py` use the packed logical layout.

## 3. Kernel → grid and argument maps

`PX` = `uchar` or `ushort` per the program instance chosen. A `pitch` argument is
always in **samples** (elements of `PX`), a `vectorsPitch` in MV elements.
CUDA `blockIdx.z` batching is **not** in the OpenCL kernels: the host passes a
pointer to one batch's data (or an offset sub-buffer). "grid" below is the total
global size; each kernel also guards `x<… && y<…`.

### Program B — `ktgmc_motion.cl`

| kernel | grid | args (order) | notes |
|---|---|---|---|
| `kt_copy_pad` | 2D `(width+2·hPad, height+2·vPad)` | dst,`dst_pitch`, src,`src_pitch`, `hPad,vPad,width,height` | dst pointer must be at padded-interior origin (see comment); edge-mirror. RIG-VERIFY. |
| `kt_pad_frame_h` | 2 groups x × rows groups | ptr,`pitch`,`hPad,width,height` | in-place; group 0 = left pad, group 1 = right. RIG-VERIFY. |
| `kt_pad_frame_v` | x groups × 2 groups y | ptr,`pitch`,`vPad,width,height` | in-place; group 0 = top, 1 = bottom. RIG-VERIFY. |
| `kt_write_default_mv` | 1D `nBlkCount` | dst:`int3*`, `nBlkCount`, `verybigSAD` | dst[i]=(0,0,verybigSAD). |
| `kt_init_scene_change` | 1D `nFlags` | sceneChange:`int*` | zeroes `sceneChange[x]`. |
| `kt_scene_change` | 1D `nBlks` | mv:`const int3*`, `nBlks`,`nTh1`, sceneChange:`int*` | `atomic_add(sceneChange,1)` when `mv[].z>nTh1`; host must pre-zero. |
| `kt_scene_change_x2` | 1D `nBlks` | mv0,mv1:`const int3*`,`nBlks`,`nTh1`, sc0,sc1:`int*` | two independent counts. |
| `kt_short_to_byte` | 2D `(width,height)` | dst:PX* `dst_pitch`, tmp:`const int*` `tmp_pitch`, `width,height`,`shift` | out = clamp(tmp>>shift,0,PX_MAX). |
| `kt_short_to_byte_or_copy_src` | 2D `(width,height)` | pflag:`const int*`, dst:PX*,`dst_pitch`, src:`const PX*`,`src_pitch`, tmp:`const int*`,`tmp_pitch`,`width,height`,`shift` | pflag[0]!=0→convert, else copy src. |
| `kt_interpolate_prediction` | 2D `(nDstBlkX,nDstBlkY)` | src_vector:`const int2*`, src_sad:`const int*`, dst_vector:`int2*`, dst_sad:`int*`, `nSrcBlkX,nSrcBlkY,nDstBlkX,nDstBlkY,normFactor,normov,atotal,aodd,aeven` | src/dst buffers hold a single batch's packed blocks (row stride `nSrcBlkX`/`nDstBlkX`). |
| `kt_mean_global_mv` | 2D `(1, nRows)` | vectors:`const int2*`,`vectorsPitch`,`nVec`, globalMVec:`int2*` | row r at `vectors + r*vectorsPitch`. |
| `kt_most_freq_mv` | 2D `(1, nRows)` | vectors:`const int2*`,`vectorsPitch`,`nVec`,`isY`, globalMVec:`int2*` | mode seed; RIG-VERIFY tie-break (see §5). |
| `kt_calc_all_sad` | 2D `(nBlkX,nBlkY)` | pSrcY/U/V, pRefY/U/V: PX planes (pointers at the block-grid origin of a padded super-frame; negative MVs read before it), vectors:`const short*` (2 shorts/block, row-major `[bx+by*nBlkX]`, matching upstream `short2*`), dst_sad:`int*`, out:`int*` (3 ints/block = packed 12-byte `VECTOR{x,y,sad}`, NOT `int3`), `nBlkX,nBlkY,nPad,BLK_SIZE,NPEL,chroma,nPitchY,nPitchUV,nImgPitchY,nImgPitchUV` | per-block SAD of src block vs MV-selected ref block (NPEL sub-pel stack). **ALG-VERIFIED** (`python/run_mv_calc_all_sad.py`); layout resolved in `docs/BLOCKSEARCH_MODEL.md` §8a. |
| `kt_prepare_search` | 2D `(nBlkX,nBlkY)` | scalar block `nBlkX,nBlkY,nBlkSize,nLogScale,nLambdaLevel,lsad,penaltyZero,penaltyGlobal,penaltyNew,nPel,nPad,nBlkSizeOvr,nExtendedWidth,nExtendedHeight`, then vectors:`const int2*`, sads:`const int*`, vectors_copy:`int2*`, dst_data:`int*`(stride 12/blk), dst_dataf:`int*`(stride 5/blk), prog:`int*`, next:`int*` | `dst_data+dst_dataf` replace CUDA `SearchBlock`; `prog` length `nBlkX`, `next` scalar. |
| `kt_rb2b_bilinear_filtered` | 2D `(nWidth, nHeight)` | src:`const PX*` `src_pitch`, dst:PX* `dst_pitch`, `nWidth`,`nHeight` | 1:2 downsample; src plane is 2·nWidth × 2·nHeight. |
| `kt_rb2b_bilinear_filtered_with_pad` | 2D `(nWidth+2·hpad, nHeight+2·vpad)` | src:`const PX*` `src_pitch`, dst:PX* `dst_pitch`, `nWidth,nHeight,hpad,vpad` | fused single-round 1:2 downsample that also fills dst pad border; dst pointer at padded-interior origin (writes dstx=−hpad..). Source needs 2·nWidth × 2·nHeight rows. |
| `kt_load_mv` | 1D `nBlk` | in:`const int3*`, vectors:`int2*`, sads:`int*`, `nBlk` | |
| `kt_store_mv` | 1D `nBlk` | dst:`int3*`, vectors:`const int2*`, sads:`const int*`, `nBlk` | |
| `kt_load_mv_batch` | 1D `nBlk` | out:`int3*`, src:`const int3*`, vectors:`int2*`, sads:`int*`, `nBlk` | per-batch MV split; out[x]=src[x] copy-through. |
| `kt_degrain_patch` | 3D `(nBlkSize, nBlkSize, nBlkX*nBlkY)` | src:PX* `src_pitch`, `nBlkX,nBlkY,nBlkSize,stepX,stepY,delta`, WSrcArr:`int*`(nBlk), WFArr/WBArr/refBaseF/refBaseB:`int*`(delta·nBlk), refFPlane/refBPlane:PX* `refF_pitch/refB_pitch`, patch:PX* | per-block Degrain1to6_C patch value; refBase indexed `k*nBlk+blk` (rig seam). ALG-VERIFIED math. |
| `kt_overlap_out` | 2D `(width,height)` | src:PX* `src_pitch`, patch:PX* `patch_stride`, `nBlkX,nBlkY,nBlkSize,stepX,stepY,overlapX,overlapY`, winBase:`short*` `win_stride`, `width,height`, dst:PX* `dst_pitch` | feathered Overlaps_C + Short2Bytes per output pixel + edge src copy. ALG-VERIFIED math. |
| `kt_init_const_vec` | 2D `(2, nRows)` | vectors:`int2*`,`vectorsPitch`, globalMV:`const int2*`, `nPel` | slot `-2`=(0,0) if gid0==0 else slot `-1`=globalMV·nPel at row base. Host must leave 2 sentinel slots before each row. |

### Program A — `ktgmc_simple.cl`

25 per-plane kernels (see `docs/MV_PORT_SPEC.md` §6 for the list). All are 2D
`(planeWidth, planeHeight)` guards over PX planes with integer `pitch` in
samples. The exact per-kernel argument lists and arithmetic are validated by the
CPU mirror `sim/ktgmc_cpu_ref.cpp` and the `build/val/*` vectors produced by
`python/run_validation.py`; reuse those input planes and expected outputs for the
rig diff (identical inputs → identical outputs, including the 32-bit SAD
overflow behaviour of `kt_plane_sad`). KGaussResize needs no extra kernels:
it launches `kt_resample_v/h` at fir 8/9 with gaussian programs — reuse the
`gres_*` planes and `gprog_*` tables (offsets + float coef bits) from
`build/val` as the rig vectors.

## 4. Recommended rig self-test

For each `ALG-VERIFIED` kernel above: fill inputs deterministically, dispatch
with the grid from §3, read back, and compare to the corresponding CPU mirror /
Python golden (`sim/*_ref.cpp`, `python/run_*.py`). Start with:
`kt_write_default_mv`, `kt_init_scene_change`+`kt_scene_change`,
`kt_load_mv`+`kt_store_mv` round-trip, `kt_interpolate_prediction`,
`kt_mean_global_mv`, `kt_prepare_search`, then the simple-plane suite from
`build/val`. Remove the kernel's `// RIG-VERIFY` marker once it passes.

## 5. Kernels intentionally NOT yet ported (deferred to full host assembly)

The warp/shared `kl_search` block loop and the `kl_degrain_*`/`kl_compensate_*`
block kernels. These additionally require the `MV.cpp` host state machine +
super-frame sub-pel layout and are out of scope of this bring-up doc (see
`docs/BLOCKSEARCH_MODEL.md` §8 for the exact remaining items).

`kt_most_freq_mv` IS ported but carries a **RIG-VERIFY** marker: it returns the
smallest most-frequent component. This is bit-exact vs CUDA whenever the mode is
unique (simulation-confirmed) and only diverges when several values tie for the
mode, in which case CUDA's pick is an artifact of its 1024-thread reduction tree
— reconcile on the rig only if bit-exact tie output matters.

## 6. Risks / open questions to resolve on rig

1. `int2`/`int3` host buffer layout (see §2) — run the size probe first.
2. `kt_copy_pad`/`kt_pad_frame_*` use group-id/local-id launch idioms that must
   be replicated with the exact grid in §3; they are RIG-VERIFY for this reason.
3. `kt_plane_sad` intentionally preserves 32-bit integer SAD overflow.
4. Licensing: KTGMC (incl. the MV engine) is GPL upstream; keep derived files
   under matching terms.
