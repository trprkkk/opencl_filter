# Handoff — remaining device-bound KTGMC MV work (for a follow-up agent / rig)

Everything in this repo that is *verifiable in this sandbox* (no OpenCL device)
has been ported and `ALG-VERIFIED`/`RIG-VERIFY`-marked. The work left below is
**device-bound**: it needs (a) a real OpenCL/CUDA rig to run and cross-check
against `AviSynthCUDAFilters`, and (b) the `MV.cpp` host state machine. This doc
tells the next agent exactly what is already built and verified, what remains,
and the single seam where the two meet.

## Already built + verified (don't redo)

All in `src/opencl/ktgmc/kernels/ktgmc_motion.cl` (22 `kt_*` kernels) unless
noted; every `// ALG-VERIFIED` kernel has a CPU mirror + Python golden under
`make test`.

- **Ref-block element offset** (the crux seam): `kt_ref_block_offset(vx,vy,
  nPitch,nImgPitch,NPEL)` is a faithful port of `dev_get_ref_block` for the
  NPEL×NPEL sub-pel super-plane stack (NPEL ∈ {1,2,4}). RIG-VERIFY (pure fn).
- **Degrain/compensate pixel-combiner core** (`kt_degrain_patch` +
  `kt_overlap_out`): faithful, ALG-VERIFIED arithmetic of the whole overlap
  path (`Degrain1to6_C` `>>8` denoise, `Overlaps_C` feathered `(v*w+256)>>6`
  window accumulation, `Short2Bytes` `>>5`/`>>11`, block-geometry = the MV.cpp
  CPU basis `StepX = nBlkSize-overlap`, `wby/wbx` 9-window selection, edge
  source-copy). Verified vs the MV.cpp-staging mirror `sim/ktgmc_degrain_ref.cpp`
  (`python/run_mv_degrain.py`, 300 cases, 8/16-bit, overlap 0 & half).
  **They take the per-block reference-plane element offsets as resolved inputs
  (`refBaseB/refBaseF`, indexed `k*nBlk+blk`)** — that is the only thing left
  to wire.
- Weight helpers `kt_degrain_weight`, `kt_norm_weights` (ALG-VERIFIED); MV
  I/O `kt_load_mv_batch`/`load`/`store`/`init_const_vec` (ALG-VERIFIED);
  scene-change, search-prep `kt_prepare_search` (ALG-VERIFIED), block pure
  helpers `kt_clip_mv/check_mv/sq_norm`, super builders RB2B + `_with_pad`,
  `kt_calc_all_sad` (block-level SAD, RIG-VERIFY), `kt_most_freq_mv`/mean_global_mv,
  interpolate_prediction, padders, etc.

## The single seam to wire on the rig (layer B → C)

`kt_degrain_patch` needs, per reference plane k and block blk, the element
offset into that plane where the block's patch starts. MV.cpp computes exactly
this for the CPU path:

```cpp
// KMDegrainCore::Proc overlap branch + use_block + KMPlane:
block.x = bx*StepX;  block.y = by*StepY;        // StepX = nBlkSizeX - nOverlapX
blx = block.x * nPel + mvB[k][blk].x;           // (compensate: * time256/256)
bly = block.y * nPel + mvB[k][blk].y;
base = pPlane_GetPointer(blx >> nLogxRatio, bly >> nLogyRatio);
```
where `KMPlane` (`SetTarget`) stacks the `nPel*nPel` sub-pel planes `nPitch *
nExtendedHeight` apart (`pPlane[i] = base + i*nPitch*nExtendedHeight`) and
`GetPointer(nX,nY)` = decompose `nX,nY` into plane index (low `log2 nPel`
bits) + integer pixel, then add `nHPadPel/nVPadPel` padding. Feeding each such
resolved base (converted to a flat element offset) into `kt_degrain_patch` as
`refBaseB/refBaseF` gives the true degrain output. `kt_overlap_out` then needs
only the plane geometry + the 9 feathered windows (`OverlapWindows`, each
`nBlkSize*nBlkSize`, produced host-side — pure cosine, device-independent).

So the rig task reduces to: **translate the KMPlane pointer semantics into flat
element offsets** and feed them to the already-verified `kt_degrain_patch` +
`kt_overlap_out`. Cross-check output frames vs the CUDA build of
`AviSynthCUDAFilters` (or mvtools CPU) on identical inputs.

## Remaining device-bound kernels (layer C) — implement on the rig

1. `kl_search` + its device functions (`Search`, `dev_calc_sad`,
   `dev_expanding_search_1/2`, `dev_hex2_search_1`, `dev_read_pixels`,
   `MinCost`, `dev_reduce_result`, `load4pix(_Aligned)`). Note the shipped CUDA
   launches compile with `CPU_EMU=true` → deterministic scalar search order;
   the whole grid is driven by a `next` work-stealing counter + `prog[]`
   spin-wait dependency (ANALYZE_SYNC) + `__threadfence`, which has no direct
   OpenCL workgroup equivalent — reproduce as a per-block serial host loop
   using the verified pure helpers + `kt_ref_block_offset` + `kt_prepare_search`
   data, then diff `VECTOR` arrays against CUDA.
2. `kl_degrain_2x3` / `kl_compensate_2x3`: their per-pixel integer math is
   already captured by the verified combiner core; what remains is reproducing
   their `(nPatternX,nPatternY,M)` host launch tiling of the global tmp (see
   `docs/MV_PORT_SPEC.md` §6.1). The `kt_overlap_out` per-pixel form is the
   simpler, order-equivalent way to get identical output.
3. `kl_prepare_degrain` / `kl_prepare_compensate`: mostly fold into the seam
   above (resolve per-block windows + ref bases); the per-block weights use the
   ALG-verified helpers; the scene-change gate is `isUsableB/F &&
   !(sceneChangeB[k] > nTh2)` and, for compensate, `vec.sad < thSAD` selects
   motion ref vs `ref0` at (0,0).

Authoritative sources: `/path/to/AviSynthCUDAFilters/KTGMC/{MVKernel.cu,MV.cpp}`
(this sandbox re-clones to `/tmp/avs_cuda/KTGMC`). Grounding docs:
`docs/MV_PORT_SPEC.md` (§5 super layout, §6.1 geometry, §6.2), `docs/BLOCKSEARCH_MODEL.md`,
`docs/HOST_CONTRACT.md`.
