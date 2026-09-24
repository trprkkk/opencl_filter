/* ============================================================================
 * ktgmc_degrain_rig.cl — PROVISIONAL (// RIG-VERIFY) ports of the two MV
 * per-block "prepare" kernels from KTGMC/MVKernel.cu (upstream
 * rigaya/AviSynthCUDAFilters @68aef6e, GPL):
 *     kl_prepare_degrain     (:1884-1968)  -> kt_prepare_degrain
 *     kl_prepare_compensate  (:2137-2181)  -> kt_prepare_compensate
 *
 * This file is DELIBERATELY SEPARATE from the verified ktgmc_motion.cl, the
 * same quarantine convention used by kfm_deblock_rig.cl and
 * avscuda_conditional_rig.cl.  Do not move a kernel out of here without the
 * evidence described in §"Graduation" below.
 *
 * -- Why they are quarantined -----------------------------------------------
 * Their per-block ARITHMETIC is self-contained and is pinned by
 * python/run_mv_prepare.py (mirror sim/mv_prepare_ref.cpp).  What is NOT
 * settled in this sandbox is the surrounding host model: docs/MV_PORT_SPEC.md
 * §6.1 records that the CUDA block geometry (blkStep = nBlkSize/2, origin
 * nPad + blk*blkStep) and the MV.cpp CPU model (StepX = nBlkSizeX - nOverlapX,
 * KMPlane::GetPointer padding/pel semantics) are two different coordinate
 * models, and only a rig can say which one a given call site is in.  These
 * kernels transcribe the CUDA side verbatim.
 *
 * -- Pointers become offsets (the §6.2 seam) --------------------------------
 * Upstream fills a DegrainBlockData / CompensateBlockData struct of raw
 * POINTERS (b.pSrc, b.pB[i], b.pF[i], b.pRef, b.winOver).  OpenCL cannot
 * store device pointers portably, and the already-ALG-VERIFIED consumer
 * kt_degrain_patch (ktgmc_motion.cl) does not want them: it takes flat
 * per-block ELEMENT OFFSETS (refBaseF/refBaseB, indexed k*nBlk+blk) and
 * weight arrays.  So these kernels emit exactly that, which is what
 * docs/MV_PORT_SPEC.md §6.2 calls "the one seam between verified kernels and
 * the rig".  Two encoding decisions, both output-equivalent, both must be
 * honoured by the host:
 *
 *   1. UNUSABLE REFERENCE.  Upstream parks the pointer at `arg.pSrc` (a
 *      different plane with a different pitch) and sets the weight to 0,
 *      with the comment that the values read there are never used.  This
 *      port emits offset 0 into the normal ref plane and weight 0.  The
 *      pixels differ, the OUTPUT cannot: kt_degrain_patch multiplies them by
 *      that zero weight.  The host must still guarantee offset 0 is
 *      dereferenceable (it always is).
 *   2. SCENE CHANGE (compensate only).  Upstream writes nullptr for both
 *      winOver and pRef.  This port writes the sentinel -1 to win_slot and
 *      ref_base, and the host must treat -1 as "skip this block" exactly as
 *      a null check would.
 *
 * -- What is NOT here -------------------------------------------------------
 * kl_degrain_2x3 (:1972) and kl_compensate_2x3 (:2185) are still unported ON
 * PURPOSE.  They accumulate into the global tmp plane with `+=` from
 * different thread blocks, and that is only race-free because the host
 * dispatches several disjoint (nPatternX, nPatternY, M) passes; the kernel
 * is defined by that launch pattern rather than by its own indexing.  A
 * transcription without the host pattern would be untestable and unsafe.
 * The verified combiner pair kt_degrain_patch + kt_overlap_out already
 * covers their arithmetic once the host supplies windows and the arrays
 * these two kernels produce.
 *
 * -- Graduation -------------------------------------------------------------
 * To move these into ktgmc_motion.cl as // ALG-VERIFIED you need, on a rig:
 * (a) confirmation of which block-geometry model the call site uses (§6.1),
 * (b) a CUDA-vs-OpenCL comparison of the emitted arrays for a real clip, and
 * (c) confirmation that the two encoding decisions above are honoured by the
 * host glue.  The arithmetic itself is already pinned in-sandbox.
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

#define MV_MAX_DELTA 6

/* dev_get_ref_block's element offset — same function as ktgmc_motion.cl's
 * kt_ref_block_offset (ALG-VERIFIED there); duplicated so this quarantined
 * file stays self-contained and can be compiled on its own. */
static int kd_ref_block_offset(int vx, int vy, int nPitch, int nImgPitch,
                               int NPEL)
{
    if (NPEL == 1)
        return vx + vy * nPitch;
    if (NPEL == 2) {
        int sx = vx & 1;
        int sy = vy & 1;
        return (vx >> 1) + (vy >> 1) * nPitch + (sx + sy * 2) * nImgPitch;
    }
    {
        int sx = vx & 3;
        int sy = vy & 3;
        return (vx >> 2) + (vy >> 2) * nPitch + (sx + sy * 4) * nImgPitch;
    }
}

/* dev_degrain_weight (ALG-VERIFIED twin in ktgmc_motion.cl) */
static int kd_degrain_weight(int thSAD, int blockSAD)
{
    if (thSAD <= blockSAD)
        return 0;
    float sq_thSAD    = (float)thSAD * (float)thSAD;
    float sq_blockSAD = (float)blockSAD * (float)blockSAD;
    return (int)(256.0f * (sq_thSAD - sq_blockSAD) / (sq_thSAD + sq_blockSAD));
}

/* dev_norm_weights<N,BINOMIAL> (ALG-VERIFIED twin in ktgmc_motion.cl).
 * Upstream and ktgmc_motion.cl spell the WSum accumulation and the final
 * WSrc subtraction as unrolled per-delta chains; these loops are the same
 * integer additions in a different order, which is exact. */
static int kd_norm_weights(int delta, int binomial, int* WRefB, int* WRefF)
{
    int WSrc = 256;
    if (binomial) {
        if (delta == 1) {
            WSrc *= 2;
        } else if (delta == 2) {
            WSrc *= 6;
            WRefB[0] *= 4; WRefF[0] *= 4;
        } else if (delta == 3) {
            WSrc *= 20;
            WRefB[0] *= 15; WRefF[0] *= 15;
            WRefB[1] *= 6;  WRefF[1] *= 6;
        } else if (delta == 4) {
            WSrc *= 70;
            WRefB[0] *= 56; WRefF[0] *= 56;
            WRefB[1] *= 28; WRefF[1] *= 28;
            WRefB[2] *= 8;  WRefF[2] *= 8;
        }
    }
    int WSum = WSrc + 1;
    for (int i = 0; i < delta; ++i)
        WSum += WRefB[i] + WRefF[i];
    for (int i = 0; i < delta; ++i) {
        WRefB[i] = WRefB[i] * 256 / WSum;
        WRefF[i] = WRefF[i] * 256 / WSum;
    }
    WSrc = 256;
    for (int i = 0; i < delta; ++i)
        WSrc -= WRefB[i] + WRefF[i];
    return WSrc;
}

/* The 9-window slot upstream selects with
 *   wby = ((blky + nBlkY - 3) / (nBlkY - 2)) * 3
 *   wbx =  (blkx + nBlkX - 3) / (nBlkX - 2)
 * NOTE these divide by (nBlkX-2)/(nBlkY-2): a 2-block-wide or 2-block-tall
 * grid divides by zero upstream too.  The port does not paper over it; the
 * host must not call with nBlkX == 2 or nBlkY == 2. */
static int kd_win_slot(int blkx, int blky, int nBlkX, int nBlkY)
{
    int wby = ((blky + nBlkY - 3) / (nBlkY - 2)) * 3;
    int wbx =  (blkx + nBlkX - 3) / (nBlkX - 2);
    return wby + wbx;
}

/* ---------------------------------------------------------------------------
 * kt_prepare_degrain — kl_prepare_degrain twin.  Grid 2D (nBlkX, nBlkY).
 *
 * Emits, per block blk = blkx + blky*nBlkX (nBlk = nBlkX*nBlkY):
 *   win_slot[blk]            0..8, the overlap window index
 *   src_base[blk]            element offset of the block in the src plane
 *   WSrcArr[blk]             normalised source weight
 *   WBArr[i*nBlk + blk]      normalised backward weights   (i < delta)
 *   WFArr[i*nBlk + blk]      normalised forward weights
 *   refBaseB[i*nBlk + blk]   element offset into ref plane B[i]
 *   refBaseF[i*nBlk + blk]   element offset into ref plane F[i]
 * which is exactly the input contract of kt_degrain_patch.
 *
 * mvB/mvF are upstream VECTOR arrays: 3 ints per block (x, y, sad), laid out
 * [i][blk] -> (i*nBlk + blk)*3.  NOT OpenCL int3 (stride 16) — see
 * docs/BLOCKSEARCH_MODEL.md §8a for why that distinction bit us once already.
 * // RIG-VERIFY (see file header).
 * -------------------------------------------------------------------------*/
kernel void kt_prepare_degrain(
    int nBlkX, int nBlkY, int nPad, int nBlkSize,
    int nTh2, int thSAD, int delta, int binomial, int NPEL, int SHIFT,
    int nPitch, int nPitchSuper, int nImgPitch,
    __global const int* __restrict sceneChangeB,   /* delta ints */
    __global const int* __restrict sceneChangeF,   /* delta ints */
    __global const int* __restrict isUsableB,      /* delta flags */
    __global const int* __restrict isUsableF,      /* delta flags */
    __global const int* __restrict mvB,            /* delta*nBlk VECTORs */
    __global const int* __restrict mvF,
    __global       int* __restrict win_slot,
    __global       int* __restrict src_base,
    __global       int* __restrict WSrcArr,
    __global       int* __restrict WBArr,
    __global       int* __restrict WFArr,
    __global       int* __restrict refBaseB,
    __global       int* __restrict refBaseF)
{
    int blkx = (int)get_global_id(0);
    int blky = (int)get_global_id(1);
    if (blkx >= nBlkX || blky >= nBlkY)
        return;

    int nBlk = nBlkX * nBlkY;
    int idx = blkx + blky * nBlkX;

    win_slot[idx] = kd_win_slot(blkx, blky, nBlkX, nBlkY);

    int blkStep = nBlkSize / 2;
    int offx = blkx * blkStep;
    int offy = blky * blkStep;
    int offsetS = (nPad + offx) + (nPad + offy) * nPitchSuper;
    src_base[idx] = offx + offy * nPitch;

    int WRefB[MV_MAX_DELTA], WRefF[MV_MAX_DELTA];
    for (int i = 0; i < MV_MAX_DELTA; ++i) { WRefB[i] = 0; WRefF[i] = 0; }

    for (int i = 0; i < delta; ++i) {
        int m = (i * nBlk + idx) * 3;

        int usableB = isUsableB[i] && !(sceneChangeB[i] > nTh2);
        if (usableB) {
            refBaseB[i * nBlk + idx] = offsetS
                + kd_ref_block_offset(mvB[m + 0] >> SHIFT, mvB[m + 1] >> SHIFT,
                                      nPitchSuper, nImgPitch, NPEL);
            WRefB[i] = kd_degrain_weight(thSAD, mvB[m + 2]);
        } else {
            /* upstream parks the pointer on pSrc; weight 0 makes it unused */
            refBaseB[i * nBlk + idx] = 0;
            WRefB[i] = 0;
        }

        int usableF = isUsableF[i] && !(sceneChangeF[i] > nTh2);
        if (usableF) {
            refBaseF[i * nBlk + idx] = offsetS
                + kd_ref_block_offset(mvF[m + 0] >> SHIFT, mvF[m + 1] >> SHIFT,
                                      nPitchSuper, nImgPitch, NPEL);
            WRefF[i] = kd_degrain_weight(thSAD, mvF[m + 2]);
        } else {
            refBaseF[i * nBlk + idx] = 0;
            WRefF[i] = 0;
        }
    }

    int WSrc = kd_norm_weights(delta, binomial, WRefB, WRefF);

    WSrcArr[idx] = WSrc;
    for (int i = 0; i < delta; ++i) {
        WBArr[i * nBlk + idx] = WRefB[i];
        WFArr[i * nBlk + idx] = WRefF[i];
    }
}

/* ---------------------------------------------------------------------------
 * kt_prepare_compensate — kl_prepare_compensate twin.  Grid 2D (nBlkX,nBlkY).
 *
 * Emits per block:
 *   win_slot[blk]   0..8, or -1 on scene change (upstream: winOver = nullptr)
 *   ref_base[blk]   element offset, or -1 on scene change
 *   ref_sel[blk]    1 = the motion plane pRef, 0 = the static plane pRef0,
 *                   -1 = scene change.  Upstream encodes this by which
 *                   pointer it stores; the host must pick the plane.
 *
 * Note the scaling `mx = (vec.x * time256 / 256) >> SHIFT`: the /256 is a C
 * integer division that TRUNCATES TOWARD ZERO, while >> SHIFT is an
 * arithmetic shift that floors.  For negative MVs those differ, so the port
 * keeps both spellings exactly rather than "simplifying" to one shift.
 * // RIG-VERIFY (see file header).
 * -------------------------------------------------------------------------*/
kernel void kt_prepare_compensate(
    int nBlkX, int nBlkY, int nPad, int nBlkSize,
    int nTh2, int time256, int thSAD, int NPEL, int SHIFT,
    int nPitchSuper, int nImgPitch,
    __global const int* __restrict sceneChange,    /* single int */
    __global const int* __restrict mv,             /* nBlk VECTORs */
    __global       int* __restrict win_slot,
    __global       int* __restrict ref_base,
    __global       int* __restrict ref_sel)
{
    int blkx = (int)get_global_id(0);
    int blky = (int)get_global_id(1);
    if (blkx >= nBlkX || blky >= nBlkY)
        return;

    int idx = blkx + blky * nBlkX;

    if (sceneChange[0] > nTh2) {
        win_slot[idx] = -1;
        ref_base[idx] = -1;
        ref_sel[idx]  = -1;
        return;
    }

    win_slot[idx] = kd_win_slot(blkx, blky, nBlkX, nBlkY);

    int blkStep = nBlkSize / 2;
    int offsetS = (nPad + blkx * blkStep)
                + (nPad + blky * blkStep) * nPitchSuper;

    int m = idx * 3;
    if (mv[m + 2] < thSAD) {
        int mx = (mv[m + 0] * time256 / 256) >> SHIFT;
        int my = (mv[m + 1] * time256 / 256) >> SHIFT;
        ref_base[idx] = offsetS + kd_ref_block_offset(mx, my, nPitchSuper,
                                                      nImgPitch, NPEL);
        ref_sel[idx] = 1;
    } else {
        ref_base[idx] = offsetS + kd_ref_block_offset(0, 0, nPitchSuper,
                                                      nImgPitch, NPEL);
        ref_sel[idx] = 0;
    }
}
