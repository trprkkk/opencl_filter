/* ============================================================================
 * ktgmc_motion.cl — motion/super-sampling kernels from
 * AviSynthCUDAFilters/KTGMC/MVKernel.cu (stage 2, motion engine).
 *
 * The full MV engine (block search, degrain, compensate) additionally depends
 * on the MV.cpp host state machine and the super-frame sub-pel plane layout
 * documented in docs/MV_PORT_SPEC.md.  The kernels below are the pieces that
 * are self-contained enough to transliterate exactly.
 *
 * Status legend:
 *   // ALG-VERIFIED : integer algorithm bit-for-bit cross-checked against an
 *                     independent CPU mirror + Python golden (make test).
 *   // RIG-VERIFY   : faithful source port; a real OpenCL/CUDA-device run is
 *                     still pending (see docs/MV_PORT_SPEC.md §7).
 *
 * Plane conventions follow the original: PX sample type, element pitch
 * (= plane row stride in samples).  For 8-bit the CUDA code used packed
 * uchar4 accesses only as a load optimization; the scalar addressing below
 * reproduces the same pixel arithmetic exactly.
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

/* ---------------------------------------------------------------------------
 * M1. kl_copy_pad — mirror-copy a width×height interior plane into a padded
 *     destination.  dst is written at (globalid - hPad, globalid - vPad), so
 *     the host must pass a dst pointer whose row start corresponds to the
 *     -hPad/-vPad origin (i.e. dst points at the padded interior offset), as
 *     the CUDA launcher did.  Out-of-range dst coords are guarded to the
 *     padded extent; source coords are edge-mirrored (reflection, -x-1).
 *     // RIG-VERIFY
 * -------------------------------------------------------------------------*/
kernel void kt_copy_pad(
    __global PX* __restrict dst, int dst_pitch,
    __global const PX* __restrict src, int src_pitch,
    int hPad, int vPad, int width, int height)
{
    int x = (int)get_global_id(0) - hPad;
    int y = (int)get_global_id(1) - vPad;

    if (x < width + hPad && y < height + vPad) {
        int srcx = x;
        if (srcx < 0)
            srcx = -srcx - 1;
        else if (srcx >= width)
            srcx = width - (srcx - width) - 1;
        int srcy = y;
        if (srcy < 0)
            srcy = -srcy - 1;
        else if (srcy >= height)
            srcy = height - (srcy - height) - 1;
        dst[x + y * dst_pitch] = src[srcx + srcy * src_pitch];
    }
}

/* ---------------------------------------------------------------------------
 * M2. kl_pad_frame_h — in-place horizontal edge replication on a buffer that
 *     already contains the interior at columns [hPad, hPad+width).  Left pad
 *     region [0,hPad) <- interior col hPad; right pad region
 *     [hPad+width, hPad+width+hPad) <- interior col hPad+width-1.
 *     Grid must cover x in [0,hPad); launch 2 groups on x (left/right).
 *     // RIG-VERIFY
 * -------------------------------------------------------------------------*/
kernel void kt_pad_frame_h(
    __global PX* ptr, int pitch, int hPad, int width, int height)
{
    bool isLeft = (get_group_id(0) == 0);
    int  x = get_local_id(0);
    int  y = get_local_id(1) + get_group_id(1) * get_local_size(1);

    if (y < height) {
        if (isLeft)
            ptr[x + y * pitch] = ptr[hPad + y * pitch];
        else
            ptr[(hPad + width + x) + y * pitch] =
                ptr[(hPad + width - 1) + y * pitch];
    }
}

/* ---------------------------------------------------------------------------
 * M3. kl_pad_frame_v — in-place vertical edge replication on a buffer that
 *     already contains the interior at rows [vPad, vPad+height).  Top pad
 *     rows [0,vPad) <- interior row vPad; bottom pad rows
 *     [vPad+height, vPad+height+vPad) <- interior row vPad+height-1.
 *     Launch 2 groups on y (top/bottom).
 *     // RIG-VERIFY
 * -------------------------------------------------------------------------*/
kernel void kt_pad_frame_v(
    __global PX* ptr, int pitch, int vPad, int width, int height)
{
    bool isTop = (get_group_id(1) == 0);
    int  x = get_local_id(0) + get_group_id(0) * get_local_size(0);
    int  y = get_local_id(1);

    if (x < width) {
        if (isTop)
            ptr[x + y * pitch] = ptr[x + vPad * pitch];
        else
            ptr[x + (vPad + height + y) * pitch] =
                ptr[x + (vPad + height - 1) * pitch];
    }
}

/* ---------------------------------------------------------------------------
 * Degrain numeric core (from AviSynthCUDAFilters/KTGMC/MVKernel.cu:
 * dev_degrain_weight + dev_norm_weights).  Pure scalar helpers used by the
 * degrain block kernels; validated against the CPU/Python references.
 * The CUDA originals are template functions on (N/delta, binomial); here the
 * ref arrays are fixed size (<=6) and delta/binomial are runtime params,
 * which reproduces the identical integer arithmetic for delta 1..6.
 * -------------------------------------------------------------------------*/

// M4. dev_degrain_weight
static int kt_degrain_weight(int thSAD, int blockSAD)
{
    if (thSAD <= blockSAD)
        return 0;
    float sq_thSAD    = (float)thSAD * (float)thSAD;
    float sq_blockSAD = (float)blockSAD * (float)blockSAD;
    return (int)(256.0f * (sq_thSAD - sq_blockSAD) / (sq_thSAD + sq_blockSAD));
}

// M5. dev_norm_weights(delta, binomial).  WRefB/WRefF have >= delta entries.
//     Returns normalized WSrc (in/out refs scaled so they sum with WSrc to 256).
static int kt_norm_weights(int delta, int binomial, int* WRefB, int* WRefF)
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
    int WSum;
    if (delta == 6)
        WSum = WRefB[0]+WRefF[0]+WSrc+WRefB[1]+WRefF[1]+WRefB[2]+WRefF[2]
             +WRefB[3]+WRefF[3]+WRefB[4]+WRefF[4]+WRefB[5]+WRefF[5]+1;
    else if (delta == 5)
        WSum = WRefB[0]+WRefF[0]+WSrc+WRefB[1]+WRefF[1]+WRefB[2]+WRefF[2]
             +WRefB[3]+WRefF[3]+WRefB[4]+WRefF[4]+1;
    else if (delta == 4)
        WSum = WRefB[0]+WRefF[0]+WSrc+WRefB[1]+WRefF[1]+WRefB[2]+WRefF[2]
             +WRefB[3]+WRefF[3]+1;
    else if (delta == 3)
        WSum = WRefB[0]+WRefF[0]+WSrc+WRefB[1]+WRefF[1]+WRefB[2]+WRefF[2]+1;
    else if (delta == 2)
        WSum = WRefB[0]+WRefF[0]+WSrc+WRefB[1]+WRefF[1]+1;
    else /* delta == 1 */
        WSum = WRefB[0]+WRefF[0]+WSrc+1;

    WRefB[0] = WRefB[0]*256/WSum; WRefF[0] = WRefF[0]*256/WSum;
    if (delta >= 2) { WRefB[1] = WRefB[1]*256/WSum; WRefF[1] = WRefF[1]*256/WSum; }
    if (delta >= 3) { WRefB[2] = WRefB[2]*256/WSum; WRefF[2] = WRefF[2]*256/WSum; }
    if (delta >= 4) { WRefB[3] = WRefB[3]*256/WSum; WRefF[3] = WRefF[3]*256/WSum; }
    if (delta >= 5) { WRefB[4] = WRefB[4]*256/WSum; WRefF[4] = WRefF[4]*256/WSum; }
    if (delta >= 6) { WRefB[5] = WRefB[5]*256/WSum; WRefF[5] = WRefF[5]*256/WSum; }

    if (delta == 6) WSrc = 256-WRefB[0]-WRefF[0]-WRefB[1]-WRefF[1]-WRefB[2]-WRefF[2]
                            -WRefB[3]-WRefF[3]-WRefB[4]-WRefF[4]-WRefB[5]-WRefF[5];
    else if (delta == 5) WSrc = 256-WRefB[0]-WRefF[0]-WRefB[1]-WRefF[1]-WRefB[2]-WRefF[2]
                            -WRefB[3]-WRefF[3]-WRefB[4]-WRefF[4];
    else if (delta == 4) WSrc = 256-WRefB[0]-WRefF[0]-WRefB[1]-WRefF[1]-WRefB[2]-WRefF[2]
                            -WRefB[3]-WRefF[3];
    else if (delta == 3) WSrc = 256-WRefB[0]-WRefF[0]-WRefB[1]-WRefF[1]-WRefB[2]-WRefF[2];
    else if (delta == 2) WSrc = 256-WRefB[0]-WRefF[0]-WRefB[1]-WRefF[1];
    else /* delta == 1 */ WSrc = 256-WRefB[0]-WRefF[0];
    return WSrc;
}

/* ---------------------------------------------------------------------------
 * MV array / conversion helpers (MVKernel.cu).  These are the small,
 * self-contained integer kernels around the block engine.  In OpenCL a VECTOR
 * (x,y,sad) is passed as an int3 buffer: mv[idx] = (x, y, sad).
 * -------------------------------------------------------------------------*/

// M6. kl_write_default_mv — initialise every MV to (0,0,verybigSAD).
//     NOTE: the upstream body does `dst[x].x=0; dst[x].y=0; dst[x].x=verybigSAD;`
//     (setting .x twice) which is evidently a typo for .sad; we implement the
//     clearly-intended default so blocks fail the SAD threshold.
//     ALG-VERIFIED (write to x=0,y=0,sad=verybigSAD)
kernel void kt_write_default_mv(
    __global int3* dst, int nBlkCount, int verybigSAD)
{
    int x = (int)get_global_id(0);
    if (x < nBlkCount) {
        dst[x].x = 0;
        dst[x].y = 0;
        dst[x].z = verybigSAD;
    }
}

// M7. kl_init_scene_change — zero the per-ref scene-change flags.
kernel void kt_init_scene_change(
    __global int* sceneChange)
{
    int x = (int)get_global_id(0);
    sceneChange[x] = 0;
}

// M8. kl_scene_change — count blocks whose SAD exceeds nTh1 into one scalar.
//     Final sceneChange == number of blocks with mv[].sad > nTh1 (integer
//     addition, order-independent).  Host zeroes *sceneChange first.
//     ALG-VERIFIED
kernel void kt_scene_change(
    __global const int3* mv, int nBlks, int nTh1,
    __global int* sceneChange)
{
    int x = (int)get_global_id(0);
    if (x < nBlks) {
        if (mv[x].z > nTh1)
            atomic_add(sceneChange, 1);
    }
}

// M9. kl_scene_change_x2 — same for two MV arrays (two separate counts).
//     ALG-VERIFIED
kernel void kt_scene_change_x2(
    __global const int3* mv0, __global const int3* mv1,
    int nBlks, int nTh1,
    __global int* sceneChange0, __global int* sceneChange1)
{
    int x = (int)get_global_id(0);
    if (x < nBlks) {
        if (mv0[x].z > nTh1) atomic_add(sceneChange0, 1);
        if (mv1[x].z > nTh1) atomic_add(sceneChange1, 1);
    }
}

// M10. kl_short_to_byte — convert the accumulated degrain tmp (which carries a
//      fixed shift) back to a pixel:  out = min(tmp >> shift, max_pixel_value).
//      CUDA: shift = 5 for 8-bit (tmp is uint16), 5+6=11 for 16-bit (tmp int32).
//      tmp values are non-negative (weighted sum of non-negative pixels).
//      ALG-VERIFIED
kernel void kt_short_to_byte(
    __global PX* __restrict dst, int dst_pitch,
    __global const int* __restrict tmp, int tmp_pitch,
    int width, int height,
    int shift)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int v = tmp[x + y * tmp_pitch] >> shift;
        if (v > PX_MAX) v = PX_MAX;
        if (v < 0) v = 0;
        dst[x + y * dst_pitch] = (PX)v;
    }
}

// M11. kl_short_to_byte_or_copy_src — same, but when *pflag is set copy src
//      instead (the scene-change path: no degrain output).  flag: 1 => convert
//      tmp, 0 => copy src.  ALG-VERIFIED (identical convert path to M10 plus a
//      copy branch).
kernel void kt_short_to_byte_or_copy_src(
    __global const int* __restrict pflag,      /* 1 = convert tmp, 0 = copy src */
    __global PX* __restrict dst, int dst_pitch,
    __global const PX* __restrict src, int src_pitch,
    __global const int* __restrict tmp, int tmp_pitch,
    int width, int height,
    int shift)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        if (pflag[0] != 0) {
            int v = tmp[x + y * tmp_pitch] >> shift;
            if (v > PX_MAX) v = PX_MAX;
            if (v < 0) v = 0;
            dst[x + y * dst_pitch] = (PX)v;
        } else {
            dst[x + y * dst_pitch] = src[x + y * src_pitch];
        }
    }
}

/* ---------------------------------------------------------------------------
 * M12. kl_interpolate_prediction — coarse→fine level MV upsampling.  For each
 *      fine-level block (x,y) picks 1-4 coarse neighbours and combines them with
 *      bilinear-style weights a11/a12/a21/a22 derived from parity offsets, then
 *      scales by normFactor (>> if >0 else <<-normFactor) and divides SAD by 16.
 *      Pure per-block integer function -> ALG-verifiable.
 *      src_vector row stride is nSrcBlkX; dst row stride nDstBlkX (the CUDA
 *      "pitch"/batch offset is handled by the host by passing per-batch
 *      pointers; blockIdx.z batching is dropped).
 *      // ALG-VERIFIED  (python/run_mv_interp.py, 200 random cases)
 * -------------------------------------------------------------------------*/
kernel void kt_interpolate_prediction(
    __global const int2* __restrict src_vector,
    __global const int*   __restrict src_sad,
    __global       int2* __restrict dst_vector,
    __global       int*   __restrict dst_sad,
    int nSrcBlkX, int nSrcBlkY,
    int nDstBlkX, int nDstBlkY,
    int normFactor, int normov, int atotal, int aodd, int aeven)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < nDstBlkX && y < nDstBlkY) {
        int i = x;
        int j = y;
        if (i >= 2 * nSrcBlkX) i = 2 * nSrcBlkX - 1;
        if (j >= 2 * nSrcBlkY) j = 2 * nSrcBlkY - 1;
        int offy = -1 + 2 * (j % 2);
        int offx = -1 + 2 * (i % 2);
        int iper2 = i >> 1;
        int jper2 = j >> 1;

        int v1x,v1y,v2x,v2y,v3x,v3y,v4x,v4y;
        int sad1,sad2,sad3,sad4;

        if ((i == 0) || (i >= 2 * nSrcBlkX - 1)) {
            if ((j == 0) || (j >= 2 * nSrcBlkY - 1)) {
                int2 v = src_vector[iper2 + jper2 * nSrcBlkX];
                int s = src_sad[iper2 + jper2 * nSrcBlkX];
                v1x=v2x=v3x=v4x=v.x; v1y=v2y=v3y=v4y=v.y;
                sad1=sad2=sad3=sad4=s;
            } else {
                int2 va = src_vector[iper2 + jper2 * nSrcBlkX];
                int2 vb = src_vector[iper2 + (jper2+offy) * nSrcBlkX];
                v1x=v2x=va.x; v1y=v2y=va.y;
                v3x=v4x=vb.x; v3y=v4y=vb.y;
                sad1=sad2=src_sad[iper2 + jper2 * nSrcBlkX];
                sad3=sad4=src_sad[iper2 + (jper2+offy) * nSrcBlkX];
            }
        } else if ((j == 0) || (j >= 2 * nSrcBlkY - 1)) {
            int2 va = src_vector[iper2 + jper2 * nSrcBlkX];
            int2 vb = src_vector[iper2 + offx + jper2 * nSrcBlkX];
            v1x=v2x=va.x; v1y=v2y=va.y;
            v3x=v4x=vb.x; v3y=v4y=vb.y;
            sad1=sad2=src_sad[iper2 + jper2 * nSrcBlkX];
            sad3=sad4=src_sad[iper2 + offx + jper2 * nSrcBlkX];
        } else {
            int2 v1 = src_vector[iper2 + jper2 * nSrcBlkX];
            int2 v2 = src_vector[iper2 + offx + jper2 * nSrcBlkX];
            int2 v3 = src_vector[iper2 + (jper2+offy) * nSrcBlkX];
            int2 v4 = src_vector[iper2 + offx + (jper2+offy) * nSrcBlkX];
            v1x=v1.x; v1y=v1.y; v2x=v2.x; v2y=v2.y;
            v3x=v3.x; v3y=v3.y; v4x=v4.x; v4y=v4.y;
            sad1=src_sad[iper2 + jper2 * nSrcBlkX];
            sad2=src_sad[iper2 + offx + jper2 * nSrcBlkX];
            sad3=src_sad[iper2 + (jper2+offy) * nSrcBlkX];
            sad4=src_sad[iper2 + offx + (jper2+offy) * nSrcBlkX];
        }

        int ax1 = (offx > 0) ? aodd : aeven;
        int ax2 = atotal - ax1;
        int ay1 = (offy > 0) ? aodd : aeven;
        int ay2 = atotal - ay1;
        int a11 = ax1*ay1, a12 = ax1*ay2, a21 = ax2*ay1, a22 = ax2*ay2;
        int vx = (a11*v1x + a21*v2x + a12*v3x + a22*v4x) / normov;
        int vy = (a11*v1y + a21*v2y + a12*v3y + a22*v4y) / normov;
        int tmp_sad = (a11*sad1 + a21*sad2 + a12*sad3 + a22*sad4) / normov;

        if (normFactor > 0) {
            vx >>= normFactor;
            vy >>= normFactor;
        } else {
            vx <<= -normFactor;
            vy <<= -normFactor;
        }

        int index = x + y * nDstBlkX;
        dst_vector[index].x = vx;
        dst_vector[index].y = vy;
        dst_sad[index] = (tmp_sad >> 4);
    }
}

/* ---------------------------------------------------------------------------
 * M13. kl_mean_global_mv — refine the per-row global MV by averaging the
 *      vectors whose components are within 6 of the previous median estimate
 *      (globalMVec[row] read from most_freq).  For each MV row: num = count of
 *      vectors v with |v.x-medianx|<6 && |v.y-mediany|<6; then
 *      globalMVec[row] = (2*sum_x/num, 2*sum_y/num).  The upstream kernel is a
 *      1024-thread staged tree/shuffle reduction, but integer addition is
 *      order-independent, so a per-row serial accumulation reproduces the
 *      exact sums (row stride = vectorsPitch elements; blockIdx.y batching is
 *      dropped, host passes per-row pointers). // ALG-VERIFIED below
 * -------------------------------------------------------------------------*/
kernel void kt_mean_global_mv(
    __global const int2* __restrict vectors, int vectorsPitch,
    int nVec,
    __global       int2* __restrict globalMVec)
{
    int y = (int)get_global_id(1);

    int medianx = globalMVec[y].x;
    int mediany = globalMVec[y].y;

    int meanvx = 0;
    int meanvy = 0;
    int num    = 0;

    __global const int2* row = vectors + (size_t)y * vectorsPitch;
    for (int i = 0; i < nVec; i++) {
        int vx = row[i].x;
        int vy = row[i].y;
        int dx = vx - medianx; if (dx < 0) dx = -dx;
        int dy = vy - mediany; if (dy < 0) dy = -dy;
        if (dx < 6 && dy < 6) {   // __sad(a,b,0)=|a-b|, threshold <6
            meanvx += vx;
            meanvy += vy;
            num += 1;
        }
    }
    globalMVec[y].x = (2 * meanvx) / num;
    globalMVec[y].y = (2 * meanvy) / num;
}

/* ---------------------------------------------------------------------------
 * M17. kl_prepare_search — per-block search setup (MVKernel.cu, ANALYZE_SYNC=1).
 *      For each block (bx,by) computes, purely from its own cell and its own
 *      vector+sad (no neighbour reads inside this kernel):
 *        nDxMax/nDyMax/nDxMin/nDyMin block-search bounds into data[0..3];
 *        predictor slot indices data[4..9] (sentinel -2=zero-vector, -1=global,
 *        data[6]=blkIdx current, data[7..9]=left/up/bottom-right neighbours,
 *        where left & bottom-right reference the prior-level COPY region offset
 *        +nBlkX*nBlkY and up references the current (already-searched) neighbour);
 *        data[10..11]=pred = this block's coarse-level MV (copied to vectors_copy);
 *        dataf[0..3]=penalties, dataf[4]=lambda (0 on row 0).
 *      The CUDA SearchBlock (data[12]+dataf[5]) is passed here as two flat int
 *      arrays with per-block strides 12 and 5.  blockIdx.z batching + prog/next
 *      row offsets are dropped (host passes per-batch pointers; prog[] is a
 *      per-column vector of length nBlkX). // ALG-VERIFIED below
 * -------------------------------------------------------------------------*/
kernel void kt_prepare_search(
    int nBlkX, int nBlkY, int nBlkSize, int nLogScale,
    int nLambdaLevel, int lsad,
    int penaltyZero, int penaltyGlobal, int penaltyNew,
    int nPel, int nPad, int nBlkSizeOvr,
    int nExtendedWidth, int nExtendedHeight,
    __global const int2* __restrict vectors,
    __global const int*   __restrict sads,
    __global       int2* __restrict vectors_copy,
    __global       int*  __restrict dst_data,   /* stride 12 per block */
    __global       int*  __restrict dst_dataf,  /* stride 5 per block  */
    __global       int*  __restrict prog,
    __global       int*  __restrict next)
{
    int bx = (int)get_global_id(0);
    int by = (int)get_global_id(1);

    if (bx < nBlkX && by < nBlkY) {
        int blkIdx = bx + by * nBlkX;
        int sad = sads[blkIdx];
        __global int* data = dst_data + blkIdx * 12;
        __global int* dataf = dst_dataf + blkIdx * 5;

        /* progress/counter init on row 0 (column 0 also zeroes *next) */
        if (by == 0) {
            prog[bx] = -1;
            if (bx == 0)
                *next = 0;
        }

        int x = nPad + nBlkSizeOvr * bx;
        int y = nPad + nBlkSizeOvr * by;
        int nPaddingScaled = nPad >> nLogScale;

        int nDxMax = nPel * (nExtendedWidth  - x - nBlkSize - nPad + nPaddingScaled) - 1;
        int nDyMax = nPel * (nExtendedHeight - y - nBlkSize - nPad + nPaddingScaled) - 1;
        int nDxMin = -nPel * (x - nPad + nPaddingScaled);
        int nDyMin = -nPel * (y - nPad + nPaddingScaled);

        data[0] = nDxMax;
        data[1] = nDyMax;
        data[2] = nDxMin;
        data[3] = nDyMin;

        int p1 = -2;            /* -2 -> zero vector */
        if (bx > 0)             /* ANALYZE_SYNC == 1 */
            p1 = blkIdx - 1 + nBlkX * nBlkY;   /* copy-region (prior-level) left */

        int p2 = -2;
        if (by > 0)
            p2 = blkIdx - nBlkX;               /* current up neighbour */
        else
            p2 = p1;                           /* let median pick left */

        int p3 = -2;
        if ((by < nBlkY - 1) && (bx < nBlkX - 1))
            p3 = blkIdx + nBlkX + 1 + nBlkX * nBlkY; /* copy-region bottom-right */

        data[4] = -2;           /* zero */
        data[5] = -1;           /* global */
        data[6] = blkIdx;       /* predictor (current) */
        data[7] = p1;           /* predictors[1] */
        data[8] = p2;           /* predictors[2] */
        data[9] = p3;           /* predictors[3] */

        int2 pred = vectors[blkIdx];
        vectors_copy[blkIdx] = pred;   /* keep prior-level vector for search */
        data[10] = pred.x;
        data[11] = pred.y;

        dataf[0] = penaltyZero;
        dataf[1] = penaltyGlobal;
        dataf[2] = 0;
        dataf[3] = penaltyNew;

        int lambda = nLambdaLevel * lsad / (lsad + (sad >> 1))
                   * lsad / (lsad + (sad >> 1));
        if (by == 0)
            lambda = 0;
        dataf[4] = lambda;
    }
}

/* ---------------------------------------------------------------------------
 * MV I/O & sentinel-init kernels (MVKernel.cu).  A VECTOR (x,y,sad) is an
 * int3; a motion "vector" (no sad) is an int2 in this port (CUDA short2).  The
 * CUDA kernels split VECTOR into a short2 vectors[] buffer + int sads[]; we
 * keep the two buffers separate but store ints.  Each MV row may carry two
 * sentinel slots before its first usable element ([-2] zero-vector, [-1]
 * global-vector), which the search setup's -2/-1 predictor indices reference.
 * -------------------------------------------------------------------------*/

// M14. kl_load_mv — split VECTOR int3 buffer into int2 vectors + int sads.
//     ALG-VERIFIED (values passed through; no arithmetic).
kernel void kt_load_mv(
    __global const int3* __restrict in,
    __global       int2* __restrict vectors,
    __global       int*  __restrict sads,
    int nBlk)
{
    int x = (int)get_global_id(0);
    if (x < nBlk) {
        int3 vin = in[x];
        vectors[x].x = vin.x;
        vectors[x].y = vin.y;
        sads[x] = vin.z;
    }
}

// M15. kl_store_mv — recombine int2 vectors + int sads into VECTOR int3.
//     ALG-VERIFIED.
kernel void kt_store_mv(
    __global int3* __restrict dst,
    __global const int2* __restrict vectors,
    __global const int*  __restrict sads,
    int nBlk)
{
    int x = (int)get_global_id(0);
    if (x < nBlk) {
        int2 v = vectors[x];
        dst[x].x = v.x;
        dst[x].y = v.y;
        dst[x].z = sads[x];
    }
}

// M16. kl_init_const_vec — write the two per-row sentinel motion vectors.
//     For each MV row r (base = row*vectorsPitch):
//       vectors[base-2] = (0,0)                       (zero-vector)
//       vectors[base-1] = globalMV * nPel              (global-vector)
//     The upstream CUDA launches 2 blocks on x (slot) x nRows (row); every
//     thread writes the same value (benign).  Grid: (2, nRows) in OpenCL.
//     ALG-VERIFIED (scaling by nPel is the only arithmetic).
kernel void kt_init_const_vec(
    __global int2* __restrict vectors, int vectorsPitch,
    __global const int2* __restrict globalMV, int nPel)
{
    int xslot = (int)get_global_id(0);   /* 0 -> slot -2, 1 -> slot -1 */
    int row   = (int)get_global_id(1);
    __global int2* base = vectors + (size_t)row * vectorsPitch;
    if (xslot == 0) {
        base[-2].x = 0;
        base[-2].y = 0;
    } else {
        int2 g = globalMV[0];
        base[-1].x = g.x * nPel;
        base[-1].y = g.y * nPel;
    }
}

/* ---------------------------------------------------------------------------
 * M18. kl_most_freq_mv — per-row "mode" of the MV component, used as the
 *      per-row global-MV seed that kl_mean_global_mv then refines.  Upstream
 *      builds a shared histogram of component value (comp+4096) over 8192 bins
 *      and returns a maximally-counted component.
 *
 *      TIE-BREAK / RIG-VERIFY: when the global-max count M is unique the result
 *      is the single most-frequent value (all sensible rules agree and this is
 *      bit-identical to CUDA).  When several values tie at M, the CUDA winner is
 *      an artifact of its 1024-thread reduction tree (per-thread residue scan +
 *      dev_reduce2 tie "lower tid"), which we found is NOT reducible to a simple
 *      min/max rule.  The port below therefore returns the SMALLEST value among
 *      the modes - a deterministic, intent-faithful choice - and is marked
 *      RIG-VERIFY: the tie case must be reconciled against the CUDA build on a
 *      real rig if bit-exact tie output is ever required (docs/MV_PORT_SPEC.md
 *      §7).  In practice the seed is then refined by kl_mean_global_mv, so a
 *      different equal-frequency mode rarely changes the final output.
 *
 *      Pure per-row serial scan, O(nVec^2); no work-group / histogram buffer
 *      needed (once-per-level seed; a two-pass histogram may optimize later but
 *      must keep this documented tie-break).
 *      // RIG-VERIFY (tie-break vs CUDA pending on-rig reconciliation)
 * -------------------------------------------------------------------------*/
kernel void kt_most_freq_mv(
    __global const int2* __restrict vectors, int vectorsPitch,
    int nVec,
    int isY,                     /* 0 -> mode of x, else mode of y */
    __global       int2* __restrict globalMVec)
{
    int row = (int)get_global_id(1);
    __global const int2* base = vectors + (size_t)row * vectorsPitch;

    int bestCnt = 0;
    int bestVal = 0x7fffffff;

    for (int i = 0; i < nVec; i++) {
        int vi = isY ? base[i].y : base[i].x;
        int cnt = 0;
        for (int j = 0; j < nVec; j++) {
            int vj = isY ? base[j].y : base[j].x;
            if (vj == vi) cnt++;
        }
        if (cnt > bestCnt || (cnt == bestCnt && vi < bestVal)) {
            bestCnt = cnt;
            bestVal = vi;
        }
    }
    if (isY)
        globalMVec[row].y = bestVal;
    else
        globalMVec[row].x = bestVal;
}

/* ---------------------------------------------------------------------------
 * M19. kl_RB2B_bilinear_filtered — anti-aliased 1:2 downsample of a source
 *      plane (source 2*nWidth x 2*nHeight -> dst nWidth x nHeight) using the
 *      separable (1,3,3,1)/8 = {1/8,3/8,3/8,1/8} half-band filter (Fizick),
 *      used by KMSuper to build reduced analysis planes (MV.cpp ReduceTo).
 *
 *      The upstream CUDA kernel (kl_RB2B_bilinear_filtered) is a *separable*
 *      two-phase filter fused with a shared tile: a vertical phase halves the
 *      height, then an in-place horizontal phase halves the width, each phase
 *      rounding separately (top/bottom edge: (a+b+1)>>1, interior:
 *      (a + 3b + 3c + d + 4)/8).  Because the two phases only ever combine, per
 *      output pixel, a bounded 2*2 source region, this single kernel recomputes
 *      the vertical result for the (<=4) intermediate columns each output pixel
 *      needs and then applies the horizontal phase — reproducing the CPU
 *      reference (RB2BilinearFiltered) bit-for-bit with the same two roundings,
 *      without an intermediate buffer.
 *
 *      Out(x,y):  intermediate col k = (2x-1..2x+2 interior, else 2x,2x+1); each
 *      vertical val V(y,col): y==0 or y==nHeight-1 edge => rows 2y,2y+1 averaged
 *      (top row uses rows 0,1), else interior => rows 2y-1,2y,2y+1,2y+2 with
 *      (1,3,3,1)/8.  // ALG-VERIFIED below
 * -------------------------------------------------------------------------*/
static int kt_rb2b_vertical(__global const PX* __restrict src, int src_pitch,
                            int col, int y, int nHeight)
{
    if (y == 0) {
        int a = (int)src[col];
        int b = (int)src[col + src_pitch];
        return (a + b + 1) >> 1;
    }
    if (y < nHeight - 1) {
        int r0 = (int)src[col + (2 * y - 1) * src_pitch];
        int r1 = (int)src[col + (2 * y)     * src_pitch];
        int r2 = (int)src[col + (2 * y + 1) * src_pitch];
        int r3 = (int)src[col + (2 * y + 2) * src_pitch];
        return (r0 + r1 * 3 + r2 * 3 + r3 + 4) / 8;
    }
    /* y == nHeight - 1 : bottom edge */
    {
        int a = (int)src[col + (2 * y)     * src_pitch];
        int b = (int)src[col + (2 * y + 1) * src_pitch];
        return (a + b + 1) >> 1;
    }
}

kernel void kt_rb2b_bilinear_filtered(
    __global const PX* __restrict src, int src_pitch,
    __global       PX* __restrict dst, int dst_pitch,
    int nWidth, int nHeight)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < nWidth && y < nHeight) {
        int v;
        if (x == 0 || x == nWidth - 1) {
            /* edge: average the two intermediate columns 2x and 2x+1 */
            int a = kt_rb2b_vertical(src, src_pitch, 2 * x,     y, nHeight);
            int b = kt_rb2b_vertical(src, src_pitch, 2 * x + 1, y, nHeight);
            v = (a + b + 1) >> 1;
        } else {
            int a = kt_rb2b_vertical(src, src_pitch, 2 * x - 1, y, nHeight);
            int b = kt_rb2b_vertical(src, src_pitch, 2 * x,     y, nHeight);
            int c = kt_rb2b_vertical(src, src_pitch, 2 * x + 1, y, nHeight);
            int d = kt_rb2b_vertical(src, src_pitch, 2 * x + 2, y, nHeight);
            v = (a + b * 3 + c * 3 + d + 4) / 8;
        }
        dst[x + y * dst_pitch] = (PX)v;
    }
}
