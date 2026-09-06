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
