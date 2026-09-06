/* ============================================================================
 * ktgmc_motion.cl — motion/super-sampling kernels from
 * AviSynthCUDAFilters/KTGMC/MVKernel.cu (stage 2, motion engine).
 *
 * The full MV engine (block search, degrain, compensate) additionally depends
 * on the MV.cpp host state machine and the super-frame sub-pel plane layout
 * documented in docs/MV_PORT_SPEC.md.  The kernels below are the pieces that
 * are self-contained enough to transliterate exactly; they are faithful source
 * ports but are marked // RIG-VERIFY until cross-checked on a real OpenCL/CUDA
 * device (see docs/MV_PORT_SPEC.md §7).  They use the same compile-time PX /
 * PX_MAX scheme as ktgmc_simple.cl.
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
