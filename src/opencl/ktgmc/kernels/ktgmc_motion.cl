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
