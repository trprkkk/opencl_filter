/* ============================================================================
 * kfm_filterbase.cl — OpenCL port of the KFM coefficient kernels that live in
 * the shared base `KFMFilterBase.cu` (KFM, MIT) and that the KAnalyzeStatic
 * pipeline (MergeStatic.cu) is built from.
 *
 * KAnalyzeStatic::GetFrameT (MergeStatic.cu) computes a static/coefficient
 * frame whose per-plane dataflow is:
 *   padded = pad(frame n)                       // host glue (VPAD mirror, rig)
 *   CompareFields(padded, flagtmp)  -> kf_calc_combe (Y/U/V)
 *   MergeUVCoefs(flagtmp)           -> kf_merge_uvcoefs (fold UV into Y)
 *   ExtendCoefs(flagtmp, flagc)     -> kf_extend_coef2 (Y only)
 *   GetTemporalDiff(diff, flagtmp)  -> kf_min_frames  (already in
 *                                       kfm_mergestatic.cl)  over 3 diff frames
 *   MergeUVCoefs(flagtmp)           -> kf_merge_uvcoefs
 *   ExtendCoefs(flagtmp, flagd)     -> kf_extend_coef2
 *   AndCoefs(flagc, flagd)          -> kf_and_coefs (already in
 *                                       kfm_mergestatic.cl)  combe ^ diff
 *   ApplyUVCoefs(flagc)             -> kf_apply_uvcoefs_420 (Y -> UV)
 * This file supplies the four kernels defined in KFMFilterBase.cu that the
 * pipeline still needs (kf_min_frames and kf_and_coefs already live in
 * kfm_mergestatic.cl).  All four are per-pixel / per-plane integer ops whose
 * channels are independent, so the CUDA 4-wide vectorisation is equivalent to a
 * scalar translation (bit-identical for plane width a multiple of 4).
 *
 * Status:
 *   // ALG-VERIFIED (python/run_kfm_filterbase.py) vs sim/kfm_filterbase_ref.cpp
 *   //   cpu_calc_combe / cpu_merge_uvcoefs / cpu_apply_uvcoefs_420 are exact
 *   //   twins; cpu_extend_coef (below) is the CUDA kl_extend_coef2 twin that
 *   //   the .cl transliterates.
 *
 * Fidelity / assembly notes:
 *  - calc_combe / merge_uvcoefs / apply_uvcoefs_420 have identical CUDA and CPU
 *    implementations upstream.
 *  - ExtendCoefs: the real CUDA device path uses kl_extend_coef2 (row index
 *    clamped to [0,height-1]), which is what this .cl transliterates.  The
 *    original CPU *fallback* branch instead runs cpu_extend_coef over the
 *    interior rows plus cpu_copy_border, which copies the extreme rows straight
 *    through (no max with the neighbour); that differs from the device kernel
 *    at row 0 and row height-1.  We follow the device kernel (the OpenCL target
 *    is the GPU), so kf_extend_coef2 is checked against its own transliteration,
 *    and the CPU-vs-CUDA border discrepancy is upstream's, noted here (and
 *    only affects 2 rows, and KAnalyzeStatic's result there anyway flows into
 *    a band-pass coefficient).
 *  - calc_combe reads source rows y-2..y+2 with no border guard: upstream feeds
 *    it a vertically-mirror-padded frame (VPAD=4), so the top/bottom rows are
 *    rig-bound on the padded plane; the arithmetic is verified over the
 *    interior (rows 2..height-3) here.  The host pad layout / offset is a
 *    RIG-VERIFY seam (the calc_combe launch geometry over the padded buffer is
 *    host glue, as with KEdgeLevel's el_to444 host sizing).
 *  - calc_combe's output is clamped to [0,255] regardless of pixel bit depth
 *    (upstream casts the clamped int to the pixel type; the coefficient planes
 *    KAnalyzeStatic uses are 0..128-scaled values, so this matches).
 *  - merge_uvcoefs / apply_uvcoefs_420 assume YV12-style logUVx=logUVy=1
 *    (KAnalyzeStatic enforces 420).
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

/* CalcCombe(a,b,c,d,e) = abs(a + c*4 + e - (b+d)*3)  (KFMFilterBase.cuh) */
static int kf_calc_combe_val(int a, int b, int c, int d, int e)
{
    int v = a + c * 4 + e - (b + d) * 3;
    if (v < 0) v = -v;
    return v;
}

/* ---------------------------------------------------------------------------
 * kf_calc_combe — CompareFields core (cpu_calc_combe / kl_calc_combe twin).
 * Combing measure at each pixel from the 5 vertical taps y-2..y+2 of a
 * (host-padded) interlaced frame: combe = CalcCombe(...) >> 2, clamped [0,255].
 * Grid: 2D (width,height).  Interior ALG-VERIFIED; border rows need the padded
 * plane (RIG-VERIFY).
 * -------------------------------------------------------------------------*/
kernel void kf_calc_combe(
    __global       PX* __restrict dst,
    __global const PX* __restrict src,
    int width, int height, int pitch)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int off = x + y * pitch;
        int combe = kf_calc_combe_val(
            (int)src[off - 2 * pitch],
            (int)src[off - 1 * pitch],
            (int)src[off + 0 * pitch],
            (int)src[off + 1 * pitch],
            (int)src[off + 2 * pitch]);
        combe >>= 2;
        if (combe < 0) combe = 0; else if (combe > 255) combe = 255;
        dst[off] = (PX)combe;
    }
}

/* ---------------------------------------------------------------------------
 * kf_merge_uvcoefs — MergeUVCoefs core (cpu_merge_uvcoefs / kl_merge_uvcoefs
 * twin).  In-place on the full-size Y coefficient plane: fY = max(fY,
 * max(fU,fV)) where the UV coefficient at (x,y) comes from the subsampled
 * plane offset (x>>logUVx, y>>logUVy).  Grid 2D (width,height)=Y plane dims.
 * YV12 logUVx=logUVy=1.  // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_merge_uvcoefs(
    __global       PX* __restrict fY,
    __global const PX* __restrict fU,
    __global const PX* __restrict fV,
    int width, int height, int pitchY, int pitchUV,
    int logUVx, int logUVy)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int offY = x + y * pitchY;
        int offUV = (x >> logUVx) + (y >> logUVy) * pitchUV;
        int u = (int)fU[offUV], v = (int)fV[offUV];
        int yv = (int)fY[offY];
        if (u > yv) yv = u;
        if (v > yv) yv = v;
        fY[offY] = (PX)yv;
    }
}

/* ---------------------------------------------------------------------------
 * kf_extend_coef2 — ExtendCoefs core = the CUDA kl_extend_coef2 device kernel
 * (see fidelity note re the divergent CPU fallback).  dst[x+y*pitch] = max of
 * src over the 3 vertical rows y-1..y+1 with y clamped to [0,height-1], so
 * borders are handled without an extra pad.  Y plane only.  ALG-VERIFIED.
 * -------------------------------------------------------------------------*/
kernel void kf_extend_coef2(
    __global       PX* __restrict dst,
    __global const PX* __restrict src,
    int width, int height, int pitch)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int y0 = y - 1; if (y0 < 0) y0 = 0;
        int y2 = y + 1; if (y2 > height - 1) y2 = height - 1;
        int off = x + y * pitch;
        int a = (int)src[x + y0 * pitch];
        int b = (int)src[off];
        int c = (int)src[x + y2 * pitch];
        int m = a;
        if (b > m) m = b;
        if (c > m) m = c;
        dst[off] = (PX)m;
    }
}

/* ---------------------------------------------------------------------------
 * kf_apply_uvcoefs_420 — ApplyUVCoefs core (cpu_apply_uvcoefs_420 /
 * kl_apply_uvcoefs_420 twin).  Down-samples the Y coefficient plane into U and
 * V (both equal) as the rounded 2x2 average:
 *   fU=fV = (fY[2x,2y]+fY[2x+1,2y]+fY[2x,2y+1]+fY[2x+1,2y+1] + 2) >> 2
 * Grid 2D (widthUV,heightUV).  Assumes logUVx=logUVy=1 (YV12). ALG-VERIFIED.
 * -------------------------------------------------------------------------*/
kernel void kf_apply_uvcoefs_420(
    __global const PX* __restrict fY,
    __global       PX* __restrict fU,
    __global       PX* __restrict fV,
    int widthUV, int heightUV, int pitchY, int pitchUV)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < widthUV && y < heightUV) {
        int offY0 = (2 * x + 0) + (2 * y + 0) * pitchY;
        int v =
            (int)fY[offY0] +
            (int)fY[offY0 + 1] +
            (int)fY[offY0 + pitchY] +
            (int)fY[offY0 + pitchY + 1];
        int avg = (v + 2) >> 2;
        int offUV = x + y * pitchUV;
        fU[offUV] = (PX)avg;
        fV[offUV] = (PX)avg;
    }
}
