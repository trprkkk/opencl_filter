/* ============================================================================
 * kfm_mergestatic.cl — OpenCL port of the KFM `MergeStatic.cu` filter kernels.
 *
 * Faithful port of rigaya/AviSynthCUDAFilters -> KFM/MergeStatic.cu (KFM is
 * MIT).  That one source file registers three AVS filters (KTemporalDiff,
 * KAnalyzeStatic, KMergeStatic) and defines four device kernels, all
 * transliterated here:
 *
 *   kf_compare_frames  — KTemporalDiff  (the whole filter, per plane)
 *   kf_min_frames      — KAnalyzeStatic  (one sub-step)
 *   kf_and_coefs       — KAnalyzeStatic  (one sub-step, float32)
 *   kf_merge_static    — KMergeStatic   (the whole filter, per plane)
 *
 * Every kernel is a per-pixel, per-plane op whose channels are independent, so
 * the CUDA 4-wide (uchar4/ushort4) vectorisation over `width4 = width>>2`
 * vectors is equivalent to a scalar translation over the real pixel columns —
 * bit-identical whenever the plane width is a multiple of 4 (the CUDA code
 * never processes the `width%4` trailing columns; a faithful host config uses a
 * width divisible by 4).  `VHelper<vpixel_t>::cast_to(x)` is a plain truncating
 * cast to uchar/ushort, which only ever sees in-range values here.
 *
 * Status:
 *   // ALG-VERIFIED (python/run_kfm_mergestatic.py) vs sim/kfm_mergestatic_ref.cpp
 *   //   cpu_compare_frames / cpu_min_frames / cpu_merge_static are exact twins
 *   //   in MergeStatic.cu (integer).  cpu_and_coefs is NOT defined upstream
 *   //   (KAnalyzeStatic's CPU branch is CUDA-only), so kf_and_coefs is checked
 *   //   against an independent float32-exact mirror/golden (see below).
 *
 * Fidelity / assembly notes:
 *  - KTemporalDiff = per plane one kf_compare_frames over frames [n-2..n+2].
 *  - KMergeStatic  = host copies the 60fps frame into dst, then per plane one
 *    kf_merge_static over the 60fps/30fps frames + the static flag plane
 *    (`flag` is the KAnalyzeStatic coefficient plane, expected in [0,128];
 *    coef=128 => take the 30fps frame, coef=0 => keep the 60fps frame).
 *  - KAnalyzeStatic is a pipeline: pad -> CompareFields(combe) ->
 *    MergeUVCoefs/ExtendCoefs -> temporal min of the diff planes ->
 *    MergeUVCoefs/ExtendCoefs -> kf_and_coefs(combe^diff) ->
 *    ApplyUVCoefs.  Only the two kernels that live in MergeStatic.cu
 *    (kf_min_frames, kf_and_coefs) are ported here; CompareFields /
 *    MergeUVCoefs / ExtendCoefs / ApplyUVCoefs belong to the separate
 *    CombingAnalyze/KFMFilterBase machinery (a later milestone), so the
 *    KAnalyzeStatic *pipeline* is not assembled here — its sub-kernels are
 *    verified individually.  (KAnalyzeStatic also demands YV12 subsampling.)
 *  - kf_and_coefs is float32 with no FMA contraction.  Upstream's real CUDA
 *    device build may contract `a*b+c` into fma; the mirror/golden here (and a
 *    faithful OpenCL run) keep the exact structure with no contraction, which
 *    can differ by <1 ulp at rounding boundaries — a // RIG-VERIFY item.
 *    invcombe/invdiff are the host constants 1.0f/thcombe, 1.0f/thdiff.
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

/* ---------------------------------------------------------------------------
 * kf_compare_frames — KTemporalDiff core (cpu_compare_frames / kl_compare_frames
 * twin).  Per pixel: dst = max(f0..f4) - min(f0..f4)  (the temporal spread across
 * the 5 frames [n-2..n+2]); a large value flags moving/transient content.
 * Grid: 2D (width, height).  Per plane (Y, then U, then V).
 * // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_compare_frames(
    __global       PX* __restrict dst,
    __global const PX* __restrict src0,
    __global const PX* __restrict src1,
    __global const PX* __restrict src2,
    __global const PX* __restrict src3,
    __global const PX* __restrict src4,
    int width, int height, int pitch)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int off = x + y * pitch;
        int a0 = (int)src0[off], a1 = (int)src1[off], a2 = (int)src2[off];
        int a3 = (int)src3[off], a4 = (int)src4[off];
        int mn = a0, mx = a0;
        if (a1 < mn) mn = a1; if (a1 > mx) mx = a1;
        if (a2 < mn) mn = a2; if (a2 > mx) mx = a2;
        if (a3 < mn) mn = a3; if (a3 > mx) mx = a3;
        if (a4 < mn) mn = a4; if (a4 > mx) mx = a4;
        dst[off] = (PX)(mx - mn);
    }
}

/* ---------------------------------------------------------------------------
 * kf_min_frames — KAnalyzeStatic temporal-diff sub-step (cpu_min_frames /
 * kl_min_frames twin).  Per pixel: dst = min(f0,f1,f2) of the 3 temporal-diff
 * frames [n-1..n+1]; pixels that are static in time keep a low value.
 * Grid: 2D (width, height).  // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_min_frames(
    __global       PX* __restrict dst,
    __global const PX* __restrict src0,
    __global const PX* __restrict src1,
    __global const PX* __restrict src2,
    int width, int height, int pitch)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int off = x + y * pitch;
        int a0 = (int)src0[off], a1 = (int)src1[off], a2 = (int)src2[off];
        int mn = a0;
        if (a1 < mn) mn = a1;
        if (a2 < mn) mn = a2;
        dst[off] = (PX)mn;
    }
}

/* ---------------------------------------------------------------------------
 * kf_and_coefs — KAnalyzeStatic combing ^ static sub-step (kl_and_coefs twin).
 * dstp is the combing-coefficient plane (read-modify-write); diffp is the
 * temporal-diff coefficient plane.  Both coefficient planes carry 8-bit-ish
 * flag values; thcombe/thdiff scale them (invdiff negated):
 *   combe = clamp(val(dstp)*invcombe - 1.0f, -0.5f,  0.5f)
 *   diffc = clamp(val(diffp)*(-invdiff) + 1.0f, -0.5f,  0.5f)
 *   tmp   = max(combe + diffc, 0.0f) * 128.0f + 0.5f
 *   dstp  = trunc(tmp)                       // cast in [0,128]
 * Result coefficient feeds kf_merge_static: only where the frame is both
 * non-combed AND static does tmp exceed ~0, blending toward the 30fps field.
 * Grid: 2D (width, height) over the Y plane (uv handled by ApplyUVCoefs).
 * // ALG-VERIFIED (independent float32 mirror, no FMA — see header note).
 * -------------------------------------------------------------------------*/
kernel void kf_and_coefs(
    __global PX* __restrict dstp,
    __global const PX* __restrict diffp,
    int width, int height, int pitch,
    float invcombe, float invdiff)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int off = x + y * pitch;
        float combe = (float)dstp[off] * invcombe + (-1.0f);
        float diffc = (float)diffp[off] * (-invdiff) + 1.0f;
        if (combe < -0.5f) combe = -0.5f; else if (combe > 0.5f) combe = 0.5f;
        if (diffc < -0.5f) diffc = -0.5f; else if (diffc > 0.5f) diffc = 0.5f;
        float s = combe + diffc;
        if (s < 0.0f) s = 0.0f;
        float tmp = s * 128.0f + 0.5f;
        dstp[off] = (PX)(int)tmp;
    }
}

/* ---------------------------------------------------------------------------
 * kf_merge_static — KMergeStatic core (cpu_merge_static / kl_merge_static twin).
 * Per pixel: coef = flag value ([0,128]; 0 => keep the 60fps field, 128 => take
 * the 30fps field), blend the two interlaced-origin frames:
 *   dst = (coef*v30 + (128-coef)*v60 + 64) >> 7
 * Grid: 2D (width, height).  Per plane.  Host has already copied src60 into dst
 * (unwritten / static pixels simply remain the 60fps frame).
 * // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_merge_static(
    __global       PX* __restrict dst,
    __global const PX* __restrict src60,
    __global const PX* __restrict src30,
    __global const PX* __restrict flag,
    int width, int height, int pitch)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int off = x + y * pitch;
        int coef = (int)flag[off];
        int v30  = (int)src30[off];
        int v60  = (int)src60[off];
        /* coef assumed in [0,128] so (128-coef)>=0 and the shift is safe. */
        int tmp = (coef * v30 + (128 - coef) * v60 + 64) >> 7;
        dst[off] = (PX)tmp;
    }
}
