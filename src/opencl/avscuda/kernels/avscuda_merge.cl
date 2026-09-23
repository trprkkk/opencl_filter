/* ============================================================================
 * avscuda_merge.cl — OpenCL port of the AvsCUDA weighted-merge / average
 * plane kernels in `AvsCUDA/filters/merge.cu` (rigaya/AviSynthCUDAFilters,
 * MIT): Merge/MergeChroma/MergeLuma planar paths.
 *
 * Upstream launches kl_merge_plane / kl_average_plane over uchar4/ushort4/
 * float4 vectors (width4 = (rowsize>>shift)+3)>>2).  Every lane is an
 * independent per-pixel op, so the scalar translation below is lane-identical.
 * Both kernels are IN-PLACE on srcp (otherp is read-only) with DUAL pitches,
 * transcribed faithfully here.
 *
 * Host dispatch contract (merge_plane / MergeAll::GetFrame, same file):
 *   weight in (0.4961, 0.5039) -> kl_average_plane (weight ignored)
 *   weight < 0.0039            -> return src untouched (no kernel)
 *   weight > 0.9961            -> plain copy of the 2nd clip (no kernel)
 *   else                       -> kl_merge_plane with
 *     weight_i = (int)(weight*32767.0f + 0.5f), invweight_i = 32767-weight_i
 * YUY2 has no CUDA path (MergeChroma throws on CUDA) — host/SIMD only.
 *
 * Integer rounding (SIMD/device scale, NOT the scalar C fallback scale):
 *   (a*invw + b*w + 16384) >> 15, then a C narrowing cast (mod 2^bits —
 *   VHelper::cast_to is a plain (uchar)/(ushort) cast, no saturation).
 *   The scalar-C twin weighted_merge_planar_c uses (+32768)>>16 with a
 *   *65535-based weight — a different scale, so it is NOT a twin; the
 *   SSE2/AVX2 fallbacks use this >>15 formula.  Valid production weights
 *   (w+iw = 32767) can never overflow int32: a*iw+b*w <= 65535*32767.
 * Float: a*invw + b*w (separate MUL+ADD — the host build must disable FP
 *   contraction for bit-exactness); average float is (a+b)*0.5f on device
 *   vs (a+b)/2.0f in average_plane_c_float (provably identical: both are
 *   correctly-rounded exact-halvings; the harness runs one form in the
 *   mirror and the other in the golden to prove it empirically).
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

/* ---------------------------------------------------------------------------
 * ka_merge — in-place weighted plane merge, dual pitch (kl_merge_plane twin;
 * uchar4/ushort4 instantiations; float is ka_merge_f32).
 * Grid: 2D (width, height).
 * // ALG-VERIFIED via python/run_avscuda_merge.py.
 * -------------------------------------------------------------------------*/
kernel void ka_merge(
    __global       PX* __restrict srcp,
    __global const PX* __restrict otherp,
    int src_pitch, int other_pitch, int width, int height,
    int weight_i, int invweight_i)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    int a = (int)srcp[x + y * src_pitch];
    int b = (int)otherp[x + y * other_pitch];
    int tmp = (a * invweight_i + b * weight_i + 16384) >> 15;
    srcp[x + y * src_pitch] = (PX)tmp;
}

/* ---------------------------------------------------------------------------
 * ka_merge_f32 — float variant (kl_merge_plane<float4> twin;
 * weighted_merge_planar_c_float is the exact CPU twin, same op order).
 * Grid: 2D (width, height).
 * // ALG-VERIFIED via python/run_avscuda_merge.py.
 * -------------------------------------------------------------------------*/
kernel void ka_merge_f32(
    __global       float* __restrict srcp,
    __global const float* __restrict otherp,
    int src_pitch, int other_pitch, int width, int height,
    float weight_f, float invweight_f)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    float a = srcp[x + y * src_pitch];
    float b = otherp[x + y * other_pitch];
    srcp[x + y * src_pitch] = a * invweight_f + b * weight_f;
}

/* ---------------------------------------------------------------------------
 * ka_average — in-place plane average, dual pitch (kl_average_plane twin;
 * uchar4/ushort4; average_plane_c is the exact CPU twin:
 * (int(a)+b+1)>>1).  Host selects this for weight in (0.4961, 0.5039).
 * Grid: 2D (width, height).
 * // ALG-VERIFIED via python/run_avscuda_merge.py.
 * -------------------------------------------------------------------------*/
kernel void ka_average(
    __global       PX* __restrict srcp,
    __global const PX* __restrict otherp,
    int src_pitch, int other_pitch, int width, int height)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    int a = (int)srcp[x + y * src_pitch];
    int b = (int)otherp[x + y * other_pitch];
    srcp[x + y * src_pitch] = (PX)((a + b + 1) >> 1);
}

/* ---------------------------------------------------------------------------
 * ka_average_f32 — float variant (kl_average_plane<float4> twin; device form
 * (a+b)*0.5f, CPU form (a+b)/2.0f — identical results, see file header).
 * Grid: 2D (width, height).
 * // ALG-VERIFIED via python/run_avscuda_merge.py.
 * -------------------------------------------------------------------------*/
kernel void ka_average_f32(
    __global       float* __restrict srcp,
    __global const float* __restrict otherp,
    int src_pitch, int other_pitch, int width, int height)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    float a = srcp[x + y * src_pitch];
    float b = otherp[x + y * other_pitch];
    srcp[x + y * src_pitch] = (a + b) * 0.5f;
}
