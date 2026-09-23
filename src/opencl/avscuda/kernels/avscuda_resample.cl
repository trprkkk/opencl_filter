/* ============================================================================
 * avscuda_resample.cl — OpenCL port of the AvsCUDA resizer kernels in
 * `AvsCUDA/filters/resample.cu` (rigaya/AviSynthCUDAFilters, MIT):
 * FilteredResizeH/V device paths.
 *
 * The four templates split by element width into the eight kernels below.
 * Lanes are independent (same program, per-lane accumulation), so the scalar
 * ports are lane-identical; packed units (uchar2/3/4, ushort3/4) become an
 * element stream with a unit stride (h) or a plain byte/element stream (v —
 * upstream runs v over uchar4/ushort4/float4 vectors of the row bytes).
 *
 * Two host/device optimizations are ELIDED (values identical, documented):
 * - v_planar's per-row __shared__ coeff tile: the port reads
 *   coeff[y*filter_size+i] straight from global (same floats).
 * - h_planar's TRANSPOSED + tiled coeff staging (make_h_coeff_for_cuda):
 *   the port takes the LOGICAL program coeff[x*filter_size+i] (the layout
 *   of ResamplingProgram::pixel_coefficient_float) and reads it directly —
 *   the host must pass the untransposed program.
 * No work-group-size or __local constraints remain; filter_size is unbounded
 * (upstream is shared-mem-capped).
 *
 * Arithmetic (transcribed verbatim): int pixels accumulate from 0.5f (the
 * rounder) with unfused per-tap MUL+ADD, then clamp to [0,limit] and C-cast
 * truncate (limit = (1<<bits)-1 as float; NaN yields limit via the fmin/fmax
 * rule, matching CUDA).  Float pixels accumulate from 0.0f, unclamped.
 * Host build must disable FP contraction for bit-exactness (upstream nvcc
 * default --fmad=true may fuse MUL+ADD — cross-toolchain device comparison
 * needs rounding-level tolerance).  Host dispatch: filter_size == 1 routes
 * to the pointresize kernels (the planar kernels still accept fs == 1).
 * Pitches are in elements (h_pointresize_bytes: bytes, like upstream).
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

/* ---------------------------------------------------------------------------
 * ka_resize_v_pointresize / _f32 — row select (kl_resize_v_planar_pointresize
 * twins; uint8/uint16/float; limit/filter_size ignored upstream too).
 * Grid: 2D (target_width, target_height).
 * // ALG-VERIFIED via python/run_avscuda_resample.py.
 * -------------------------------------------------------------------------*/
kernel void ka_resize_v_pointresize(
    __global       PX* __restrict dst,
    __global const PX* __restrict src, int dst_pitch, int src_pitch,
    __global const int* __restrict pixel_offset,
    int target_width, int target_height, float limit, int filter_size)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= target_width || y >= target_height) return;
    (void)limit; (void)filter_size;
    dst[x + y * dst_pitch] = src[x + pixel_offset[y] * src_pitch];
}

kernel void ka_resize_v_pointresize_f32(
    __global       float* __restrict dst,
    __global const float* __restrict src, int dst_pitch, int src_pitch,
    __global const int* __restrict pixel_offset,
    int target_width, int target_height, float limit, int filter_size)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= target_width || y >= target_height) return;
    (void)limit; (void)filter_size;
    dst[x + y * dst_pitch] = src[x + pixel_offset[y] * src_pitch];
}

/* ---------------------------------------------------------------------------
 * ka_resize_v_planar / _f32 — vertical filter (kl_resize_v_planar twins).
 * Grid: 2D (target_width, target_height).
 * // ALG-VERIFIED via python/run_avscuda_resample.py.
 * -------------------------------------------------------------------------*/
kernel void ka_resize_v_planar(
    __global       PX* __restrict dst,
    __global const PX* __restrict src, int dst_pitch, int src_pitch,
    __global const int* __restrict pixel_offset,
    __global const float* __restrict pixel_coefficient,
    int target_width, int target_height, float limit, int filter_size)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= target_width || y >= target_height) return;
    float result = 0.5f;
    int y_offset = pixel_offset[y];
    for (int i = 0; i < filter_size; ++i) {
        result += pixel_coefficient[y * filter_size + i] *
                  (float)src[x + (y_offset + i) * src_pitch];
    }
    float c = max(0.0f, min(result, limit));
    dst[x + y * dst_pitch] = (PX)(int)c;
}

kernel void ka_resize_v_planar_f32(
    __global       float* __restrict dst,
    __global const float* __restrict src, int dst_pitch, int src_pitch,
    __global const int* __restrict pixel_offset,
    __global const float* __restrict pixel_coefficient,
    int target_width, int target_height, float limit, int filter_size)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= target_width || y >= target_height) return;
    (void)limit;
    float result = 0.0f;
    int y_offset = pixel_offset[y];
    for (int i = 0; i < filter_size; ++i) {
        result += pixel_coefficient[y * filter_size + i] *
                  src[x + (y_offset + i) * src_pitch];
    }
    dst[x + y * dst_pitch] = result;
}

/* ---------------------------------------------------------------------------
 * ka_resize_h_pointresize_bytes — unit select over a byte stream
 * (kl_resize_h_pointresize twin; covers unit sizes 1/2/3/4/6/8 via unit_bytes;
 * byte-identical to whole-unit copy; pitches in bytes; width = units).
 * Grid: 2D (target_width * unit_bytes, target_height).
 * // ALG-VERIFIED via python/run_avscuda_resample.py.
 * -------------------------------------------------------------------------*/
kernel void ka_resize_h_pointresize_bytes(
    __global       uchar* __restrict dst,
    __global const uchar* __restrict src, int dst_pitch, int src_pitch,
    __global const int* __restrict pixel_offset, int unit_bytes,
    int target_width, int target_height, float limit, int filter_size)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= target_width * unit_bytes || y >= target_height) return;
    (void)limit; (void)filter_size;
    int unit = x / unit_bytes, lane = x % unit_bytes;
    dst[x + y * dst_pitch] =
        src[pixel_offset[unit] * unit_bytes + lane + y * src_pitch];
}

/* ---------------------------------------------------------------------------
 * ka_resize_h_planar_u8 / _u16 / _f32 — horizontal filter over an element
 * stream (kl_resize_h_planar twins; unit_elems = elements per packed unit,
 * 1 for planar; upstream float is scalar-only so f32 always runs unit 1).
 * Grid: 2D (target_width * unit_elems, target_height).
 * // ALG-VERIFIED via python/run_avscuda_resample.py.
 * -------------------------------------------------------------------------*/
kernel void ka_resize_h_planar_u8(
    __global       uchar* __restrict dst,
    __global const uchar* __restrict src, int dst_pitch, int src_pitch,
    __global const int* __restrict pixel_offset,
    __global const float* __restrict pixel_coefficient, int unit_elems,
    int target_width, int target_height, float limit, int filter_size)
{
    int e = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (e >= target_width * unit_elems || y >= target_height) return;
    int unit = e / unit_elems, lane = e % unit_elems;
    float result = 0.5f;
    int x_offset = pixel_offset[unit];
    for (int i = 0; i < filter_size; ++i) {
        result += pixel_coefficient[unit * filter_size + i] *
                  (float)src[(x_offset + i) * unit_elems + lane + y * src_pitch];
    }
    float c = max(0.0f, min(result, limit));
    dst[e + y * dst_pitch] = (uchar)(int)c;
}

kernel void ka_resize_h_planar_u16(
    __global       ushort* __restrict dst,
    __global const ushort* __restrict src, int dst_pitch, int src_pitch,
    __global const int* __restrict pixel_offset,
    __global const float* __restrict pixel_coefficient, int unit_elems,
    int target_width, int target_height, float limit, int filter_size)
{
    int e = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (e >= target_width * unit_elems || y >= target_height) return;
    int unit = e / unit_elems, lane = e % unit_elems;
    float result = 0.5f;
    int x_offset = pixel_offset[unit];
    for (int i = 0; i < filter_size; ++i) {
        result += pixel_coefficient[unit * filter_size + i] *
                  (float)src[(x_offset + i) * unit_elems + lane + y * src_pitch];
    }
    float c = max(0.0f, min(result, limit));
    dst[e + y * dst_pitch] = (ushort)(int)c;
}

kernel void ka_resize_h_planar_f32(
    __global       float* __restrict dst,
    __global const float* __restrict src, int dst_pitch, int src_pitch,
    __global const int* __restrict pixel_offset,
    __global const float* __restrict pixel_coefficient, int unit_elems,
    int target_width, int target_height, float limit, int filter_size)
{
    int e = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (e >= target_width * unit_elems || y >= target_height) return;
    int unit = e / unit_elems, lane = e % unit_elems;
    (void)limit;
    float result = 0.0f;
    int x_offset = pixel_offset[unit];
    for (int i = 0; i < filter_size; ++i) {
        result += pixel_coefficient[unit * filter_size + i] *
                  src[(x_offset + i) * unit_elems + lane + y * src_pitch];
    }
    dst[e + y * dst_pitch] = result;
}
