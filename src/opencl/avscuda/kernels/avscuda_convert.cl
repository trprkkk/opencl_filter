/* ============================================================================
 * avscuda_convert.cl — OpenCL port of the AvsCUDA bit-depth conversion
 * kernels in `AvsCUDA/filters/Convert.cu` (rigaya/AviSynthCUDAFilters, MIT):
 * ConvertBits/ConvertTo8bit/ConvertTo16bit/ConvertToFloat.
 *
 * Template params become runtime args (shift / tgt_bits / src_bits / chroma);
 * the five templates split by element width into the ten kernels below (the
 * upstream BitsToType rule: 8 bit <-> uint8_t, anything else <-> uint16_t,
 * 32 bit <-> float).  Host dispatch (get_cuda_conv_bits, same file):
 *   SRC==TGT -> Copy (host API, no kernel); SRC==32 -> from_float;
 *   TGT==32 -> to_float; SRC<TGT -> higher; else dither ? dither : no_dither.
 * SRC_BITS in {8,10,12,14,16,32}, TGT_BITS in {8,10,12,14,16,32}; dither shifts
 * (SRC-TGT) are always even and in {2,4,6,8}, so Dither<2/4/6/8> below cover
 * every producible case (anything else yields 0, like the default Dither<>).
 *
 * Dither tables are verbatim copies of c_dither2/4/6/8 (Bayer ordered).
 * DITHER_W = 1<<(SHIFT>>1), MASK = W-1; pixel (x,y) adds table[y&MASK][x&MASK].
 *
 * Notes transcribed with the kernels:
 * - no_dither is PLAIN TRUNCATION (src>>SHIFT): the `+ HALF` rounding line is
 *   commented out upstream (and the HALF enum is dead).  NOT a bug to fix.
 * - from_float clamps with the rgy_util.h clamp MACRO
 *   ((x<=h)?((x>=l)?x:l):h) — NaN compares false and yields MAX_VAL (OpenCL's
 *   builtin clamp() would yield 0 for NaN, so the macro is spelled out).
 *   The (TGT_TYPE) cast then truncates toward zero; outputs are always in
 *   [0, MAX_VAL], NaN included.  16-bit MAX_VAL is 65280, not 65535.
 * - to_float chroma is ((float)v - HALF) * FACTOR with FACTOR = 1.0f/MAX_VAL
 *   (the upstream (float) cast around the subtraction is a no-op: int - float
 *   is already float).  Unfused float32 throughout — the host build must
 *   disable FP contraction for bit-exactness.
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

__constant uchar c_dither2[2][2] = {
    { 0, 2 },
    { 3, 1 }
};

__constant uchar c_dither4[4][4] = {
    { 0,  8,  2, 10 },
    { 12,  4, 14,  6 },
    { 3, 11,  1,  9 },
    { 15,  7, 13,  5 }
};

__constant uchar c_dither6[8][8] = {
    { 0, 32,  8, 40,  2, 34, 10, 42 },
    { 48, 16, 56, 24, 50, 18, 58, 26 },
    { 12, 44,  4, 36, 14, 46,  6, 38 },
    { 60, 28, 52, 20, 62, 30, 54, 22 },
    { 3, 35, 11, 43,  1, 33,  9, 41 },
    { 51, 19, 59, 27, 49, 17, 57, 25 },
    { 15, 47,  7, 39, 13, 45,  5, 37 },
    { 63, 31, 55, 23, 61, 29, 53, 21 }
};

__constant uchar c_dither8[16][16] = {
    { 0,192, 48,240, 12,204, 60,252,  3,195, 51,243, 15,207, 63,255 },
    { 128, 64,176,112,140, 76,188,124,131, 67,179,115,143, 79,191,127 },
    { 32,224, 16,208, 44,236, 28,220, 35,227, 19,211, 47,239, 31,223 },
    { 160, 96,144, 80,172,108,156, 92,163, 99,147, 83,175,111,159, 95 },
    { 8,200, 56,248,  4,196, 52,244, 11,203, 59,251,  7,199, 55,247 },
    { 136, 72,184,120,132, 68,180,116,139, 75,187,123,135, 71,183,119 },
    { 40,232, 24,216, 36,228, 20,212, 43,235, 27,219, 39,231, 23,215 },
    { 168,104,152, 88,164,100,148, 84,171,107,155, 91,167,103,151, 87 },
    { 2,194, 50,242, 14,206, 62,254,  1,193, 49,241, 13,205, 61,253 },
    { 130, 66,178,114,142, 78,190,126,129, 65,177,113,141, 77,189,125 },
    { 34,226, 18,210, 46,238, 30,222, 33,225, 17,209, 45,237, 29,221 },
    { 162, 98,146, 82,174,110,158, 94,161, 97,145, 81,173,109,157, 93 },
    { 10,202, 58,250,  6,198, 54,246,  9,201, 57,249,  5,197, 53,245 },
    { 138, 74,186,122,134, 70,182,118,137, 73,185,121,133, 69,181,117 },
    { 42,234, 26,218, 38,230, 22,214, 41,233, 25,217, 37,229, 21,213 },
    { 170,106,154, 90,166,102,150, 86,169,105,153, 89,165,101,149, 85 }
};

/* Dither-table lookup shared by the two dither kernels (Dither<SHIFT>::get
 * with the (1<<(SHIFT>>1))-1 mask; unproducible shifts yield 0). */
inline int ka_dither_get(int shift, int x, int y)
{
    int m = (1 << (shift >> 1)) - 1;
    int xx = x & m, yy = y & m;
    if (shift == 2) return (int)c_dither2[yy][xx];
    if (shift == 4) return (int)c_dither4[yy][xx];
    if (shift == 6) return (int)c_dither6[yy][xx];
    if (shift == 8) return (int)c_dither8[yy][xx];
    return 0;
}

/* ---------------------------------------------------------------------------
 * ka_convert_lower_dither_u8 / _u16 — ordered-dither down-convert
 * (kl_convert_to_lower_bits_dither twins; src ushort, dst uchar/ushort).
 * Grid: 2D (width, height).
 * // ALG-VERIFIED via python/run_avscuda_convert.py.
 * -------------------------------------------------------------------------*/
kernel void ka_convert_lower_dither_u8(
    __global       uchar* __restrict dst, int dst_pitch,
    __global const ushort* __restrict src, int src_pitch,
    int width, int height, int shift, int tgt_bits)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    int tmp = ((int)src[x + y * src_pitch] + ka_dither_get(shift, x, y)) >> shift;
    dst[x + y * dst_pitch] = (uchar)min(tmp, (1 << tgt_bits) - 1);
}

kernel void ka_convert_lower_dither_u16(
    __global       ushort* __restrict dst, int dst_pitch,
    __global const ushort* __restrict src, int src_pitch,
    int width, int height, int shift, int tgt_bits)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    int tmp = ((int)src[x + y * src_pitch] + ka_dither_get(shift, x, y)) >> shift;
    dst[x + y * dst_pitch] = (ushort)min(tmp, (1 << tgt_bits) - 1);
}

/* ---------------------------------------------------------------------------
 * ka_convert_lower_nodither_u8 / _u16 — truncating down-convert
 * (kl_convert_to_lower_bits_no_dither twins; plain src>>SHIFT, min-clamped).
 * Grid: 2D (width, height).
 * // ALG-VERIFIED via python/run_avscuda_convert.py.
 * -------------------------------------------------------------------------*/
kernel void ka_convert_lower_nodither_u8(
    __global       uchar* __restrict dst, int dst_pitch,
    __global const ushort* __restrict src, int src_pitch,
    int width, int height, int shift, int tgt_bits)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    int tmp = ((int)src[x + y * src_pitch]) >> shift;
    dst[x + y * dst_pitch] = (uchar)min(tmp, (1 << tgt_bits) - 1);
}

kernel void ka_convert_lower_nodither_u16(
    __global       ushort* __restrict dst, int dst_pitch,
    __global const ushort* __restrict src, int src_pitch,
    int width, int height, int shift, int tgt_bits)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    int tmp = ((int)src[x + y * src_pitch]) >> shift;
    dst[x + y * dst_pitch] = (ushort)min(tmp, (1 << tgt_bits) - 1);
}

/* ---------------------------------------------------------------------------
 * ka_convert_higher_from_u8 / _from_u16 — up-convert by left shift
 * (kl_convert_to_higher_bits twins; dst ushort, min-clamped).
 * Grid: 2D (width, height).
 * // ALG-VERIFIED via python/run_avscuda_convert.py.
 * -------------------------------------------------------------------------*/
kernel void ka_convert_higher_from_u8(
    __global       ushort* __restrict dst, int dst_pitch,
    __global const uchar* __restrict src, int src_pitch,
    int width, int height, int shift, int tgt_bits)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    int tmp = ((int)src[x + y * src_pitch]) << shift;
    dst[x + y * dst_pitch] = (ushort)min(tmp, (1 << tgt_bits) - 1);
}

kernel void ka_convert_higher_from_u16(
    __global       ushort* __restrict dst, int dst_pitch,
    __global const ushort* __restrict src, int src_pitch,
    int width, int height, int shift, int tgt_bits)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    int tmp = ((int)src[x + y * src_pitch]) << shift;
    dst[x + y * dst_pitch] = (ushort)min(tmp, (1 << tgt_bits) - 1);
}

/* ---------------------------------------------------------------------------
 * ka_convert_from_float_u8 / _u16 — float plane to int
 * (kl_convert_from_float twins; chroma adds HALF; rgy clamp macro spelled
 * out so NaN yields MAX_VAL exactly like upstream).
 * Grid: 2D (width, height).
 * // ALG-VERIFIED via python/run_avscuda_convert.py.
 * -------------------------------------------------------------------------*/
kernel void ka_convert_from_float_u8(
    __global       uchar* __restrict dst, int dst_pitch,
    __global const float* __restrict src, int src_pitch,
    int width, int height, int tgt_bits, int chroma)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    float max_val = (float)(255 << (tgt_bits - 8));
    float half = (float)(128 << (tgt_bits - 8));
    float s = src[x + y * src_pitch];
    float tmp = (chroma != 0) ? (s * max_val + half + 0.5f)
                              : (s * max_val + 0.5f);
    float c = (tmp <= max_val) ? ((tmp >= 0.0f) ? tmp : 0.0f) : max_val;
    dst[x + y * dst_pitch] = (uchar)(int)c;
}

kernel void ka_convert_from_float_u16(
    __global       ushort* __restrict dst, int dst_pitch,
    __global const float* __restrict src, int src_pitch,
    int width, int height, int tgt_bits, int chroma)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    float max_val = (float)(255 << (tgt_bits - 8));
    float half = (float)(128 << (tgt_bits - 8));
    float s = src[x + y * src_pitch];
    float tmp = (chroma != 0) ? (s * max_val + half + 0.5f)
                              : (s * max_val + 0.5f);
    float c = (tmp <= max_val) ? ((tmp >= 0.0f) ? tmp : 0.0f) : max_val;
    dst[x + y * dst_pitch] = (ushort)(int)c;
}

/* ---------------------------------------------------------------------------
 * ka_convert_to_float_from_u8 / _from_u16 — int plane to float
 * (kl_convert_to_float twins; chroma subtracts HALF first).
 * Grid: 2D (width, height).
 * // ALG-VERIFIED via python/run_avscuda_convert.py.
 * -------------------------------------------------------------------------*/
kernel void ka_convert_to_float_from_u8(
    __global       float* __restrict dst, int dst_pitch,
    __global const uchar* __restrict src, int src_pitch,
    int width, int height, int src_bits, int chroma)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    float max_val = (float)(255 << (src_bits - 8));
    float factor = 1.0f / max_val;
    float half = (float)(128 << (src_bits - 8));
    float v = (float)src[x + y * src_pitch];
    dst[x + y * dst_pitch] = (chroma != 0) ? ((v - half) * factor)
                                           : (v * factor);
}

kernel void ka_convert_to_float_from_u16(
    __global       float* __restrict dst, int dst_pitch,
    __global const ushort* __restrict src, int src_pitch,
    int width, int height, int src_bits, int chroma)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    float max_val = (float)(255 << (src_bits - 8));
    float factor = 1.0f / max_val;
    float half = (float)(128 << (src_bits - 8));
    float v = (float)src[x + y * src_pitch];
    dst[x + y * dst_pitch] = (chroma != 0) ? ((v - half) * factor)
                                           : (v * factor);
}
