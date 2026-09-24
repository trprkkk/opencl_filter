/* ============================================================================
 * masktools_lut.cl — OpenCL port of the complete CUDA surface of upstream
 * rigaya/masktools (submodule of AviSynthCUDAFilters @68aef6e, masktools
 * @24ba826, GPL).  The submodule contains exactly five __global__ kernels,
 * in two files:
 *   common/functions/functions_cuda.cu
 *     kl_fill (:16)  -> km_fill / km_fill_f32
 *     kl_copy (:62)  -> km_copy_bytes
 *   masktools/filters/lut/lut_kernel.cu
 *     kl_lut_x   (:15) -> km_lut_x
 *     kl_lut_xy  (:48) -> km_lut_xy
 *     kl_lut_xyz (:88) -> km_lut_xyz
 *
 * All five are elementwise with 2D bounds guards: no __local, no
 * reductions, no work-group-size constraint.  Upstream launches (32,16)
 * for fill/copy and (16,8) for the LUTs; any decomposition is equivalent.
 *
 * Vector convention: upstream works on uchar4/ushort4/float4, so widths and
 * pitches are in 4-PIXEL VECTORS (`width4`, `pitch4`) and this port keeps
 * those units, indexing 4 scalar lanes per work item.  NOTE the host-side
 * truncation upstream performs and this port inherits: `w4 = width >> 2`
 * (and `rowsize >> 2` for the copy), so a width that is not a multiple of
 * 4 leaves the tail pixels UNTOUCHED.
 *
 * -- UPSTREAM DEFECT, transcribed faithfully --------------------------------
 * The CUDA LUT dispatcher `lut_cuda_16` (lut_kernel.cu:157-165) instantiates
 * `bits_per_pixel = 8` for ALL of 10/12/14/16-bit, so the 16-bit kernels get
 * shift 8 and `mask = (1 << 8) - 1 = 255`, while the host builds the table
 * with the REAL depth (`idx = (x << bits_per_pixel) + y` over `1 << bits`
 * entries, lut_data.cpp:25-30) and the CPU path indexes it that way
 * (lutxy.cpp:30).  Consequences on the CUDA path, all reproduced here
 * because this is a transcription, not a fix:
 *   - lut_x   16-bit: reads lut[X & 255] — only the first 256 entries.
 *   - lut_xy  16-bit: ((X << 8) + Y) & 255 == Y & 255 — X is dropped
 *     entirely and the result depends only on the second clip.
 *   - lut_xyz 16-bit: likewise collapses to Z & 255 (and the 16-bit
 *     3-input table is not even built upstream — lut_data.cpp:32-42 has
 *     `case 3` commented out).
 * The port takes `lut_bits` and `mask` as arguments so the host decides;
 * a host mirroring upstream passes 8/255 for every 16-bit depth.  Flagged
 * for a rig-side decision — do NOT "fix" it here without evidence.
 * // ALG-VERIFIED via python/run_masktools_lut.py.
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

/* ---------------------------------------------------------------------------
 * km_fill — kl_fill twin (Functions::memset_plane*_cuda).  Every lane of
 * the 4-pixel vector gets (PX)v, exactly like VHelper<vpixel_t>::make(v).
 * -------------------------------------------------------------------------*/
kernel void km_fill(
    __global PX* __restrict dst, int v, int width4, int height, int pitch4)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width4 && y < height) {
        int o = (x + y * pitch4) * 4;
        dst[o + 0] = (PX)v;
        dst[o + 1] = (PX)v;
        dst[o + 2] = (PX)v;
        dst[o + 3] = (PX)v;
    }
}

/* float4 variant (memset_plane_32_cuda); value passed as its bit pattern
 * so the host never depends on float argument conversion. */
kernel void km_fill_f32(
    __global float* __restrict dst, int v_bits, int width4, int height,
    int pitch4)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width4 && y < height) {
        float v = as_float(v_bits);
        int o = (x + y * pitch4) * 4;
        dst[o + 0] = v;
        dst[o + 1] = v;
        dst[o + 2] = v;
        dst[o + 3] = v;
    }
}

/* ---------------------------------------------------------------------------
 * km_copy_bytes — kl_copy twin (Functions::copy_plane_cuda).  Upstream
 * always instantiates uchar4 and derives w4 from a BYTE rowsize, so this
 * is a byte-granular copy independent of the pixel format; pitches are in
 * 4-byte units, like upstream's `pitch >> 2`.
 * -------------------------------------------------------------------------*/
kernel void km_copy_bytes(
    __global uchar* __restrict dst, int dst_pitch4,
    __global const uchar* __restrict src, int src_pitch4,
    int width4, int height)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width4 && y < height) {
        int d = (x + y * dst_pitch4) * 4;
        int s = (x + y * src_pitch4) * 4;
        dst[d + 0] = src[s + 0];
        dst[d + 1] = src[s + 1];
        dst[d + 2] = src[s + 2];
        dst[d + 3] = src[s + 3];
    }
}

/* mask is applied only on the 16-bit path, exactly as upstream branches on
 * sizeof(pixel_t) == 1 (compile-time folded here too). */
static int km_lut_idx(int idx, int mask)
{
    return (sizeof(PX) == 1) ? idx : (idx & mask);
}

/* ---------------------------------------------------------------------------
 * km_lut_x — kl_lut_x twin.  One shared pitch4 for src and dst (upstream
 * passes a single pitch).
 * -------------------------------------------------------------------------*/
kernel void km_lut_x(
    __global PX* __restrict dst, __global const PX* __restrict src,
    int pitch4, int width4, int height,
    __global const PX* __restrict lut, int mask)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width4 && y < height) {
        int o = (x + y * pitch4) * 4;
        for (int k = 0; k < 4; ++k)
            dst[o + k] = lut[km_lut_idx((int)src[o + k], mask)];
    }
}

/* ---------------------------------------------------------------------------
 * km_lut_xy — kl_lut_xy twin; index (X << lut_bits) + Y
 * (lut_index_xy<bits_per_pixel>).  Upstream only ever instantiates
 * lut_bits == 8 — including for 10/12/14/16-bit, see the header.
 * -------------------------------------------------------------------------*/
kernel void km_lut_xy(
    __global PX* __restrict dst,
    __global const PX* __restrict src0, __global const PX* __restrict src1,
    int pitch4, int width4, int height,
    __global const PX* __restrict lut, int mask, int lut_bits)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width4 && y < height) {
        int o = (x + y * pitch4) * 4;
        for (int k = 0; k < 4; ++k) {
            int idx = ((int)src0[o + k] << lut_bits) + (int)src1[o + k];
            dst[o + k] = lut[km_lut_idx(idx, mask)];
        }
    }
}

/* ---------------------------------------------------------------------------
 * km_lut_xyz — kl_lut_xyz twin; index (X << 2*lut_bits) + (Y << lut_bits) + Z.
 * -------------------------------------------------------------------------*/
kernel void km_lut_xyz(
    __global PX* __restrict dst,
    __global const PX* __restrict src0, __global const PX* __restrict src1,
    __global const PX* __restrict src2,
    int pitch4, int width4, int height,
    __global const PX* __restrict lut, int mask, int lut_bits)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width4 && y < height) {
        int o = (x + y * pitch4) * 4;
        for (int k = 0; k < 4; ++k) {
            int idx = ((int)src0[o + k] << (lut_bits * 2))
                    + ((int)src1[o + k] << lut_bits)
                    + (int)src2[o + k];
            dst[o + k] = lut[km_lut_idx(idx, mask)];
        }
    }
}
