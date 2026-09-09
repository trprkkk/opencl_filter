/* ============================================================================
 * kfm_deblock.cl — OpenCL port of the KFM `KDeblock` core kernel.
 *
 * Faithful port of rigaya/AviSynthCUDAFilters -> KFM/Deblock.cu (KFM is MIT).
 * KDeblock is a deblocking filter that (per 8x8 block and per QP-determined
 * threshold) applies an 8x8 DCT, a hard-threshold (requantisation) of the
 * AC coefficients, and an inverse DCT, accumulating `count = 1<<quality`
 * differently-offset 8x8 reconstructions of each block into a 16-bit plane.
 * This file ports the CUDA device kernel `kl_deblock` (the heart of the
 * filter); it reads a padded source plane + a per-block QP table and writes the
 * interleaved 16-bit accumulator that the later merge pass consumes.
 *
 * Kernel math (per block (bx,by), a faithful scalar transcription of
 * kl_deblock; channels/pixels are independent):
 *   local_out[16][16] = 0
 *   for ty in [0, count_minus_1]:              // count = 1<<quality
 *     (ox0,oy0) = g_deblock_offset[count_minus_1 + ty]
 *     load d[8][8] from src[(bx*8+ox0+x) + (by*8+oy0+y)*src_pitch]
 *     qp = qp_table[bx + by*qp_pitch]
 *     thresh = qp_apply_thresh(qp, thresh_a, thresh_b)*((1<<2)+strength) - 1
 *     dct8x8(d); hardthresh(d, thresh) [skips DC, index 0]; idct8x8(d)
 *     half = (1<<shift)>>1
 *     for each (x,y): tmp = clamp((int)(d[x+y*8] + half) >> shift, 0, maxv)
 *       local_out[oy0+y][ox0+x] += tmp
 *   off_z = (bx&1) + (by&1)*2                  // block-parity sub-plane
 *   write local_out[y][x] to out[(bx*8+x) + ((bh*off_z+by)*8+y)*out_pitch]
 *
 * The 8x8 DCT/IDCT is the fixed float32 butterfly of Devblock.cu (dev_dct8 /
 * dev_idct8, S1..S2 constants).  kl_deblock always runs DCT->hardthresh->IDCT;
 * the *CPU-only* fallback `cpu_deblock`/`cpu_deblock_avx` additionally has a
 * `thresh <= 0` identity shortcut (multiply by 64, skip the transform) that the
 * device kernel does NOT have — this OpenCL port follows the device kernel
 * (the transform path), which is what the CPU twins compute identically.
 *
 * Status:
 *   // ALG-VERIFIED (python/run_kfm_deblock.py, 300 cases) vs the CPU mirror
 *   // sim/kfm_deblock_ref.cpp (device-faithful kl_deblock scalar), float32.
 *   // Float work is IEEE float32 with no FMA contraction (mirror built
 *   // -ffp-contract=off; python golden emulates float32 per operation).
 *
 * Host seam (RIG-VERIFY): KDeblock::DeblockPlane first mirror-pads the plane
 * (8 px each side) into `src`, builds the QP table (kl_make_qp_table) and later
 * merges this accumulator (kl_merge_deblock) with a Bayer dither.  Those pad /
 * qp-table / merge steps are separate kernels/host glue; this file ports the
 * core `kl_deblock` only.  Because the CUDA block reads an 8x8 tile at an
 * offset up to +7, the src passed in must be the padded plane (a faithful host
 * config).  A block's offset index range [count_minus_1, 2*count_minus_1]
 * stays within g_deblock_offset[127] for quality <= 6 (count <= 64).
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

/* The output accumulator is 16-bit regardless of the input bit depth. */
#define OUTP ushort

/* DCT constants (Devblock.cu): sin/cos of the 8-point DCT with the *2 sqrt
 * scaling folded in, as float32 literals (identical to the .cu `#defines`). */
#define S1   0.19509032201612825f   /* sin(1*pi/(2*8)) */
#define C1   0.9807852804032304f    /* cos(1*pi/(2*8)) */
#define S3   0.5555702330196022f    /* sin(3*pi/(2*8)) */
#define C3   0.8314696123025452f    /* cos(3*pi/(2*8)) */
#define S2S6 1.3065629648763766f    /* sqrt(2)*sin(6*pi/(2*8)) */
#define S2C6 0.5411961001461971f    /* sqrt(2)*cos(6*pi/(2*8)) */
#define S2   1.4142135623730951f    /* sqrt(2) */

/* Per-quality list of `count` block offsets (Devblock.cu g_deblock_offset).
 * Index range used for a given quality is [count-1, 2*(count-1)]. */
static const int g_offx[127] = {
  0,0,4, 0,2,6,4, 0,5,2,7,4,1,6,3, 0,4,1,5,3,7,2,6,0,4,1,5,3,7,2,6,
  0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,
  0,4,0,4,2,6,2,6,0,4,0,4,2,6,2,6,1,5,1,5,3,7,3,7,1,5,1,5,3,7,3,7,
  0,4,0,4,2,6,2,6,0,4,0,4,2,6,2,6,1,5,1,5,3,7,3,7,1,5,1,5,3,7,3,7,
};
static const int g_offy[127] = {
  0,0,4, 0,2,4,6, 0,1,2,3,4,5,6,7, 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,
  0,2,4,6,1,3,5,7,0,2,4,6,1,3,5,7,0,2,4,6,1,3,5,7,0,2,4,6,1,3,5,7,
  0,4,4,0,2,6,6,2,2,6,6,2,0,4,4,0,1,5,5,1,3,7,7,3,3,7,7,3,1,5,5,1,
  1,5,5,1,3,7,7,3,3,7,7,3,1,5,5,1,0,4,4,0,2,6,6,2,2,6,6,2,0,4,4,0,
};

static float kf_clampf(float v, float lo, float hi)
{
    if (v < lo) v = lo; else if (v > hi) v = hi;
    return v;
}

/* qp_apply_thresh(qp, thresh_a, thresh_b) = clamp(thresh_a*qp+thresh_b,0,qp). */
static float kf_qp_thresh(int qp, float thresh_a, float thresh_b)
{
    return kf_clampf(((float)qp) * thresh_a + thresh_b, 0.0f, (float)qp);
}

/* One 1D 8-point DCT (dev_dct8<stride>) on d[0*stride..7*stride]. */
static void kf_dct8(float* d, int stride)
{
    float a0 = d[7*stride] + d[0*stride];
    float a1 = d[6*stride] + d[1*stride];
    float a2 = d[5*stride] + d[2*stride];
    float a3 = d[4*stride] + d[3*stride];
    float a4 = d[3*stride] - d[4*stride];
    float a5 = d[2*stride] - d[5*stride];
    float a6 = d[1*stride] - d[6*stride];
    float a7 = d[0*stride] - d[7*stride];
    float b0 = a3 + a0;
    float b1 = a2 + a1;
    float b2 = a1 - a2;
    float b3 = a0 - a3;
    float b4 = (S3 - C3) * a7 + C3 * (a4 + a7);
    float b5 = (S1 - C1) * a6 + C1 * (a5 + a6);
    float b6 = -(C1 + S1) * a5 + C1 * (a5 + a6);
    float b7 = -(C3 + S3) * a4 + C3 * (a4 + a7);
    float c0 = b1 + b0;
    float c1 = b0 - b1;
    float c2 = (S2S6 - S2C6) * b3 + S2C6 * (b2 + b3);
    float c3 = -(S2C6 + S2S6) * b2 + S2C6 * (b2 + b3);
    float c4 = b6 + b4;
    float c5 = b7 - b5;
    float c6 = b4 - b6;
    float c7 = b5 + b7;
    float d4 = c7 - c4;
    float d5 = c5 * S2;
    float d6 = c6 * S2;
    float d7 = c4 + c7;
    d[0*stride] = c0;
    d[4*stride] = c1;
    d[2*stride] = c2;
    d[6*stride] = c3;
    d[7*stride] = d4;
    d[3*stride] = d5;
    d[5*stride] = d6;
    d[1*stride] = d7;
}

/* One 1D 8-point inverse DCT (dev_idct8<stride>). */
static void kf_idct8(float* d, int stride)
{
    float c0 = d[0*stride];
    float c1 = d[4*stride];
    float c2 = d[2*stride];
    float c3 = d[6*stride];
    float d4 = d[7*stride];
    float d5 = d[3*stride];
    float d6 = d[5*stride];
    float d7 = d[1*stride];
    float c4 = d7 - d4;
    float c5 = d5 * S2;
    float c6 = d6 * S2;
    float c7 = d4 + d7;
    float b0 = c1 + c0;
    float b1 = c0 - c1;
    float b2 = -(S2C6 + S2S6) * c3 + S2C6 * (c2 + c3);
    float b3 = (S2S6 - S2C6) * c2 + S2C6 * (c2 + c3);
    float b4 = c6 + c4;
    float b5 = c7 - c5;
    float b6 = c4 - c6;
    float b7 = c5 + c7;
    float a0 = b3 + b0;
    float a1 = b2 + b1;
    float a2 = b1 - b2;
    float a3 = b0 - b3;
    float a4 = -(C3 + S3) * b7 + C3 * (b4 + b7);
    float a5 = -(C1 + S1) * b6 + C1 * (b5 + b6);
    float a6 = (S1 - C1) * b5 + C1 * (b5 + b6);
    float a7 = (S3 - C3) * b4 + C3 * (b4 + b7);
    d[0*stride] = a7 + a0;
    d[1*stride] = a6 + a1;
    d[2*stride] = a5 + a2;
    d[3*stride] = a4 + a3;
    d[4*stride] = a3 - a4;
    d[5*stride] = a2 - a5;
    d[6*stride] = a1 - a6;
    d[7*stride] = a0 - a7;
}

/* Forward 8x8 DCT on a row-major float[64] (cpu_dct8x8 = dev_dct8x8 layout). */
static void kf_dct8x8(float* d)
{
    for (int i = 0; i < 8; ++i) kf_dct8(d + i * 8, 1);  /* rows   */
    for (int i = 0; i < 8; ++i) kf_dct8(d + i, 8);      /* cols   */
}
static void kf_idct8x8(float* d)
{
    for (int i = 0; i < 8; ++i) kf_idct8(d + i, 8);      /* cols   */
    for (int i = 0; i < 8; ++i) kf_idct8(d + i * 8, 1);  /* rows   */
}
static void kf_hardthresh(float* d, float threshold)
{
    for (int i = 1; i < 64; ++i) {           /* index 0 = DC is never thresholded */
        if (d[i] < -threshold || d[i] > threshold) continue;
        d[i] = 0.0f;
    }
}

/* ---------------------------------------------------------------------------
 * kf_deblock — KDeblock core (kl_deblock twin).  One work item per 8x8 block
 * (bx,by), grid (bw,bh); bw = (width+15)>>3 etc. of the padded plane.
 * Grid: 2D (bw,bh).  dst is a 16-bit accumulator plane (USHORT per pixel).
 * // ALG-VERIFIED on a self-consistent padded-src config; the mirror-pad / qp
 * // table / merge host steps are separate (see header).
 * -------------------------------------------------------------------------*/
kernel void kf_deblock(
    __global const PX* __restrict src, int src_pitch,
    int bw, int bh,
    __global       OUTP* __restrict out, int out_pitch,
    __global const OUTP* __restrict qp_table, int qp_pitch,
    int count_minus_1, int shift, int maxv,
    float strength, float thresh_a, float thresh_b)
{
    int bx = (int)get_global_id(0);
    int by = (int)get_global_id(1);
    if (bx >= bw || by >= bh) return;

    int local_out[16][16];
    for (int r = 0; r < 16; ++r)
        for (int cc = 0; cc < 16; ++cc)
            local_out[r][cc] = 0;

    for (int ty = 0; ty <= count_minus_1; ++ty) {
        int ox0 = g_offx[count_minus_1 + ty];
        int oy0 = g_offy[count_minus_1 + ty];
        int ox = bx * 8 + ox0;
        int oy = by * 8 + oy0;

        float d[64];
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 8; ++x)
                d[x + y * 8] = (float)src[(ox + x) + (oy + y) * src_pitch];

        int qp = (int)qp_table[bx + by * qp_pitch];
        float thresh = kf_qp_thresh(qp, thresh_a, thresh_b) *
                       ((1 << 2) + strength) - 1.0f;

        kf_dct8x8(d);
        kf_hardthresh(d, thresh);
        kf_idct8x8(d);

        int half = (1 << shift) >> 1;
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                int tmp = (int)(d[x + y * 8] + (float)half) >> shift;
                if (tmp < 0) tmp = 0; else if (tmp > maxv) tmp = maxv;
                local_out[oy0 + y][ox0 + x] += tmp;
            }
        }
    }

    int off_z = (bx & 1) + (by & 1) * 2;
    int offx = bx * 8;
    int offy = (bh * off_z + by) * 8;
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x)
            out[(offx + x) + (offy + y) * out_pitch] = (OUTP)local_out[y][x];
}
