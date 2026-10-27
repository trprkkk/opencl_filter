/* ============================================================================
 * kfm_deblock.cl — OpenCL port of the KFM `KDeblock` core kernel.
 *
 * Faithful port of rigaya/AviSynthCUDAFilters -> KFM/Deblock.cu (KFM is MIT).
 * KDeblock is a deblocking filter that (per 8x8 block and per QP-determined
 * threshold) applies an 8x8 DCT, a hard-threshold (requantisation) of the
 * AC coefficients, and an inverse DCT, accumulating `count = 1<<quality`
 * differently-offset 8x8 reconstructions of each block into a 16-bit plane.
 *
 * Contents — this file holds ONLY // ALG-VERIFIED kernels (cross-checked by
 * make test vs a CPU mirror + independent Python golden):
 *   kf_deblock        kl_deblock twin (core DCT stage)
 *   kf_make_qp_table  kl_make_qp_table twin (QP table)
 *   kf_deblock_show   kl_deblock_show twin (show==2)
 *   kf_max_vh/v/h     kl_max_vh/v/h twins (DC-mask dilation)
 *   kf_scale_qp       kl_scale_qp twin (ShowQP rescaler)
 *   kf_sharpen_coeff  kl_sharpen_coeff twin (QP -> sharpen LUT)
 *   kf_merge_deblock  kl_merge_deblock twin (Bayer accumulator merge)
 * plus the kf_norm_qscale helper, the g_deblock_offset tables (g_offx/g_offy),
 * the g_sharpen_coeff LUT and the g_ldither Bayer table, and the sharpen pair
 *   kf_sharpen        kl_sharpen twin (SharpenFilter; texture->manual bilinear)
 *   kf_show_sharpen_coeff  kl_show_sharpen_coeff twin (coeff visualiser)
 * plus the kf_sharpen_bilinear helper — graduated 2026-10-28 out of the
 * (now deleted) kfm_deblock_rig.cl after the CUDA texture-precision run
 * (docs/RIG_HANDOFF_KDEBLOCK.md §7; see their per-kernel comments for the
 * accepted texture-quantization divergence).
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
 * (8 px each side) into `src` (kl_padv/kl_padh live in KFMFilterBase and are
 * not ported here), builds the QP table (kf_make_qp_table) and finally merges
 * the accumulator (kf_merge_deblock) with the Bayer dither.  Because the CUDA
 * dither.  Because the CUDA block reads an 8x8 tile at an offset up to +7,
 * the src passed to kf_deblock must be the padded plane (a faithful host
 * config).  A block's offset index range [count_minus_1, 2*count_minus_1]
 * stays within g_deblock_offset[127] for quality <= 6 (count <= 64).
 * Accumulator layout note: CUDA kl_deblock stores packed ushort2 at
 * blockIdx.x*4 (out_pitch in ushort2 units) while kf_deblock writes scalar
 * ushort at bx*8 (out_pitch in ushort units); these are the same bytes when
 * the byte stride matches (ushort2-packed == row-major ushort, including the
 * atomicAdd-packed halves on little-endian).  kf_merge_deblock therefore
 * consumes kf_deblock's output directly with tmp_pitch_u4 = acc_pitch_ushort
 * >> 2, tmp_ipitch_rows = bh*8 rows per parity slice, and the tmp base
 * pre-offset by (+8 ushorts, +8 rows) exactly as the CUDA host pre-offsets
 * (+2 ushort4, +8 rows).  Device-confirmed 2026-10-28 (rig/dev_kfm_deblock.py):
 * the handoff §5 layout E2E — device accumulator feeding device
 * kf_merge_deblock vs the packed-quad golden — is bit-exact on the rig, both
 * PX.  The CUDA-twin texture comparison for the sharpen pair was closed
 * 2026-10-28 (rig/cuda_deblock_sharpen.cu; the pair now lives here,
 * // ALG-VERIFIED — see their per-kernel comments and
 * docs/RIG_HANDOFF_KDEBLOCK.md §7).
 * ==========================================================================*/

/* Device-verified (rig/dev_kfm_deblock.py, rig/diag_fma_deblock.py): default
 * driver builds contract mul+add into FMA and drift the DCT/IDCT accumulator
 * by +/-1..2 vs the per-operation float32 golden; with this pragma the device
 * matches the golden bit-exact on every case (both PX instantiations).  The
 * golden and the -ffp-contract=off mirror pin non-contracted float math. */
#pragma OPENCL FP_CONTRACT OFF

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
 * Index range used for a given quality is [count-1, 2*(count-1)].
 * OpenCL requires program-scope const arrays to live in __constant. */
__constant int g_offx[127] = {
  0,0,4, 0,2,6,4, 0,5,2,7,4,1,6,3, 0,4,1,5,3,7,2,6,0,4,1,5,3,7,2,6,
  0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,
  0,4,0,4,2,6,2,6,0,4,0,4,2,6,2,6,1,5,1,5,3,7,3,7,1,5,1,5,3,7,3,7,
  0,4,0,4,2,6,2,6,0,4,0,4,2,6,2,6,1,5,1,5,3,7,3,7,1,5,1,5,3,7,3,7,
};
__constant int g_offy[127] = {
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

        int halfc = (1 << shift) >> 1;   /* 'half' is an OpenCL C keyword */
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                int tmp = (int)(d[x + y * 8] + (float)halfc) >> shift;
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

/* ---------------------------------------------------------------------------
 * kf_norm_qscale — normalize a QP value by the codec's QP scale type
 * (norm_qscale in Deblock.cu, __host__ __device__):
 *   type 0 (FF_QSCALE_TYPE_MPEG1): qscale << 2
 *   type 1 (FF_QSCALE_TYPE_MPEG2): qscale << 1
 *   type 2 (FF_QSCALE_TYPE_H264) : qscale
 *   type 3 (FF_QSCALE_TYPE_VP56) : 63 - qscale + 2   (= 65 - qscale)
 * =========================================================================*/
static int kf_norm_qscale(int qscale, int type)
{
    switch (type) {
    case 0: return qscale << 2;
    case 1: return qscale << 1;
    case 2: return qscale;
    case 3: return (63 - qscale + 2);
    }
    return qscale;
}

/* ---------------------------------------------------------------------------
 * kf_make_qp_table — build the per-8px-block QP table that kf_deblock consumes
 * (kl_make_qp_table twin; cpu_make_qp_table is the authoritative CPU twin).
 *
 * The source QP plane(s) are macroblock (16px) grids of size
 * (in_width x in_height); each output block (x,y) of the (out_width x
 * out_height) table samples the source macroblock at
 * (min(x>>qp_shift_x, in_width-1), min(y>>qp_shift_y, in_height-1)).  When two
 * QP sources (B and non-B / two QP clips) are present they are element-max'd;
 * a per-macroblock DC luma level (dc_table) blends the two "block distortion"
 * (b) and "non-block distortion" (nonb) components:
 *   b     = norm_qscale(in_qp,  qp_scale)
 *   nonb  = norm_qscale(nonb_qp, qp_scale)
 *   ratio = min(1.0f, dc * dc_coeff)
 *   qp    = max(1, (int)(b*ratio + nonb*(1-ratio) + 0.5f))
 * where dc_coeff = b_ratio/255.0f (host-supplied).  When no source QP table is
 * present at all the table is constant `qp_scale` (= the host `force_qp`).
 *
 * The presence booleans carry what the CUDA expresses as NULL pointers
 * (dc_tableN / in_table1 optional).  Grid: 2D (out_width,out_height).
 * // ALG-VERIFIED via python/run_kfm_deblock_qp.py (see KFM_PORT_SPEC.md).
 * -------------------------------------------------------------------------*/
kernel void kf_make_qp_table(
    int in_width, int in_height,
    __global const uchar* __restrict in_table0,
    __global const uchar* __restrict nonb_table0,
    int has_table1,
    __global const uchar* __restrict in_table1,
    __global const uchar* __restrict nonb_table1,
    int in_pitch, int qp_scale, float dc_coeff,
    int has_dc0,
    __global const uchar* __restrict dc_table0,
    int has_dc1,
    __global const uchar* __restrict dc_table1,
    int dc_pitch,
    int qp_shift_x, int qp_shift_y,
    int out_width, int out_height,
    __global OUTP* __restrict out_table, int out_pitch)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= out_width || y >= out_height) return;

    int qp;
    if (in_table0) {
        int qp_x = min(x >> qp_shift_x, in_width - 1);
        int qp_y = min(y >> qp_shift_y, in_height - 1);
        int in_qp = (int)in_table0[qp_x + qp_y * in_pitch];
        int nonb_qp = (int)nonb_table0[qp_x + qp_y * in_pitch];
        int dc = has_dc0 ? (int)dc_table0[qp_x + qp_y * dc_pitch] : 255;
        if (has_table1) {
            in_qp = max(in_qp, (int)in_table1[qp_x + qp_y * in_pitch]);
            nonb_qp = max(nonb_qp, (int)nonb_table1[qp_x + qp_y * in_pitch]);
            dc = max(dc, has_dc1 ? (int)dc_table1[qp_x + qp_y * dc_pitch] : 255);
        }
        int b = kf_norm_qscale(in_qp, qp_scale);
        int nonb = kf_norm_qscale(nonb_qp, qp_scale);
        float b_ratio = min(1.0f, (float)dc * dc_coeff);
        qp = max(1, (int)(b * b_ratio + nonb * (1.0f - b_ratio) + 0.5f));
    } else {
        qp = qp_scale;
    }
    out_table[x + y * out_pitch] = (OUTP)qp;
}

/* ---------------------------------------------------------------------------
 * kf_deblock_show — KDeblock `show == 2` visualisation (kl_deblock_show twin;
 * cpu_deblock_show is the authoritative CPU twin).  Paints each 8px QP block
 * either `230` (deblocking enabled for it) or `16` (not), into the visible
 * plane.  Blocks tile the plane without overlap at an origin offset of -4:
 * block (bx,by) covers x in [bx*8-4, bx*8+3], y in [by*8-4, by*8+3], clipped to
 * the plane.  `enabled` = qp_apply_thresh(qp) >= (qp>>1).
 * Grid: 2D (width,height) over visible pixels (equivalent, deterministic).
 * // ALG-VERIFIED via python/run_kfm_deblock_qp.py.
 * -------------------------------------------------------------------------*/
kernel void kf_deblock_show(
    __global PX* __restrict dst, int dst_pitch,
    int width, int height,
    int bw, int bh,
    __global const OUTP* __restrict qp_table, int qp_pitch,
    float thresh_a, float thresh_b)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;

    int bx = (x + 4) >> 3;   /* pixel x belongs to block bx (origin bx*8-4) */
    int by = (y + 4) >> 3;
    OUTP qp = qp_table[bx + by * qp_pitch];
    int is_enabled = (kf_qp_thresh((int)qp, thresh_a, thresh_b) >= (int)(qp >> 1));
    dst[x + y * dst_pitch] = is_enabled ? (PX)230 : (PX)16;
}

/* QP-block (qp>>3) -> sharpen-strength LUT (Deblock.cu d_sharpen_coeff /
 * g_sharpen_coeff, 30 entries; index >= 25 saturates to 255 in the kernel;
 * byte-verified by mechanical diff vs upstream d_/g_ copies).
 * OpenCL requires program-scope const arrays to live in __constant. */
__constant uchar g_sharpen_coeff[30] = {
    0,   0,   0,   0,   0, // 0
    0,   0,   0,   0,  10, // 5(40)
   50,  90, 120, 150, 160, // 10(80)
  170, 180, 190, 200, 210, // 15(120)
  220, 230, 240, 245, 250, // 20(160)
  255, 255, 255, 255, 255, // 25(200)
};

/* ---------------------------------------------------------------------------
 * kf_max_vh / kf_max_v / kf_max_h — DC-mask box-max dilation used by the
 * QPForDeblock helper (kl_max_vh / kl_max_v / kl_max_h twins; cpu_max_v /
 * cpu_max_h are the exact CPU twins, kl_max_vh is device-only upstream).  The
 * CUDA host instantiates RADIUS=5 in all call sites; radius is a kernel arg
 * here.  kf_max_v is the scalar per-pixel form of the uchar4-vector CUDA
 * kernel (lanes independent ⇒ identical); grid is pixels, not uchar4 lanes.
 * Reads span [-radius, +radius] around every pixel, so src/dst must be the
 * interior of a plane padded by >= radius (the CUDA host passes pad+8+8*pitch
 * with an 8 px margin) — identical edge contract as upstream.  Grid: 2D
 * (width, height).
 * // ALG-VERIFIED via python/run_kfm_deblock_aux.py (150 cases each): 8px-margin
 * padded harness, radius 1..8 (5 = production), spike crafts; max_vh is
 * additionally cross-checked via the separable max_h o max_v identity.
 * -------------------------------------------------------------------------*/
kernel void kf_max_vh(
    __global uchar* __restrict dst, __global const uchar* __restrict src,
    int width, int height, int pitch, int radius)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;

    uchar sum = 0;
    for (int j = -radius; j <= radius; ++j) {
        for (int i = -radius; i <= radius; ++i) {
            sum = max(sum, src[(x + i) + (y + j) * pitch]);
        }
    }
    dst[x + y * pitch] = sum;
}

kernel void kf_max_v(
    __global uchar* __restrict dst, __global const uchar* __restrict src,
    int width, int height, int pitch, int radius)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;

    uchar sum = 0;
    for (int i = -radius; i <= radius; ++i) {
        sum = max(sum, src[x + (y + i) * pitch]);
    }
    dst[x + y * pitch] = sum;
}

kernel void kf_max_h(
    __global uchar* __restrict dst, __global const uchar* __restrict src,
    int width, int height, int pitch, int radius)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;

    uchar sum = 0;
    for (int i = -radius; i <= radius; ++i) {
        sum = max(sum, src[(x + i) + y * pitch]);
    }
    dst[x + y * pitch] = sum;
}

/* ---------------------------------------------------------------------------
 * kf_scale_qp — rescale a QP plane by codec QP-scale-type (kl_scale_qp /
 * cpu_scale_qp twin; the ShowQP debug filter).  Per pixel:
 *   dst = (uchar)norm_qscale(src, scale_type)
 * The int->uchar conversion wraps mod 256 exactly like the CUDA assignment.
 * Grid: 2D (width, height).
 * // ALG-VERIFIED via python/run_kfm_deblock_aux.py (200 cases): full-range
 * inputs, mod-256 wrap pins (type 0), and out-of-range scale types.
 * -------------------------------------------------------------------------*/
kernel void kf_scale_qp(
    int width, int height,
    __global uchar* __restrict dst, int dst_pitch,
    __global const uchar* __restrict src, int src_pitch, int scale_type)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;

    dst[x + y * dst_pitch] = (uchar)kf_norm_qscale((int)src[x + y * src_pitch], scale_type);
}

/* ---------------------------------------------------------------------------
 * kf_sharpen_coeff — QP-block -> sharpen-strength LUT (kl_sharpen_coeff /
 * cpu_sharpen_coeff twin; feeds the SharpenFilter, not KDeblock itself):
 *   q = qp[x + y*qp_pitch] >> 3;  dst = (q >= 25) ? 255 : g_sharpen_coeff[q]
 * Grid: 2D (width, height) over the QP-block grid.
 * // ALG-VERIFIED via python/run_kfm_deblock_aux.py (200 cases): qp swept
 * 0..65535 with q=24/25 boundary emphasis; LUT bytes diffed vs upstream.
 * -------------------------------------------------------------------------*/
kernel void kf_sharpen_coeff(
    __global uchar* __restrict dst, int width, int height, int pitch,
    __global const ushort* __restrict qp, int qp_pitch)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;

    int q = ((int)qp[x + y * qp_pitch]) >> 3;
    dst[x + y * pitch] = (q >= 25) ? (uchar)255 : g_sharpen_coeff[q];
}

/* Bayer-ordered dither matrix (Deblock.cu g_ldither[8][2], each an uchar4),
 * flattened per lane.  The scalar merge kernel indexes [y&7][(x>>2)&1][x&3]:
 * note the middle index runs over ushort4 columns, NOT scalar pixels;
 * byte-verified by mechanical diff vs upstream).
 * OpenCL requires program-scope const arrays to live in __constant. */
__constant uchar g_ldither[8][2][4] = {
  { {  0,  48,  12,  60 }, {  3,  51,  15,  63 } },
  { { 32,  16,  44,  28 }, { 35,  19,  47,  31 } },
  { {  8,  56,   4,  52 }, { 11,  59,   7,  55 } },
  { { 40,  24,  36,  20 }, { 43,  27,  39,  23 } },
  { {  2,  50,  14,  62 }, {  1,  49,  13,  61 } },
  { { 34,  18,  46,  30 }, { 33,  17,  45,  29 } },
  { { 10,  58,   6,  54 }, {  9,  57,   5,  53 } },
  { { 42,  26,  38,  22 }, { 41,  25,  37,  21 } },
};

/* ---------------------------------------------------------------------------
 * kf_merge_deblock — merge the 16-bit block-parity accumulator into the final
 * plane (kl_merge_deblock / cpu_merge_deblock twin).  Per visible pixel:
 *   sum = acc[slice0] + acc[slice1] + acc[slice2] + acc[slice3]   (int)
 *   v   = (float)sum * (1/(1<<shift)) + (float)dither * (1/64)
 *   out = (PX)fmin(v, maxv)                              (C-truncation cast)
 * where dither = g_ldither[y&7][(x>>2)&1][x&3] and shift = mergeShift =
 * quality+6-deblockShift, maxv = (1<<bits)-1.  The 4 parity slices are stacked
 * vertically with tmp_ipitch_rows rows each (bh*8); tmp_pitch_u4 is the
 * accumulator pitch in ushort4 units (= acc_pitch_ushort >> 2); the tmp base
 * is pre-offset by (+8 ushorts, +8 rows) exactly as the CUDA host passes
 * tmpOut+2+8*pitch.  Grid: 2D (vis_width, vis_height) pixels; vis_width must
 * be a multiple of 4 (CUDA covers width>>2 uchar4/ushort4 lanes, i.e. the
 * width&~3 left pixels; pass vis_width = width & ~3).
 * // ALG-VERIFIED via python/run_kfm_deblock_aux.py (250 merge + 15 end-to-end
 * // + 5 layout cases): quality 1..6 x bits 8/10/12/16, spike crafts pinning the
 * // k/X/L summation, dither-boundary flips pinning [y&7][X&1][L], and the
 * // handoff section-5 packing identity (packed ushort2 == scalar ushort).
 * -------------------------------------------------------------------------*/
kernel void kf_merge_deblock(
    __global const ushort* __restrict tmp, int tmp_pitch_u4, int tmp_ipitch_rows,
    __global PX* __restrict out, int out_pitch,
    int vis_width, int vis_height, int shift, float maxv)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= vis_width || y >= vis_height) return;

    int X = x >> 2;   /* ushort4 column (dither middle index runs over this) */
    int L = x & 3;    /* lane within the ushort4 */
    int sum = 0;
    for (int k = 0; k < 4; ++k) {
        int row = tmp_ipitch_rows * k + y;
        sum += (int)tmp[((X + row * tmp_pitch_u4) << 2) + L];
    }
    float v = (float)sum * (1.0f / (float)(1 << shift)) +
              (float)g_ldither[y & 7][X & 1][L] * (1.0f / 64.0f);
    v = fmin(v, maxv);
    out[x + y * out_pitch] = (PX)v;
}

/* ---------------------------------------------------------------------------
 * kf_sharpen_bilinear — manual bilinear sample of the uchar coeff plane at
 * pixel (x,y), i.e. coeff-texel (x/8, y/8).  This replaces the CUDA texture
 * fetch `tex2D<float>(coeff, x/8+0.5, y/8+0.5)` (Clamp/Linear/
 * NormalizedFloat over the qp-sized coeff plane — the clamp never engages
 * because width is a multiple of 8 and the coeff plane is qp-sized with a
 * +1-block margin, so taps ix+1/iy+1 are always in bounds, exactly as the CPU
 * twins rely on).  Expression and op order follow the CPU twins verbatim;
 * returns the UNNORMALIZED (0..255-scale) value — callers normalize.
 * -------------------------------------------------------------------------*/
static float kf_sharpen_bilinear(
    __global const uchar* __restrict coeff, int coeff_pitch,
    int x, int y)
{
    float fx = (float)x * (1.0f / 8.0f);
    float fy = (float)y * (1.0f / 8.0f);
    int ix = (int)fx;
    int iy = (int)fy;
    float c00 = (float)coeff[ix + iy * coeff_pitch];
    float c01 = (float)coeff[(ix + 1) + iy * coeff_pitch];
    float c10 = (float)coeff[ix + (iy + 1) * coeff_pitch];
    float c11 = (float)coeff[(ix + 1) + (iy + 1) * coeff_pitch];
    float fracx = fx - (float)ix;
    float fracy = fy - (float)iy;
    return (c00 * (1.0f - fracx) + c01 * fracx) * (1.0f - fracy) +
           (c10 * (1.0f - fracx) + c11 * fracx) * fracy;
}

/* ---------------------------------------------------------------------------
 * kf_sharpen — SharpenFilter core (kl_sharpen twin; cpu_sharpen is the CPU
 * twin with guarded borders).  Per pixel:
 *   s = src; (l,h) = min/max of s over the 3x3 window with edge-clamped reads
 *   c = bilinear(coeff, x/8, y/8) / 255        (manual; replaces tex2D)
 *   u = unsharp[x + y*pitch]                   (shares the dst pitch, as CUDA)
 *   dst = (int)clamp(s + (s-u)*c + 0.5f, l, h) (C-truncation)
 * The 3x3 window transcribes the DEVICE form verbatim, INCLUDING the upstream
 * quirk `min(x + 1, height - 1)` (x clamped by height, not width — differs
 * from the CPU twin wherever x+1 > height-1, i.e. when width > height).
 * The CPU twin instead guards borders with ifs and skips the window when
 * c == 0 (equivalent result: clamp(s+0.5,l,h) truncated == s since l<=s<=h).
 * Host contract (as the SharpenFilter host guarantees): width is a multiple
 * of 8; coeff is the qp-sized uchar plane (block margin included), so the
 * bilinear taps are always in bounds; unsharp shares dst geometry/pitch.
 * Grid: 2D (width, height).
 * // ALG-VERIFIED: deterministic behaviour pinned by sim/kfm_deblock_aux_ref.cpp
 * // + python/run_kfm_deblock_aux.py (P mode: quirk configs, c==0 identity,
 * // float32-exact golden; make test).  OpenCL device run PASS vs the
 * // manual-bilinear golden (2026-10-28, rig/dev_kfm_deblock.py, both PX,
 * // bit-exact; the upstream quirk min(x+1,height-1) proven to bite on
 * // device).  CUDA twin comparison closed 2026-10-28
 * // (rig/cuda_deblock_sharpen.cu, rig/CUDA_REPORT_kdeblock_texture.md):
 * // CUDA == goldens except 7/28,840 px1 + 428/28,040 px2, all +/-1 on the
 * // sharpen output boundary — ACCEPTED DIVERGENCE (docs/RIG_HANDOFF_KDEBLOCK.md
 * // §7): the CUDA texture path quantizes the HW-bilinear result to the 16-bit
 * // unorm grid `round_half_up(v*257)/65535` (closed-form model bit-exact vs
 * // CUDA over 11,520 probe + 108,024 output px) while this port uses the
 * // load-based manual float32 bilinear; plus 1/28,040 px from upstream's own
 * // default-FMA contraction (this file pins FP_CONTRACT OFF).  Characterised
 * // and accepted per §7; no tolerance weakening of the goldens.
 * -------------------------------------------------------------------------*/
kernel void kf_sharpen(
    __global PX* __restrict dst, int width, int height, int pitch,
    __global const PX* __restrict src, int src_pitch,
    __global const uchar* __restrict coeff, int coeff_pitch,
    __global const PX* __restrict unsharp)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;

    int s = (int)src[x + y * src_pitch];
    int l = s;
    int h = s;
    int v;

    v = (int)src[max(x - 1, 0) + max(y - 1, 0) * src_pitch];
    l = min(l, v); h = max(h, v);
    v = (int)src[(x + 0) + max(y - 1, 0) * src_pitch];
    l = min(l, v); h = max(h, v);
    v = (int)src[min(x + 1, height - 1) + max(y - 1, 0) * src_pitch];
    l = min(l, v); h = max(h, v);
    v = (int)src[max(x - 1, 0) + (y + 0) * src_pitch];
    l = min(l, v); h = max(h, v);
    v = (int)src[min(x + 1, height - 1) + (y + 0) * src_pitch];
    l = min(l, v); h = max(h, v);
    v = (int)src[max(x - 1, 0) + min(y + 1, height - 1) * src_pitch];
    l = min(l, v); h = max(h, v);
    v = (int)src[(x + 0) + min(y + 1, height - 1) * src_pitch];
    l = min(l, v); h = max(h, v);
    v = (int)src[min(x + 1, height - 1) + min(y + 1, height - 1) * src_pitch];
    l = min(l, v); h = max(h, v);

    float c = kf_sharpen_bilinear(coeff, coeff_pitch, x, y) * (1.0f / 255.0f);
    int u = (int)unsharp[x + y * pitch];
    float r = (float)s + (float)(s - u) * c + 0.5f;
    if (r < (float)l) r = (float)l; else if (r > (float)h) r = (float)h;
    dst[x + y * pitch] = (PX)(int)r;
}

/* ---------------------------------------------------------------------------
 * kf_show_sharpen_coeff — SharpenFilter `show` visualiser
 * (kl_show_sharpen_coeff twin; cpu_show_sharpen_coeff is the exact CPU twin
 * for the manual-bilinear form).  Per pixel:
 *   dst = (PX)(int)bilinear(coeff, x/8, y/8)
 * i.e. the unnormalized manual bilinear, C-truncated — exactly the CPU twin's
 * expression (the device computes (int)(normalized_tex*255), equal up to
 * texture-filter precision).  Same coeff host contract as kf_sharpen.
 * Grid: 2D (width, height).
 * // ALG-VERIFIED: deterministic behaviour pinned by sim/kfm_deblock_aux_ref.cpp
 * // (W mode) + python/run_kfm_deblock_aux.py (ramps, fractional taps,
 * // float32-exact golden; make test).  OpenCL device run PASS vs that golden
 * // (2026-10-28, rig/dev_kfm_deblock.py, both PX; matches the trunc texture
 * // emulation exactly).  CUDA twin comparison closed 2026-10-28
 * // (rig/cuda_deblock_sharpen.cu, rig/CUDA_REPORT_kdeblock_texture.md):
 * // CUDA == goldens bit-exact, 0/51,144 px differ (both PX) — no gap.
 * // For the texture-path quantization note shared with kf_sharpen (16-bit
 * // unorm grid `round_half_up(v*257)/65535` vs this port's load-based fetch),
 * // see kf_sharpen's comment — here it is provably output-invisible.
 * -------------------------------------------------------------------------*/
kernel void kf_show_sharpen_coeff(
    __global PX* __restrict dst, int width, int height, int pitch,
    __global const uchar* __restrict coeff, int coeff_pitch)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;

    float c = kf_sharpen_bilinear(coeff, coeff_pitch, x, y);
    dst[x + y * pitch] = (PX)(int)c;
}
