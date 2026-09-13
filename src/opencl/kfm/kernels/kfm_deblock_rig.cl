/* ============================================================================
 * !!! PROVISIONAL — UNVERIFIED — DO NOT SHIP !!!
 *
 * kfm_deblock_rig.cl — KFM `KDeblock`-family kernels that are faithful
 * transcriptions of their CUDA device kernels but have NOT been verified
 * (no CPU mirror, no Python golden, no device run).  Everything in this file
 * is // RIG-VERIFY by definition.  It is deliberately kept OUT of
 * kfm_deblock.cl (which holds only // ALG-VERIFIED kernels) so that
 * "needs verification" is visible at a glance, and so the file can be handed
 * to another agent/rig for verification as a self-contained unit.
 *
 * Contents (all // RIG-VERIFY):
 *   kf_merge_deblock  kl_merge_deblock twin (Bayer accumulator merge)
 *   kf_max_vh/v/h     kl_max_vh/v/h twins (DC-mask box-max dilation)
 *   kf_scale_qp       kl_scale_qp twin (ShowQP rescaler)
 *   kf_sharpen_coeff  kl_sharpen_coeff twin (QP -> sharpen-strength LUT)
 *   kf_sharpen        kl_sharpen twin (SharpenFilter; texture->manual bilinear)
 *   kf_show_sharpen_coeff  kl_show_sharpen_coeff twin (coeff visualiser)
 * plus the g_ldither Bayer table, the g_sharpen_coeff LUT, the
 * kf_sharpen_bilinear helper, and a private copy
 * of the kf_norm_qscale helper (duplicated from kfm_deblock.cl so this file
 * is standalone — keep the two copies in sync).
 *
 * Upstream grounding: rigaya/AviSynthCUDAFilters, KFM/Deblock.cu at commit
 *   cceb8da0e623e6bf5eea2cf655b06d4428e1600b
 * (all eight kernels + both tables were re-checked semantically identical at
 * upstream HEAD 8e086bb; only brace style drifted).  Per-kernel line numbers
 * and the full verification recipe live in docs/RIG_HANDOFF_KDEBLOCK.md —
 * READ THAT FILE BEFORE TOUCHING THIS ONE.
 *
 * Graduation rule: when a kernel in this file is verified (CPU mirror +
 * independent Python golden, bit-exact, wired into `make test` per the repo
 * convention — or a bit-exact device run against the CUDA twin), MOVE it to
 * kfm_deblock.cl, mark it // ALG-VERIFIED, delete it here, and update the
 * docs.  When this file becomes empty, delete it.
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

/* Bayer-ordered dither matrix (Deblock.cu g_ldither[8][2], each an uchar4),
 * flattened per lane.  The scalar merge kernel indexes [y&7][(x>>2)&1][x&3]:
 * note the middle index runs over ushort4 columns, NOT scalar pixels. */
static const uchar g_ldither[8][2][4] = {
  { {  0,  48,  12,  60 }, {  3,  51,  15,  63 } },
  { { 32,  16,  44,  28 }, { 35,  19,  47,  31 } },
  { {  8,  56,   4,  52 }, { 11,  59,   7,  55 } },
  { { 40,  24,  36,  20 }, { 43,  27,  39,  23 } },
  { {  2,  50,  14,  62 }, {  1,  49,  13,  61 } },
  { { 34,  18,  46,  30 }, { 33,  17,  45,  29 } },
  { { 10,  58,   6,  54 }, {  9,  57,   5,  53 } },
  { { 42,  26,  38,  22 }, { 41,  25,  37,  21 } },
};

/* QP-block (qp>>3) -> sharpen-strength LUT (Deblock.cu d_sharpen_coeff /
 * g_sharpen_coeff, 30 entries; index >= 25 saturates to 255 in the kernel). */
static const uchar g_sharpen_coeff[30] = {
    0,   0,   0,   0,   0, // 0
    0,   0,   0,   0,  10, // 5(40)
   50,  90, 120, 150, 160, // 10(80)
  170, 180, 190, 200, 210, // 15(120)
  220, 230, 240, 245, 250, // 20(160)
  255, 255, 255, 255, 255, // 25(200)
};

/* Private copy of the kf_norm_qscale helper (see kfm_deblock.cl): normalize a
 * QP value by the codec's QP scale type (norm_qscale in Deblock.cu):
 *   type 0 (FF_QSCALE_TYPE_MPEG1): qscale << 2
 *   type 1 (FF_QSCALE_TYPE_MPEG2): qscale << 1
 *   type 2 (FF_QSCALE_TYPE_H264) : qscale
 *   type 3 (FF_QSCALE_TYPE_VP56) : 63 - qscale + 2   (= 65 - qscale) */
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
 * // RIG-VERIFY: faithful scalar transcription (lanes are independent, so the
 * // scalar port is lane-identical to the vector CUDA kernel); no CPU mirror /
 * // golden yet.  The tmp base/pitch-unit conventions above must be honoured
 * // by the host exactly, and the kf_deblock-output compatibility argued in
 * // the kfm_deblock.cl header must be proven, on a real device before use.
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
 * // RIG-VERIFY: faithful transcriptions, no mirror/golden yet.
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
 * // RIG-VERIFY: faithful transcription, no mirror/golden yet.
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
 * // RIG-VERIFY: faithful transcription, no mirror/golden yet.
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
 * // RIG-VERIFY: device-faithful transcription, but the texture fetch is
 * // manual float32 bilinear while CUDA uses HW fixed-point texture filtering
 * // — the two agree structurally yet may differ at float rounding boundaries,
 * // which is unknowable without a device run.  No mirror/golden yet; see the
 * // handoff doc for the device-comparison verification recipe.
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
 * // RIG-VERIFY: same texture-precision caveat as kf_sharpen; no
 * // mirror/golden yet (but the CPU-twin relation is exact, so a mirror+golden
 * // pair CAN pin this kernel's deterministic behaviour — see handoff doc).
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
