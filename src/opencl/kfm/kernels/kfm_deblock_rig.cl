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
 *   kf_sharpen        kl_sharpen twin (SharpenFilter; texture->manual bilinear)
 *   kf_show_sharpen_coeff  kl_show_sharpen_coeff twin (coeff visualiser)
 * plus the g_ldither Bayer table and the kf_sharpen_bilinear helper.
 * (kf_max_vh/v/h, kf_scale_qp, kf_sharpen_coeff graduated to kfm_deblock.cl
 * as // ALG-VERIFIED; the g_sharpen_coeff LUT moved with them.)
 *
 * Upstream grounding: rigaya/AviSynthCUDAFilters, KFM/Deblock.cu at commit
 *   cceb8da0e623e6bf5eea2cf655b06d4428e1600b
 * (all eight kernels + both tables were re-checked semantically identical at
 * upstream HEAD 8e086bb; only brace style drifted).  Graduated kernels were
 * re-verified against upstream HEAD 68aef6e.  Per-kernel line numbers
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
