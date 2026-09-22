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
 *   kf_sharpen        kl_sharpen twin (SharpenFilter; texture->manual bilinear)
 *   kf_show_sharpen_coeff  kl_show_sharpen_coeff twin (coeff visualiser)
 * plus the kf_sharpen_bilinear helper.
 * (kf_merge_deblock, kf_max_vh/v/h, kf_scale_qp, kf_sharpen_coeff graduated
 * to kfm_deblock.cl as // ALG-VERIFIED; the g_ldither and g_sharpen_coeff
 * tables moved with them.)
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
