/* ============================================================================
 * kfm_noiseclip.cl — OpenCL port of the KFM `KNoiseClip` filter kernel.
 *
 * Faithful port of rigaya/AviSynthCUDAFilters -> KFM/DecombeUCF.cu (KFM is
 * MIT).  KNoiseClip is a self-contained, **8-bit-only** AVS filter
 * (`KNoiseClip(clip, noise, nmin_y, range_y, nmin_uv, range_uv)`) that maps
 * each src pixel against a same-location `noise` pixel into one of five
 * band-marker values, producing a difference/activity map used by the
 * KDecombUCF family.  It is driven per plane by a single kernel
 * (`cpu_noise_clip` / `kl_noise_clip` — exact CPU twin), so this filter is
 * kernel-complete: only the AviSynth host glue (frame fetch + NewVideoFrame)
 * is needed to realise it on a rig.
 *
 * Per-pixel scalar algorithm (channels independent; the CUDA uchar4
 * vectorisation is over 4 scalar pixels, so a scalar port is bit-identical):
 *   s   = (src - noise + 256) >> 1        // int; s in [0,255], 128 == equal
 *   out = 128                             if s == 128
 *       = 0                               if s < 128 and (127-range) < s
 *                                              and s < (128-nmin)
 *       = 56                              if s < 128 otherwise
 *       = 255                             if s > 128 and (128+nmin) < s
 *                                              and s < (129+range)
 *       = 199                             if s > 128 otherwise
 *   (this is cpu dev_limitter(s, nmin, range) verbatim; nmin in [0,..],
 *    range in [0,..], defaults nmin=1, range=128.)
 *
 * Status:
 *   // ALG-VERIFIED (python/run_kfm_noiseclip.py, 300 cases) vs the CPU mirror
 *   // sim/kfm_noiseclip_ref.cpp (cpu_noise_clip twin), integer-exact.
 *
 * Notes:
 *  - 8-bit only upstream (ComponentSize==1; 16-bit throws).  Here the kernel
 *    is declared over uchar, so instantiate for 8-bit planes only.
 *  - Host requires plane width a multiple of 4 (CUDA uchar4) — the scalar port
 *    works for any width, but a faithful host config uses %4==0.
 *  - Y uses nmin_y/range_y; U and V each use nmin_uv/range_uv.
 * ==========================================================================*/

/* dev_limitter(s, nmin, range) — cpu dev_limitter twin. */
static int kf_limitter(int s, int nmin, int range)
{
    if (s == 128)
        return 128;
    if (s < 128)
        return ((127 - range) < s && s < (128 - nmin)) ? 0 : 56;
    return ((128 + nmin) < s && s < (129 + range)) ? 255 : 199;
}

/* Grid: 2D (width, height) of one plane.  dst/src/noise are uchar planes. */
kernel void kf_noise_clip(
    __global       uchar* __restrict dst,
    __global const uchar* __restrict src,
    __global const uchar* __restrict noise,
    int width, int height, int pitch,
    int nmin, int range)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int off = x + y * pitch;
        int s = ((int)src[off] - (int)noise[off] + 256) >> 1;
        dst[off] = (uchar)kf_limitter(s, nmin, range);
    }
}
