/* ============================================================================
 * kfm_deband.cl — OpenCL port of the KFM `KDeband` filter (debanding).
 *
 * Faithful port of rigaya/AviSynthCUDAFilters -> KFM/KDeband.cu
 * (KFM is MIT licensed).  The authoritative CPU reference is the exact twin
 * `cpu_reduce_banding` in that same file, so this is a clean, verifiable,
 * per-pixel port: every output element depends only on its own pixel and a few
 * in-range neighbours chosen by a deterministic pseudo-random offset, so a
 * scalar translation is bit-identical.
 *
 * Kernel math (identical to cpu_reduce_banding / kl_reduce_banding):
 *   For pixel (x,y):  offset = y*pitch + x.
 *     rand_step = width*height.  (Upstream builds ONE rand buffer of length
 *       width*height*2 for the luma plane and reuses it for each plane with that
 *       plane's own width/height stride; the offsets below always stay in bounds
 *       when per-plane pitch == per-plane width, which is the faithful config.)
 *     range_limited = min(min(range, y), min(height-y-1, min(x, width-x-1)))
 *       -- clamps the random sampling distance so neighbours are always inside
 *          the image (no border padding is needed).
 *     refA = random_range(rand[offset + rand_step*0], range_limited)
 *     refB = random_range(rand[offset + rand_step*1], range_limited)
 *     random_range(r, range) = (((range<<1)+1) * r) >> 8 - range,   (range<=127)
 *     src_val = src[offset]
 *     sample_mode 0 : avg = src[offset + (refA*pitch+refB)]
 *                     diff = |src_val - avg|
 *     sample_mode 1 : p = src[offset + ref], m = src[offset - ref], ref=refA*pitch+refB
 *                     avg = (p+m)>>1
 *                     diff = blur_first ? |src_val-avg|
 *                                       : max(|src_val-p|,|src_val-m|)
 *     sample_mode 2 : four refs from ref_0=refA*pitch+refB, ref_1=refA-refB*pitch
 *                     avg = (p0+m0+p1+m1)>>2
 *                     diff = blur_first ? |src_val-avg|
 *                                       : max of the four |src_val-*|
 *     dst[offset] = (diff <= thresh) ? avg : src_val
 *
 * The per-plane PX instantiation keeps 8/16-bit ranges correct.  thresh is
 * passed already host-scaled (MV.cpp/KFM scaleParam: thresh*(1<<(bits-8))+0.5).
 *
 * The XorShift pseudo-random byte stream that fills `rand` is generated once on
 * the host (seed 0) and is device-independent; see the kernel comments below /
 * python/run_kfm_deband.py for the exact stream.
 *
 * Status legend (matches the KTGMC files):
 *   // ALG-VERIFIED : integer algorithm cross-checked (make test) against an
 *                     independent CPU mirror + Python golden.
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

// random_range(uint8_t random, char range) from KDeband.cu.  range_limited is
// clamped to <= range which the filter limits to [0,127], so it fits in a char
// and the char/int arithmetic below reproduces the CPU exactly.
//   returns an int in [-range, range].
static int kf_random_range(int random, int range)
{
    return ((((range << 1) + 1) * random) >> 8) - range;
}

/* ---------------------------------------------------------------------------
 * kf_deband_reduce_banding — the KDeband core (cpu_reduce_banding twin).
 * grid: 2D (width, height).  dst/src planes are height rows of `pitch` PX
 * samples each; rand is the host-built pseudo-random byte stream
 * (length >= 2*width*height for pitch==width; see header note).
 * sample_mode/blur_first are passed at runtime (the CUDA original specialised
 * them as template params — that is only a code-gen choice; arithmetic is the
 * same for the 3x2 table entries).
 * // ALG-VERIFIED (python/run_kfm_deband.py: cpu_reduce_banding-equivalent
 * // mirror vs an independent Python golden, 8/16-bit, modes 0-2, blur on/off)
 * -------------------------------------------------------------------------*/
kernel void kf_deband_reduce_banding(
    __global       PX* __restrict dst,
    __global const PX* __restrict src,
    __global const uchar* __restrict rand,
    int width, int height, int pitch,
    int range, int thresh,
    int sample_mode, int blur_first)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height)
        return;

    int rand_step = width * height;
    int offset = y * pitch + x;

    int rl = range;
    if (y < rl) rl = y;
    int t = height - y - 1; if (t < rl) rl = t;
    if (x < rl) rl = x;
    t = width - x - 1; if (t < rl) rl = t;
    int range_limited = rl;

    int refA = kf_random_range((int)rand[offset + rand_step * 0], range_limited);
    int refB = kf_random_range((int)rand[offset + rand_step * 1], range_limited);

    int src_val = (int)src[offset];
    int avg, diff;

    if (sample_mode == 0) {
        int ref = refA * pitch + refB;
        avg = (int)src[offset + ref];
        diff = src_val - avg; if (diff < 0) diff = -diff;
    } else if (sample_mode == 1) {
        int ref = refA * pitch + refB;
        int ref_p = (int)src[offset + ref];
        int ref_m = (int)src[offset - ref];
        avg = (ref_p + ref_m) >> 1;
        int dp = src_val - ref_p; if (dp < 0) dp = -dp;
        int dm = src_val - ref_m; if (dm < 0) dm = -dm;
        diff = blur_first ? (src_val - avg < 0 ? avg - src_val : src_val - avg)
                          : (dp > dm ? dp : dm);
    } else {
        int ref_0 = refA * pitch + refB;
        int ref_1 = refA - refB * pitch;
        int ref_0p = (int)src[offset + ref_0];
        int ref_0m = (int)src[offset - ref_0];
        int ref_1p = (int)src[offset + ref_1];
        int ref_1m = (int)src[offset - ref_1];
        avg = (ref_0p + ref_0m + ref_1p + ref_1m) >> 2;
        int d0p = src_val - ref_0p; if (d0p < 0) d0p = -d0p;
        int d0m = src_val - ref_0m; if (d0m < 0) d0m = -d0m;
        int d1p = src_val - ref_1p; if (d1p < 0) d1p = -d1p;
        int d1m = src_val - ref_1m; if (d1m < 0) d1m = -d1m;
        int m0 = d0p > d0m ? d0p : d0m;
        int m1 = d1p > d1m ? d1p : d1m;
        diff = blur_first ? (src_val - avg < 0 ? avg - src_val : src_val - avg)
                          : (m0 > m1 ? m0 : m1);
    }

    dst[offset] = (diff <= thresh) ? (PX)avg : (PX)src_val;
}
