/* ============================================================================
 * kfm_combinganalyze.cl — OpenCL port of the CombingAnalyze.cu kernels (KFM,
 * MIT): the KSwitchFlag/KCombeMask/KRemoveCombe/KCleanSuper/KContainsCombe
 * stages (per-pixel uint8 ops + the uchar2 super-frame helpers).  The
 * KFMSuper block analyzer (kl_analyze_frame, warp-reduce) and the FMCount
 * reduction pair (kl_count_cmflags(_2planes)) live here too once ported; the
 * 8-tap calc_combe/calc_diff helpers they share are the static functions
 * below (transcribed verbatim, tap-level verified).
 *
 * Status: // ALG-VERIFIED (python/run_kfm_combinganalyze.py) vs
 *   sim/kfm_combinganalyze_ref.cpp — every kernel has an exact cpu_* twin
 *   upstream (bilinear twins take an extra unused PNeoEnv* arg; sum_box3x3's
 *   src is non-const upstream though never written — both transcribed const).
 *
 * Fidelity / assembly notes:
 *  - sum_box3x3 / bilinear_h / bilinear_v / remove_combe2 read a 1-px halo
 *    unguarded around their source (rows y-1/y+1, cols x0/x0+1).  Upstream
 *    feeds padded planes (bilinear: padv/padh(1)'d flag plane, fed with the
 *    documented row offsets; remove_combe2: VPAD frame), so the .cl contract
 *    is halo-padded sources and ALL outputs are verified with padded inputs
 *    (interior-origin pointers are a RIG-VERIFY host seam, as with
 *    kf_padv/kf_analyze_frame).  sum_box3x3's flag-plane halo content is a
 *    host-seam detail — the twins read identical addresses either way.
 *  - bilinear's (x-HALF)>>SHIFT keeps C arithmetic-shift semantics for the
 *    negative top rows/cols (universal on real devices; the mirror and the
 *    Python >> agree bit-for-bit).  SCALE/SHIFT are kernel args (production
 *    (4,2)/(8,3)); HALF = SCALE/2.
 *  - temporal_soften is float32: (f0+f1)+f2 has no FMA-able pattern, and the
 *    mult/divide round IEEE-correctly on any device — EXCEPT the (1.0f/3.0f)
 *    constant fold, which OpenCL does not guarantee correctly-rounded
 *    (RIG-VERIFY: must fold to 0x3EAAAAAB, as every mainstream compiler
 *    does; else pass k in from the host).  Output depends only on the byte
 *    sum t (all adds exact), so the t = 0..765 sweep is exhaustive.
 *  - binary_flag runs in-place upstream (dst == srcY) — race-free, as every
 *    lane touches only its own index.  sum_box3x3 ping-pongs through a tmp
 *    frame (never in-place).  contains_durty_block's *work = 1 is an
 *    idempotent race (all writers store 1).
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

/* absdiff over ints (the scalar twins use |a-b| on 8/16-bit lanes). */
static int kf_absdiff_i(int a, int b)
{
    int d = a - b;
    if (d < 0) d = -d;
    return d;
}

/* ---------------------------------------------------------------------------
 * 8-tap combing/diff helpers (calc_combe / calc_diff, CombingAnalyze.cu —
 * __host__ __device__ shared code used by kl_analyze_frame).  Transcribed
 * verbatim; verified at tap level (modes M/A).  Smooth fields give ~0,
 * field-alternating combs give large positive values, and the measure can go
 * negative (caller clamps after the shift).
 * -------------------------------------------------------------------------*/
static int kf_calc_combe8(int L0, int L1, int L2, int L3,
                          int L4, int L5, int L6, int L7)
{
    int diff8 = kf_absdiff_i(L0, L7);
    int diffT = kf_absdiff_i(L0, L1) + kf_absdiff_i(L1, L2)
        + kf_absdiff_i(L2, L3) + kf_absdiff_i(L3, L4) + kf_absdiff_i(L4, L5)
        + kf_absdiff_i(L5, L6) + kf_absdiff_i(L6, L7) - diff8;
    int diffE = kf_absdiff_i(L0, L2) + kf_absdiff_i(L2, L4)
        + kf_absdiff_i(L4, L6) + kf_absdiff_i(L6, L7) - diff8;
    int diffO = kf_absdiff_i(L0, L1) + kf_absdiff_i(L1, L3)
        + kf_absdiff_i(L3, L5) + kf_absdiff_i(L5, L7) - diff8;
    return diffT - diffE - diffO;
}

static int kf_calc_diff8(int L00, int L10, int L01, int L11,
                         int L02, int L12, int L03, int L13)
{
    return kf_absdiff_i(L00, L10) + kf_absdiff_i(L01, L11)
        + kf_absdiff_i(L02, L12) + kf_absdiff_i(L03, L13);
}

/* ---------------------------------------------------------------------------
 * kf_copy_first — first-lane extract (cpu_copy_first / kl_copy_first twin).
 * dst = src[...].x: the .x lane of each source vector (uchar2 super-frame
 * pixels in production; `lanes` covers the template-generic stride, 2 or 4).
 * Grid: 2D (width,height).  // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_copy_first(
    __global       uchar* __restrict dst, int dpitch,
    __global const uchar* __restrict src,
    int width, int height, int spitch, int lanes)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        dst[x + y * dpitch] = src[(x + y * spitch) * lanes];
    }
}

/* ---------------------------------------------------------------------------
 * kf_combe_to_flag — 2x2 combe downsample (cpu_combe_to_flag /
 * kl_combe_to_flag twin; uint8).  flag = (sum of the 2x2 combe quad + 2) >> 2
 * (round-half-up quarter mean).  The host launches it over the flag interior
 * (offset fpitch+1, size-1 — first row/col skipped); the kernel itself is a
 * plain downsample.  Grid: 2D (nBlkX,nBlkY).  // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_combe_to_flag(
    __global       uchar* __restrict flag,
    int nBlkX, int nBlkY, int fpitch,
    __global const uchar* __restrict combe, int cpitch)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < nBlkX && y < nBlkY) {
        int s = (int)combe[(2 * x + 0) + (2 * y + 0) * cpitch]
            + (int)combe[(2 * x + 1) + (2 * y + 0) * cpitch]
            + (int)combe[(2 * x + 0) + (2 * y + 1) * cpitch]
            + (int)combe[(2 * x + 1) + (2 * y + 1) * cpitch];
        flag[x + y * fpitch] = (uchar)((s + 2) >> 2);
    }
}

/* ---------------------------------------------------------------------------
 * kf_sum_box3x3 — 3x3 box smooth (cpu_sum_box3x3 / kl_sum_box3x3 twin; uint8).
 * dst = min((3x3 sum) >> 2, maxv) — the sum is quartered, NOT divided by 9
 * (upstream comment: always 1/4).  Reads the 3x3 neighbourhood unguarded, so
 * src carries a 1-px halo (see header); out-of-place (upstream ping-pongs
 * through a tmp frame — in-place would race).  Grid: 2D (width,height).
 * // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_sum_box3x3(
    __global       uchar* __restrict dst,
    __global const uchar* __restrict src,
    int width, int height, int pitch, int maxv)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int off = x + y * pitch;
        int sumv = (int)src[off - 1 - pitch] + (int)src[off - pitch]
            + (int)src[off + 1 - pitch]
            + (int)src[off - 1] + (int)src[off] + (int)src[off + 1]
            + (int)src[off - 1 + pitch] + (int)src[off + pitch]
            + (int)src[off + 1 + pitch];
        int v = sumv >> 2;
        dst[off] = (uchar)((v < maxv) ? v : maxv);
    }
}

/* ---------------------------------------------------------------------------
 * kf_binary_flag — Y/C threshold OR (cpu_binary_flag / kl_binary_flag twin;
 * uint8, non-template).  dst = (Y >= thY || C >= thC) ? 128 : 0 over a shared
 * pitch.  Runs in-place upstream (dst == srcY); lane-local, hence race-free
 * either way.  Grid: 2D (nBlkX,nBlkY).  // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_binary_flag(
    __global       uchar* __restrict dst,
    __global const uchar* __restrict srcY,
    __global const uchar* __restrict srcC,
    int nBlkX, int nBlkY, int pitch, int thY, int thC)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < nBlkX && y < nBlkY) {
        int off = x + y * pitch;
        int Y = (int)srcY[off];
        int C = (int)srcC[off];
        dst[off] = (uchar)(((Y >= thY) || (C >= thC)) ? 128 : 0);
    }
}

/* ---------------------------------------------------------------------------
 * kf_bilinear_h / kf_bilinear_v — separable bilinear upscale
 * (cpu_bilinear_h/v / kl_bilinear_h/v twins; uint8).  dst = (s0*c0 + s1*c1 +
 * HALF) >> SHIFT with HALF = SCALE/2, x0 = (x-HALF)>>SHIFT (arithmetic),
 * c0 = ((x0+1)<<SHIFT)-(x-HALF), c1 = SCALE-c0; v transposed.  Reads two
 * source cols (h) / rows (v) around the mapped coordinate, so src carries a
 * 1-col (h) / 1-row (v) halo (see header).  Output is always in [0,255] —
 * no clamp.  Grid: 2D (width,height).  // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_bilinear_h(
    __global       uchar* __restrict dst,
    int width, int height, int dpitch,
    __global const uchar* __restrict src, int spitch, int scale, int shift)
{
    int half = scale / 2;
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int x0 = (x - half) >> shift;
        int c0 = ((x0 + 1) << shift) - (x - half);
        int c1 = scale - c0;
        int s0 = (int)src[(x0 + 0) + y * spitch];
        int s1 = (int)src[(x0 + 1) + y * spitch];
        dst[x + y * dpitch] = (uchar)((s0 * c0 + s1 * c1 + half) >> shift);
    }
}

kernel void kf_bilinear_v(
    __global       uchar* __restrict dst,
    int width, int height, int dpitch,
    __global const uchar* __restrict src, int spitch, int scale, int shift)
{
    int half = scale / 2;
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int y0 = (y - half) >> shift;
        int c0 = ((y0 + 1) << shift) - (y - half);
        int c1 = scale - c0;
        int s0 = (int)src[x + (y0 + 0) * spitch];
        int s1 = (int)src[x + (y0 + 1) * spitch];
        dst[x + y * dpitch] = (uchar)((s0 * c0 + s1 * c1 + half) >> shift);
    }
}

/* ---------------------------------------------------------------------------
 * kf_temporal_soften — 3-frame temporal mean (cpu_temporal_soften /
 * kl_temporal_soften twin; uchar4 lanes, i.e. per-byte over the super-frame
 * words).  dst = (int)(((float)s0 + (float)s1 + (float)s2) * (1.0f/3.0f)),
 * truncated toward zero (non-negative: floor), wrapping mod 256 like
 * VHelper::cast_to (always in range in practice).  Float32, no FMA pattern;
 * the (1.0f/3.0f) fold is a RIG-VERIFY item (see header).  Grid: 2D
 * (width,height).  // ALG-VERIFIED (float32-exact golden)
 * -------------------------------------------------------------------------*/
kernel void kf_temporal_soften(
    __global       uchar* __restrict dst,
    __global const uchar* __restrict src0,
    __global const uchar* __restrict src1,
    __global const uchar* __restrict src2,
    int width, int height, int pitch)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int off = x + y * pitch;
        float t = (float)src0[off] + (float)src1[off] + (float)src2[off];
        int v = (int)(t * (1.0f / 3.0f));
        dst[off] = (uchar)v;
    }
}

/* ---------------------------------------------------------------------------
 * kf_remove_combe2 — combe-gated vertical binomial filter (cpu_remove_combe2
 * / kl_remove_combe2 twin; uint8/uint16 pixels).  score = combe[(x>>2) +
 * (y>>2)*c_pitch].x; pixels in 4x4 blocks scoring >= thcombe are replaced by
 * (src[y-1] + 2*src[y] + src[y+1] + 2) >> 2, others pass through.  Reads src
 * rows y-1/y+1 unguarded (VPAD-padded frame upstream — halo contract, see
 * header).  combe is the .x lane of the uchar2 super frame.  Grid: 2D
 * (width,height).  // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
static int kf_binomial_merge(int a, int b, int c)
{
    return (a + 2 * b + c + 2) >> 2;
}

kernel void kf_remove_combe2(
    __global       PX* __restrict dst,
    __global const PX* __restrict src,
    int width, int height, int pitch,
    __global const uchar* __restrict combe, int c_pitch, int thcombe)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int off = x + y * pitch;
        int score = (int)combe[(x >> 2) + (y >> 2) * c_pitch];
        if (score >= thcombe) {
            int v = kf_binomial_merge((int)src[off - pitch], (int)src[off],
                                      (int)src[off + pitch]);
            dst[off] = (PX)v;
        } else {
            dst[off] = src[off];
        }
    }
}

/* ---------------------------------------------------------------------------
 * kf_clean_super — super-frame combe cleaner (cpu_clean_super twin, per-plane;
 * CUDA spreads the U/V pair over blockIdx.z — values are plane-independent,
 * so the host launches this per plane, as the cpu twin does).  v = cur;
 * if (prev.y <= thresh && cur.y <= thresh) v.x = 0; dst = v.  The uchar2
 * planes are split into .x/.y uchar buffers (same element pitch).  Grid: 2D
 * (width,height).  // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_clean_super(
    __global       uchar* __restrict dst_x,
    __global       uchar* __restrict dst_y,
    __global const uchar* __restrict prev_x,
    __global const uchar* __restrict prev_y,
    __global const uchar* __restrict cur_x,
    __global const uchar* __restrict cur_y,
    int width, int height, int pitch, int thresh)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int off = x + y * pitch;
        int py = (int)prev_y[off];
        int cx = (int)cur_x[off];
        int cy = (int)cur_y[off];
        dst_y[off] = (uchar)cy;
        dst_x[off] = (uchar)(((py <= thresh) && (cy <= thresh)) ? 0 : cx);
        (void)prev_x; /* .x of prev is never read (as upstream) */
    }
}

/* ---------------------------------------------------------------------------
 * kf_init_contains_durty_block / kf_contains_durty_block — combe-present scan
 * (KContainsCombe; uint8).  init zeroes the single-int work flag (1 thread);
 * the scan stores 1 if any flag pixel is nonzero — an idempotent race, all
 * writers store the same value.  Value semantics: work = OR-reduction of the
 * plane.  Grid: 1 / 2D (width,height).  // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_init_contains_durty_block(__global int* __restrict work)
{
    work[0] = 0;
}

kernel void kf_contains_durty_block(
    __global const uchar* __restrict flagp,
    int width, int height, int pitch,
    __global int* __restrict work)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        if (flagp[x + y * pitch]) {
            work[0] = 1;
        }
    }
}
