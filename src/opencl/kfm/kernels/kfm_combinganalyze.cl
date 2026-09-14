/* ============================================================================
 * kfm_combinganalyze.cl — OpenCL port of the CombingAnalyze.cu kernels (KFM,
 * MIT): the KSwitchFlag/KCombeMask/KRemoveCombe/KCleanSuper/KContainsCombe
 * stages (per-pixel uint8 ops + the uchar2 super-frame helpers), the KFMSuper
 * block analyzer (kf_super_analyze — serial per-cell transcription of the
 * warp-reduce kl_analyze_frame, value-identical for integer sums), and the
 * FMCount reduction pair (kf_init_fmcount / kf_count_cmflags /
 * kf_count_cmflags_2planes — work-group reduce + global atomics).  All 15
 * CombingAnalyze.cu device kernels are ported; the 8-tap calc_combe/calc_diff
 * helpers are the static functions below (transcribed verbatim, tap-level
 * verified).
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
 *  - super_analyze leaves flag col 0 and rows 0-1 untouched (border cells
 *    write nothing — verified sentinel-pinned); their content is a host-seam
 *    detail, as upstream.  The serial per-cell loop replaces the warp
 *    reduction (integer sums are order-exact).
 *  - count_cmflags(_2planes) requires 32x16 work-groups (fixed 512-tree in
 *    __local memory, as upstream's FM_COUNT block); atomics accumulate into
 *    FMCount[2] at slot (i ^ !parity) — order-exact for ints.  The fused
 *    2planes form provably equals two single-plane passes (mode Q).
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

/* Clamp to uchar range (std::clamp(v,0,255) upstream). */
static int kf_clamp_u8(int v)
{
    if (v < 0) v = 0; else if (v > 255) v = 255;
    return v;
}

/* ---------------------------------------------------------------------------
 * kf_super_analyze — KFMSuper block analyzer (kl_analyze_frame /
 * cpu_analyze_frame twin, CombingAnalyze.cu; uchar2 flags, parity-templated
 * upstream — parity is an int arg here; uint8/uint16 pixels).  Per 4x4-stride
 * block cell (bx,by), the 8 taps x = bx*4+tx accumulate 4 sums: top/bottom
 * combe (TFF: combe over f0's 8 rows for top, f1/f0-interleaved for bottom;
 * BFF mirrored) plus top/bottom field diffs; cell (bx,by) writes flag rows
 * 2*(by+1)+{0,1} at col bx+1 as clamp(sum>>shift).  Cells with bx==nBlkX-1 or
 * by==nBlkY-1 write nothing (flag col 0 and rows 0-1 stay untouched, as
 * upstream — their content is a host-seam detail), and shift = BPC-8+4
 * upstream (4/12; any int works).  CUDA spreads the 8 tx lanes over a warp
 * and reduces; integer sums are order-exact, so the serial per-cell loop
 * here is value-identical (it is the cpu twin).  The uchar2 flag plane is
 * split into .x/.y uchar buffers (same element pitch).  Grid: 2D over
 * (nBlkX-1, nBlkY-1) — the guard also tolerates larger launches.
 * // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_super_analyze(
    __global       uchar* __restrict flag_x,
    __global       uchar* __restrict flag_y, int fpitch,
    __global const PX* __restrict f0,
    __global const PX* __restrict f1, int pitch,
    int nBlkX, int nBlkY, int shift, int parity)
{
    int bx = (int)get_global_id(0);
    int by = (int)get_global_id(1);
    if (bx < nBlkX - 1 && by < nBlkY - 1) {
        int x0 = bx * 4; /* DC_OVERLAP */
        int y0 = by * 4;
        int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        for (int tx = 0; tx < 8; ++tx) { /* DC_BLOCK_SIZE, serial warp */
            int x = x0 + tx;
            int f0r0 = (int)f0[x + (y0 + 0) * pitch];
            int f0r1 = (int)f0[x + (y0 + 1) * pitch];
            int f0r2 = (int)f0[x + (y0 + 2) * pitch];
            int f0r3 = (int)f0[x + (y0 + 3) * pitch];
            int f0r4 = (int)f0[x + (y0 + 4) * pitch];
            int f0r5 = (int)f0[x + (y0 + 5) * pitch];
            int f0r6 = (int)f0[x + (y0 + 6) * pitch];
            int f0r7 = (int)f0[x + (y0 + 7) * pitch];
            int f1r0 = (int)f1[x + (y0 + 0) * pitch];
            int f1r1 = (int)f1[x + (y0 + 1) * pitch];
            int f1r2 = (int)f1[x + (y0 + 2) * pitch];
            int f1r3 = (int)f1[x + (y0 + 3) * pitch];
            int f1r4 = (int)f1[x + (y0 + 4) * pitch];
            int f1r5 = (int)f1[x + (y0 + 5) * pitch];
            int f1r6 = (int)f1[x + (y0 + 6) * pitch];
            int f1r7 = (int)f1[x + (y0 + 7) * pitch];
            int t0, t2;
            if (parity) { /* TFF */
                t0 = kf_calc_combe8(f0r0, f0r1, f0r2, f0r3,
                                    f0r4, f0r5, f0r6, f0r7);
                t2 = kf_calc_combe8(f1r0, f0r1, f1r2, f0r3,
                                    f1r4, f0r5, f1r6, f0r7);
            } else { /* BFF */
                t2 = kf_calc_combe8(f0r0, f0r1, f0r2, f0r3,
                                    f0r4, f0r5, f0r6, f0r7);
                t0 = kf_calc_combe8(f0r0, f1r1, f0r2, f1r3,
                                    f0r4, f1r5, f0r6, f1r7);
            }
            int t1 = kf_calc_diff8(f0r0, f1r0, f0r2, f1r2,
                                   f0r4, f1r4, f0r6, f1r6);
            int t3 = kf_calc_diff8(f0r1, f1r1, f0r3, f1r3,
                                   f0r5, f1r5, f0r7, f1r7);
            sum0 += t0; sum1 += t1; sum2 += t2; sum3 += t3;
        }
        int c = bx + 1;
        int r0 = 2 * (by + 1) + 0;
        int r1 = 2 * (by + 1) + 1;
        flag_x[c + r0 * fpitch] = (uchar)kf_clamp_u8(sum0 >> shift);
        flag_y[c + r0 * fpitch] = (uchar)kf_clamp_u8(sum1 >> shift);
        flag_x[c + r1 * fpitch] = (uchar)kf_clamp_u8(sum2 >> shift);
        flag_y[c + r1 * fpitch] = (uchar)kf_clamp_u8(sum3 >> shift);
    }
}

/* FMCount block geometry (FM_COUNT_TH_W/H upstream — fixed 32x16 groups). */
#define KCA_FM_TH_W 32
#define KCA_FM_TH_H 16
#define KCA_FM_THREADS 512

/* ---------------------------------------------------------------------------
 * kf_init_fmcount — zero the two FMCount slots (kl_init_fmcount twin;
 * FMCount = {move, shima, lshima} ints, split into 3 arrays of 2).
 * Launched with 2 work-items, as <<<1,2>>> upstream.  // ALG-VERIFIED
 * -------------------------------------------------------------------------*/
kernel void kf_init_fmcount(
    __global int* __restrict dst_move,
    __global int* __restrict dst_shima,
    __global int* __restrict dst_lshima)
{
    int tx = (int)get_global_id(0);
    if (tx < 2) {
        dst_move[tx] = 0;
        dst_shima[tx] = 0;
        dst_lshima[tx] = 0;
    }
}

/* ---------------------------------------------------------------------------
 * kf_count_cmflags — FMCount threshold census (kl_count_cmflags /
 * cpu_count_cmflags twin).  Per pixel, for i in {0,1}: (combe_i.y >= thM,
 * combe_i.x >= thS, combe_i.x >= thLS) accumulate into slot (i ^ !parity) —
 * combe0 counts land in slot !parity, combe1 in slot parity.  Structure is
 * faithful: per-item 0/1 counts, work-group tree reduction over the fixed
 * 512-item (32x16) group in __local memory, then ONE global atomic per group
 * per field (skipped when the group sum is 0, as upstream).  Launch
 * contract: work-group size exactly 32x16; the NDRange may over-cover (OOB
 * items contribute 0 via the width/height guard).  Integer sums are
 * order-exact, so any reduction tree shape and any atomic order give
 * bit-identical totals.  Grid: 2D ceil(width/32)*32 x ceil(height/16)*16.
 * // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_count_cmflags(
    __global int* __restrict dst_move,
    __global int* __restrict dst_shima,
    __global int* __restrict dst_lshima,
    __global const uchar* __restrict c0_x,
    __global const uchar* __restrict c0_y,
    __global const uchar* __restrict c1_x,
    __global const uchar* __restrict c1_y, int pitch,
    int width, int height, int parity, int thM, int thS, int thLS)
{
    int bx = (int)get_global_id(0);
    int by = (int)get_global_id(1);
    int lid = (int)get_local_id(0)
        + (int)get_local_id(1) * (int)get_local_size(0);
    __local int sbuf[KCA_FM_THREADS * 3];
    for (int i = 0; i < 2; ++i) {
        int cnt0 = 0, cnt1 = 0, cnt2 = 0;
        if (bx < width && by < height) {
            int off = bx + by * pitch;
            int vx = (i == 0) ? (int)c0_x[off] : (int)c1_x[off];
            int vy = (i == 0) ? (int)c0_y[off] : (int)c1_y[off];
            if (vy >= thM) cnt0 = 1;
            if (vx >= thS) cnt1 = 1;
            if (vx >= thLS) cnt2 = 1;
        }
        sbuf[lid] = cnt0;
        sbuf[lid + 512] = cnt1;
        sbuf[lid + 1024] = cnt2;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = 256; s > 0; s >>= 1) {
            if (lid < s) {
                sbuf[lid] += sbuf[lid + s];
                sbuf[lid + 512] += sbuf[lid + 512 + s];
                sbuf[lid + 1024] += sbuf[lid + 1024 + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            int slot = i ^ (parity ? 0 : 1); /* i ^ !parity */
            if (sbuf[0] > 0) atomic_add(&dst_move[slot], sbuf[0]);
            if (sbuf[512] > 0) atomic_add(&dst_shima[slot], sbuf[512]);
            if (sbuf[1024] > 0)
                atomic_add(&dst_lshima[slot], sbuf[1024]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/* ---------------------------------------------------------------------------
 * kf_count_cmflags_2planes — dual-plane FMCount census
 * (kl_count_cmflags_2planes twin; U+V fused — per-item counts run 0..2).
 * Same structure/contract as kf_count_cmflags.  Upstream's CPU path instead
 * calls the single-plane twin twice (U then V); the fused form provably
 * equals that composition (mode Q's golden IS the two-singles composition
 * while the mirror runs the fused loop).  // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_count_cmflags_2planes(
    __global int* __restrict dst_move,
    __global int* __restrict dst_shima,
    __global int* __restrict dst_lshima,
    __global const uchar* __restrict c0Ux,
    __global const uchar* __restrict c0Uy,
    __global const uchar* __restrict c1Ux,
    __global const uchar* __restrict c1Uy,
    __global const uchar* __restrict c0Vx,
    __global const uchar* __restrict c0Vy,
    __global const uchar* __restrict c1Vx,
    __global const uchar* __restrict c1Vy, int pitch,
    int width, int height, int parity, int thM, int thS, int thLS)
{
    int bx = (int)get_global_id(0);
    int by = (int)get_global_id(1);
    int lid = (int)get_local_id(0)
        + (int)get_local_id(1) * (int)get_local_size(0);
    __local int sbuf[KCA_FM_THREADS * 3];
    for (int i = 0; i < 2; ++i) {
        int cnt0 = 0, cnt1 = 0, cnt2 = 0;
        if (bx < width && by < height) {
            int off = bx + by * pitch;
            int vx = (i == 0) ? (int)c0Ux[off] : (int)c1Ux[off];
            int vy = (i == 0) ? (int)c0Uy[off] : (int)c1Uy[off];
            if (vy >= thM) cnt0++;
            if (vx >= thS) cnt1++;
            if (vx >= thLS) cnt2++;
            vx = (i == 0) ? (int)c0Vx[off] : (int)c1Vx[off];
            vy = (i == 0) ? (int)c0Vy[off] : (int)c1Vy[off];
            if (vy >= thM) cnt0++;
            if (vx >= thS) cnt1++;
            if (vx >= thLS) cnt2++;
        }
        sbuf[lid] = cnt0;
        sbuf[lid + 512] = cnt1;
        sbuf[lid + 1024] = cnt2;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = 256; s > 0; s >>= 1) {
            if (lid < s) {
                sbuf[lid] += sbuf[lid + s];
                sbuf[lid + 512] += sbuf[lid + 512 + s];
                sbuf[lid + 1024] += sbuf[lid + 1024 + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            int slot = i ^ (parity ? 0 : 1); /* i ^ !parity */
            if (sbuf[0] > 0) atomic_add(&dst_move[slot], sbuf[0]);
            if (sbuf[512] > 0) atomic_add(&dst_shima[slot], sbuf[512]);
            if (sbuf[1024] > 0)
                atomic_add(&dst_lshima[slot], sbuf[1024]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
