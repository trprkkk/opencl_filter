/* kfm_decombeucf.cl — OpenCL port of the reduction/accumulation device kernels in
 * upstream KFM/DecombeUCF.cu (@8e086bb).  Covers the 7 remaining __global__
 * kernels AFTER kl_noise_clip (ported earlier in kfm_combinganalyze.cl):
 *   kl_init_uint64          -> kf_init_uint64          (single-item zero-fill)
 *   kl_calculate_field_diff -> kf_calculate_field_diff (gated 5-tap combe sum,
 *                              32x16 group reduce, uint64 atomic add)
 *   kl_init_block_sum       -> kf_init_block_sum       (two-plane zero-fill +
 *                              highest_sum[0] = 0)
 *   kl_add_block_sum        -> kf_add_block_sum        (per-block sumAbs/sumSig
 *                              accumulate, BLOCK_SIZE 4/8/16/32)
 *   kl_block_sum_max        -> kf_block_sum_max        (per-cell metric max,
 *                              group max-reduce, atomic max)
 *   kl_analyze_noise        -> kf_analyze_noise        (4-way |.-128|/|delta|
 *                              sums, uint64 atomic adds)
 *   kl_analyze_diff         -> kf_analyze_diff         (combe sum0 + field-mixed
 *                              TFF sum1, uint64 atomic adds)
 *
 * Scalar port: one work-item per pixel (PX = uchar/ushort for the 8/16-bit
 * paths) or per block cell.  Pitches/widths are in PIXELS (upstream counts
 * vpixel_t vectors; width_px = 4 * width4).  Padded inputs (field_diff,
 * analyze_diff) carry 2 extra rows above and below; the origin pointer points
 * at unpadded row 0 so the kernel text keeps upstream's (y-2)..(y+2) indexing
 * verbatim, exactly as upstream passes the padded frame's origin.
 *
 * Fidelity notes (verify by diffing against DecombeUCF.cu + ReduceKernel.cuh):
 *  - calc_combe_5tap transcribes KFMFilterBase.cuh's CalcCombe(a,b,c,d,e) =
 *    abs(a + c*4 + e - (b+d)*3) lane-for-lane (a=y-2 .. e=y+2); the scalar
 *    port evaluates one lane per pixel instead of one int4 per vector.
 *  - dev_reduce/dev_reduceN are warp-shuffle trees with a shared-memory
 *    staging phase; the port uses a full __local halving tree instead.  The
 *    reduced ops (int add, int max) are associative AND commutative, so every
 *    tree shape — and the serial golden — is value-identical.  The N=2/N=4
 *    bufs keep upstream's buf[i*MAX+tid] lane-major layout.
 *  - kf_add_block_sum serialises the TH_Z thread decomposition (upstream
 *    splits each block's pixels across threads, then block-sums); per-cell
 *    totals are order-exact int adds, and the racy-looking `+=` is race-free
 *    (single writer per cell) in both versions.  BLOCK_SIZE (32/16/8/4,
 *    upstream's template table) is a runtime arg here; all four are verified.
 *  - kf_block_sum_max reads the sum planes as interleaved int quads
 *    (cell c, lane k at [c*4+k]) — the scalar view of upstream's (int4*)
 *    reinterpret cast; the 0 floor (upstream's tmpmax = 0 seed) and the
 *    metric sumAbs + sumSig*4 are preserved.
 *  - kf_init_uint64 / kf_init_block_sum take an explicit length + guard; the
 *    kernels are otherwise verbatim (upstream relies on launching exactly N
 *    threads).  The `x == 0 -> maxSum[0] = 0` rule is kept.
 *  - Gating/branches kept verbatim: combe > nt (strict); analyze_diff's
 *    TFF weave (odd y: f0,f1,f0,f1,f0 / even y: f1,f0,f1,f0,f1).
 *
 * Launch contracts: the 32x16 reduce kernels (field_diff, block_sum_max,
 * analyze_*) REQUIRE local size 32x16 (tid = lx + ly*32 over a 512-slot
 * __local buf).  init/add kernels accept any 1-D/2-D decomposition.
 *
 * Atomics: OpenCL-2.0-style atomic_add/atomic_max; the 64-bit adds are
 * additionally covered by cl_khr_int64_base_atomics for OpenCL 1.2 devices.
 * // RIG-VERIFY: devices with neither need a per-group-partials + host-sum
 * fallback (not implemented); the single-group test path is unaffected.
 *
 * Status: // ALG-VERIFIED (python/run_kfm_decombeucf.py) vs
 *   sim/kfm_decombeucf_ref.cpp.
 */
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#define KDUC_X 32
#define KDUC_Y 16
#define KDUC_THREADS (KDUC_X * KDUC_Y) /* 512, upstream CALC_FIELD_DIFF_THREADS */

/* 5-tap vertical combe: upstream KFMFilterBase.cuh CalcCombe, one lane. */
inline int calc_combe_5tap(int a, int b, int c, int d, int e) {
    return abs(a + c * 4 + e - (b + d) * 3);
}

/* Upstream kl_init_uint64 (+ explicit length; upstream launches exactly N). */
__kernel void kf_init_uint64(__global ulong *sum, int length) {
    int x = get_global_id(0);
    if (x < length) {
        sum[x] = 0;
    }
}

/* Upstream kl_calculate_field_diff.  ptr = unpadded origin of a +-2-row padded
 * plane; width/height/pitch in pixels.  Local size MUST be 32x16. */
__kernel void kf_calculate_field_diff(__global const PX *ptr, int nt,
                                      int width, int height, int pitch,
                                      __global ulong *sum) {
    __local int sbuf[KDUC_THREADS];
    int x = get_global_id(0);
    int y = get_global_id(1);
    int tid = get_local_id(0) + get_local_id(1) * KDUC_X;

    int tmpsum = 0;
    if (x < width && y < height) {
        int s0 = ptr[x + (y - 2) * pitch];
        int s1 = ptr[x + (y - 1) * pitch];
        int s2 = ptr[x + (y + 0) * pitch];
        int s3 = ptr[x + (y + 1) * pitch];
        int s4 = ptr[x + (y + 2) * pitch];
        int combe = calc_combe_5tap(s0, s1, s2, s3, s4);
        tmpsum = (combe > nt) ? combe : 0;
    }

    sbuf[tid] = tmpsum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = KDUC_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sbuf[tid] += sbuf[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        /* tmpsum is non-negative (gated sums), matching upstream's implicit
         * int -> unsigned long long conversion in atomicAdd(sum, tmpsum). */
        atomic_add(sum, (ulong)sbuf[0]);
    }
}

/* Upstream kl_init_block_sum (+ explicit length; upstream launches exactly N).
 * sumAbs/sumSig hold blocks_w*blocks_h cells at row stride block_pitch. */
__kernel void kf_init_block_sum(__global int *sumAbs, __global int *sumSig,
                                __global int *maxSum, int length) {
    int x = get_global_id(0);
    if (x < length) {
        sumAbs[x] = 0;
        sumSig[x] = 0;
    }
    if (x == 0) {
        maxSum[0] = 0;
    }
}

/* Upstream kl_add_block_sum (BLOCK_SIZE = 4/8/16/32 via block_size).  One
 * work-item per block cell; the pixel loop is a serial transcription of the
 * TH_Z thread decomposition (order-exact int adds).  The += onto the init
 * sums is race-free (single writer per cell) as upstream. */
__kernel void kf_add_block_sum(__global const PX *src0, __global const PX *src1,
                               int width, int height, int pitch,
                               int blocks_w, int blocks_h, int block_pitch,
                               __global int *sumAbs, __global int *sumSig,
                               int block_size) {
    int bx = get_global_id(0);
    int by = get_global_id(1);
    if (bx >= blocks_w || by >= blocks_h) {
        return;
    }
    int abssum = 0;
    int sigsum = 0;
    for (int ty = 0; ty < block_size; ty++) {
        int y = by * block_size + ty;
        for (int tx = 0; tx < block_size; tx++) {
            int x = bx * block_size + tx;
            if (x >= width || y >= height) {
                continue;
            }
            int a = src0[y * pitch + x];
            int b = src1[y * pitch + x];
            abssum += abs(a - b);
            sigsum += a - b;
        }
    }
    int cell = bx + by * block_pitch;
    sumAbs[cell] += abssum;
    sumSig[cell] += sigsum;
}

/* Upstream kl_block_sum_max.  sumAbs/sumSig are interleaved int quads
 * (cell c lane k at [c*4+k], the scalar view of the (int4*) cast);
 * blocks_w/blocks_h/block_pitch count QUAD cells.  highest_sum is
 * atomicMax'ed (init 0 upstream).  Local size MUST be 32x16. */
__kernel void kf_block_sum_max(__global const int *sumAbs,
                               __global const int *sumSig,
                               int blocks_w, int blocks_h, int block_pitch,
                               __global int *highest_sum) {
    __local int sbuf[KDUC_THREADS];
    int x = get_global_id(0);
    int y = get_global_id(1);
    int tid = get_local_id(0) + get_local_id(1) * KDUC_X;

    int tmpmax = 0;
    if (x < blocks_w && y < blocks_h) {
        int c4 = (x + y * block_pitch) * 4;
        int m0 = sumAbs[c4 + 0] + sumSig[c4 + 0] * 4;
        int m1 = sumAbs[c4 + 1] + sumSig[c4 + 1] * 4;
        int m2 = sumAbs[c4 + 2] + sumSig[c4 + 2] * 4;
        int m3 = sumAbs[c4 + 3] + sumSig[c4 + 3] * 4;
        tmpmax = m0;
        if (m1 > tmpmax) tmpmax = m1;
        if (m2 > tmpmax) tmpmax = m2;
        if (m3 > tmpmax) tmpmax = m3;
        if (tmpmax < 0) tmpmax = 0;
    }

    sbuf[tid] = tmpmax;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = KDUC_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sbuf[tid] = max(sbuf[tid], sbuf[tid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        atomic_max(highest_sum, sbuf[0]);
    }
}

/* Upstream kl_analyze_noise (8-bit only upstream: vpixel_t = uchar4). */
__kernel void kf_analyze_noise(__global ulong *result,
                               __global const uchar *src0,
                               __global const uchar *src1,
                               __global const uchar *src2,
                               int width, int height, int pitch) {
    __local int sbuf[4 * KDUC_THREADS];
    int x = get_global_id(0);
    int y = get_global_id(1);
    int tid = get_local_id(0) + get_local_id(1) * KDUC_X;

    int v0 = 0, v1 = 0, v2 = 0, v3 = 0;
    if (x < width && y < height) {
        int a = src0[y * pitch + x];
        int b = src1[y * pitch + x];
        int c = src2[y * pitch + x];
        /* dev_horizontal_sum(abs(sN + (-128))) / abs(sN - sM), one lane. */
        v0 = abs(a - 128);
        v1 = abs(b - 128);
        v2 = abs(b - a);
        v3 = abs(c - b);
    }

    sbuf[0 * KDUC_THREADS + tid] = v0;
    sbuf[1 * KDUC_THREADS + tid] = v1;
    sbuf[2 * KDUC_THREADS + tid] = v2;
    sbuf[3 * KDUC_THREADS + tid] = v3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = KDUC_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sbuf[0 * KDUC_THREADS + tid] += sbuf[0 * KDUC_THREADS + tid + s];
            sbuf[1 * KDUC_THREADS + tid] += sbuf[1 * KDUC_THREADS + tid + s];
            sbuf[2 * KDUC_THREADS + tid] += sbuf[2 * KDUC_THREADS + tid + s];
            sbuf[3 * KDUC_THREADS + tid] += sbuf[3 * KDUC_THREADS + tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        atomic_add(&result[0], (ulong)sbuf[0 * KDUC_THREADS]);
        atomic_add(&result[1], (ulong)sbuf[1 * KDUC_THREADS]);
        atomic_add(&result[2], (ulong)sbuf[2 * KDUC_THREADS]);
        atomic_add(&result[3], (ulong)sbuf[3 * KDUC_THREADS]);
    }
}

/* Upstream kl_analyze_diff (8-bit only upstream: vpixel_t = uchar4).
 * f0/f1 = unpadded origins of +-2-row padded planes.  Local size 32x16. */
__kernel void kf_analyze_diff(__global ulong *result,
                              __global const uchar *f0,
                              __global const uchar *f1,
                              int width, int height, int pitch) {
    __local int sbuf[2 * KDUC_THREADS];
    int x = get_global_id(0);
    int y = get_global_id(1);
    int tid = get_local_id(0) + get_local_id(1) * KDUC_X;

    int s0 = 0, s1 = 0;
    if (x < width && y < height) {
        int a = f0[x + (y - 2) * pitch];
        int b = f0[x + (y - 1) * pitch];
        int c = f0[x + (y + 0) * pitch];
        int d = f0[x + (y + 1) * pitch];
        int e = f0[x + (y + 2) * pitch];
        s0 = calc_combe_5tap(a, b, c, d, e);

        /* TFF weave of the current bottom field with f1's top field. */
        if (y & 1) {
            a = f0[x + (y - 2) * pitch];
            b = f1[x + (y - 1) * pitch];
            c = f0[x + (y + 0) * pitch];
            d = f1[x + (y + 1) * pitch];
            e = f0[x + (y + 2) * pitch];
        } else {
            a = f1[x + (y - 2) * pitch];
            b = f0[x + (y - 1) * pitch];
            c = f1[x + (y + 0) * pitch];
            d = f0[x + (y + 1) * pitch];
            e = f1[x + (y + 2) * pitch];
        }
        s1 = calc_combe_5tap(a, b, c, d, e);
    }

    sbuf[0 * KDUC_THREADS + tid] = s0;
    sbuf[1 * KDUC_THREADS + tid] = s1;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = KDUC_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sbuf[0 * KDUC_THREADS + tid] += sbuf[0 * KDUC_THREADS + tid + s];
            sbuf[1 * KDUC_THREADS + tid] += sbuf[1 * KDUC_THREADS + tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        atomic_add(&result[0], (ulong)sbuf[0 * KDUC_THREADS]);
        atomic_add(&result[1], (ulong)sbuf[1 * KDUC_THREADS]);
    }
}
