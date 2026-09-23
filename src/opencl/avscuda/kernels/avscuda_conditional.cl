/* ============================================================================
 * avscuda_conditional.cl — OpenCL port of the AvsCUDA Conditional / runtime
 * metric kernels in `AvsCUDA/filters/ConditionalFunctions.cu`
 * (rigaya/AviSynthCUDAFilters, MIT): AveragePlane, PlaneDifference
 * (ComparePlane/LumaDifference) and MinMaxPlane histogram paths.
 *
 * Upstream reduces with a 16x16 shared-memory tree + one atomicAdd per block.
 * For INTEGER sums the reduction order is immaterial (exact arithmetic), so
 * the ports below use one atomic_add per work-item straight into the global
 * counter — the same metric with no arrival-order dependence.  The FLOAT sum/
 * SAD reductions are order-sensitive AND need float atomics, so they live as
 * faithful // RIG-VERIFY transcriptions in avscuda_conditional_rig.cl
 * (device-run comparison mandatory); the float HISTOGRAM is deterministic
 * (pure per-lane index math + int atomics) and is verified here.
 *
 * Launch/host contracts (same file):
 * - width MUST be a multiple of 4 (host throws otherwise on all three paths).
 * - Counters are pre-zeroed via ka_init_sum_* (kl_init_sum twin, <<<1,1>>>).
 * - 32 vs 64 bit counters: host picks u64 unless
 *   total_pixels * (1<<bits_per_pixel) <= INT_MAX (device threshold; the CPU
 *   path uses 255/65535 instead — slightly less conservative).  The u32
 *   kernels preserve 32-bit wraparound; u64 needs OpenCL 2.0 or
 *   cl_khr_int64_base_atomics (atom_add).
 * - Histograms: length = 1<<min(16,bits) (256 / 1K / 4K / 16K / 64K), int
 *   counters pre-zeroed via ka_init_hist (kl_init_hist twin).
 * - Average / SAD-per-pixel / RGB *4/3 rescaling are host-side divisions of
 *   the counter (calc_sum_of_pixels / calc_sad return double).
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

/* ---------------------------------------------------------------------------
 * ka_init_sum_u32 / _u64 / _f32 — zero a reduction counter (kl_init_sum twin;
 * upstream launches <<<1,1>>>; here (ptr, length) + guard per repo convention,
 * production length is 1).
 * Grid: 1D (length).
 * // ALG-VERIFIED via python/run_avscuda_conditional.py.
 * -------------------------------------------------------------------------*/
kernel void ka_init_sum_u32(__global uint* __restrict sum, int length)
{
    int i = (int)get_global_id(0);
    if (i >= length) return;
    sum[i] = 0u;
}

kernel void ka_init_sum_u64(__global ulong* __restrict sum, int length)
{
    int i = (int)get_global_id(0);
    if (i >= length) return;
    sum[i] = 0ul;
}

kernel void ka_init_sum_f32(__global float* __restrict sum, int length)
{
    int i = (int)get_global_id(0);
    if (i >= length) return;
    sum[i] = 0.0f;
}

/* ---------------------------------------------------------------------------
 * ka_sum_pixels_u8_u32 / _u64 — AveragePlane pixel sum over uchar (identity
 * clamp_to_range; maxv kept for signature fidelity, ignored).
 * Grid: 2D (width, height); width a multiple of 4.
 * // ALG-VERIFIED via python/run_avscuda_conditional.py.
 * -------------------------------------------------------------------------*/
kernel void ka_sum_pixels_u8_u32(
    __global const uchar* __restrict src, int width, int height, int pitch,
    int maxv, __global uint* __restrict sum)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    (void)maxv;
    atomic_add(sum, (uint)src[x + y * pitch]);
}

kernel void ka_sum_pixels_u8_u64(
    __global const uchar* __restrict src, int width, int height, int pitch,
    int maxv, __global ulong* __restrict sum)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    (void)maxv;
    atom_add(sum, (ulong)src[x + y * pitch]);
}

/* ---------------------------------------------------------------------------
 * ka_sum_pixels_u16_u32 / _u64 — AveragePlane pixel sum over ushort with the
 * min(v,maxv) content clamp (maxv = (1<<bits)-1, bits in {10,12,14,16}).
 * Grid: 2D (width, height); width a multiple of 4.
 * // ALG-VERIFIED via python/run_avscuda_conditional.py.
 * -------------------------------------------------------------------------*/
kernel void ka_sum_pixels_u16_u32(
    __global const ushort* __restrict src, int width, int height, int pitch,
    int maxv, __global uint* __restrict sum)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    atomic_add(sum, (uint)min((int)src[x + y * pitch], maxv));
}

kernel void ka_sum_pixels_u16_u64(
    __global const ushort* __restrict src, int width, int height, int pitch,
    int maxv, __global ulong* __restrict sum)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    atom_add(sum, (ulong)min((int)src[x + y * pitch], maxv));
}

/* ---------------------------------------------------------------------------
 * ka_sad_u8_u32 / _u64 — PlaneDifference |a-b| sum over uchar, single pitch
 * (both planes share one pitch upstream).
 * Grid: 2D (width, height); width a multiple of 4.
 * // ALG-VERIFIED via python/run_avscuda_conditional.py.
 * -------------------------------------------------------------------------*/
kernel void ka_sad_u8_u32(
    __global const uchar* __restrict src0,
    __global const uchar* __restrict src1,
    int width, int height, int pitch,
    int maxv, __global uint* __restrict sum)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    (void)maxv;
    int off = x + y * pitch;
    int d = (int)src0[off] - (int)src1[off];
    atomic_add(sum, (uint)((d >= 0) ? d : -d));
}

kernel void ka_sad_u8_u64(
    __global const uchar* __restrict src0,
    __global const uchar* __restrict src1,
    int width, int height, int pitch,
    int maxv, __global ulong* __restrict sum)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    (void)maxv;
    int off = x + y * pitch;
    int d = (int)src0[off] - (int)src1[off];
    atom_add(sum, (ulong)((d >= 0) ? d : -d));
}

/* ---------------------------------------------------------------------------
 * ka_sad_u16_u32 / _u64 — |min(a,maxv)-min(b,maxv)| sum over ushort.
 * Grid: 2D (width, height); width a multiple of 4.
 * // ALG-VERIFIED via python/run_avscuda_conditional.py.
 * -------------------------------------------------------------------------*/
kernel void ka_sad_u16_u32(
    __global const ushort* __restrict src0,
    __global const ushort* __restrict src1,
    int width, int height, int pitch,
    int maxv, __global uint* __restrict sum)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    int off = x + y * pitch;
    int d = min((int)src0[off], maxv) - min((int)src1[off], maxv);
    atomic_add(sum, (uint)((d >= 0) ? d : -d));
}

kernel void ka_sad_u16_u64(
    __global const ushort* __restrict src0,
    __global const ushort* __restrict src1,
    int width, int height, int pitch,
    int maxv, __global ulong* __restrict sum)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    int off = x + y * pitch;
    int d = min((int)src0[off], maxv) - min((int)src1[off], maxv);
    atom_add(sum, (ulong)((d >= 0) ? d : -d));
}

/* ---------------------------------------------------------------------------
 * ka_count_hist_u8 / _u16 / _f32 — MinMaxPlane histogram (kl_count_hist
 * twins; per-lane atomic +1 into int counters; length arg + guard is the
 * repo convention — in-production indices are bounded by construction:
 * u8 <= 255 < 256; u16 <= maxv < 1<<bits; f32 <= 65535 < 65536, NaN
 * included (NaN -> 65535 via the fmin/fmax rule, matching CUDA)).
 * Grid: 2D (width, height); width a multiple of 4.
 * // ALG-VERIFIED via python/run_avscuda_conditional.py.
 * -------------------------------------------------------------------------*/
kernel void ka_count_hist_u8(
    __global const uchar* __restrict src, int width, int height, int pitch,
    int maxv, __global int* __restrict hist, int length)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    (void)maxv;
    int idx = (int)src[x + y * pitch];
    if (idx < length) atomic_add(&hist[idx], 1);
}

kernel void ka_count_hist_u16(
    __global const ushort* __restrict src, int width, int height, int pitch,
    int maxv, __global int* __restrict hist, int length)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    int idx = min((int)src[x + y * pitch], maxv);
    if (idx < length) atomic_add(&hist[idx], 1);
}

kernel void ka_count_hist_f32(
    __global const float* __restrict src, int width, int height, int pitch,
    int maxv, __global int* __restrict hist, int length)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    (void)maxv;
    float t = src[x + y * pitch] * 65535.0f + 0.5f;
    float c = max(0.0f, min(t, 65535.0f));
    int idx = (int)c;
    if (idx < length) atomic_add(&hist[idx], 1);
}

/* ---------------------------------------------------------------------------
 * ka_init_hist — zero an int histogram (kl_init_hist twin; already
 * length-guarded upstream).
 * Grid: 1D (length).
 * // ALG-VERIFIED via python/run_avscuda_conditional.py.
 * -------------------------------------------------------------------------*/
kernel void ka_init_hist(__global int* __restrict hist, int length)
{
    int i = (int)get_global_id(0);
    if (i >= length) return;
    hist[i] = 0;
}
