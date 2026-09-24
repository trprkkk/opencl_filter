/* ============================================================================
 * PROVISIONAL / UNVERIFIED — device-run comparison mandatory.
 * avscuda_conditional_rig.cl — faithful transcriptions of the two FLOAT
 * Conditional reductions in `AvsCUDA/filters/ConditionalFunctions.cu`
 * (kl_sum_of_pixels<float4,float,float>, kl_sad<float4,float,float>).
 * NOT covered by `make test`.  // RIG-VERIFY (both kernels below).
 *
 * Why these cannot be ALG-VERIFIED here: the reduction adds floats across
 * work-groups through a global atomicAdd, whose arrival order is UNDEFINED —
 * the CUDA result itself is nondeterministic at rounding level, so no
 * golden can define "the" bit-exact answer.  The ports below are faithful
 * transcriptions whose device output must be compared against CUDA output
 * with a tolerance (relative ~1e-6 of the accumulator magnitude), over
 * several runs to sample arrival orders.
 *
 * Transcription notes (two substitutions, both documented):
 * 1. The CUDA warp-shuffle tree becomes a portable __local sequential-
 *    addressing tree over 256 work-items (required work-group size 16x16,
 *    matching SUM_TH_W/H; launch with exactly that local size — enforced by
 *    reqd_work_group_size(16,16,1) on both kernels: any other local size
 *    fails at enqueue with CL_INVALID_WORK_GROUP_SIZE instead of silently
 *    computing garbage).  Halving
 *    order: stride 128, 64, ..., 1; within a step, item tid adds item
 *    tid+stride (unfused float adds).  Any intra-block order is as (in)valid
 *    as the shuffle order given the global nondeterminism.
 * 2. The global atomicAdd(float) becomes a 1.2-core compare-and-swap loop on
 *    the bit pattern (ka_atomic_add_f32) — the same atomic RMW-add operation
 *    without needing cl_ext_float_atomics.
 * Per-lane math is verbatim: sum lanes add raw; sad lanes add |a-b| (fabs).
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif
/* NOTE: this file is width-independent (float only); PX is unused but the
 * guard above keeps the lint invocation uniform across kernel files. */

/* Atomic float add via CAS on the bit pattern (OpenCL 1.2 core). */
inline void ka_atomic_add_f32(__global float* addr, float val)
{
    __global uint* uaddr = (__global uint*)addr;
    uint old = *uaddr, assumed;
    do {
        assumed = old;
        float sum = as_float(assumed) + val;
        old = atomic_cmpxchg(uaddr, assumed, as_uint(sum));
    } while (old != assumed);
}

/* ---------------------------------------------------------------------------
 * ka_sum_pixels_f32 — // RIG-VERIFY (see file header).
 * -------------------------------------------------------------------------*/
kernel __attribute__((reqd_work_group_size(16, 16, 1))) void ka_sum_pixels_f32(
    __global const float* __restrict src, int width, int height, int pitch,
    int maxv, __global float* __restrict sum)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    int tid = (int)get_local_id(0) + (int)get_local_id(1) * 16;
    (void)maxv;
    float tmpsum = 0.0f;
    if (x < width && y < height) {
        tmpsum = src[x + y * pitch];
    }
    __local float sbuf[256];
    sbuf[tid] = tmpsum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sbuf[tid] = sbuf[tid] + sbuf[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        ka_atomic_add_f32(sum, sbuf[0]);
    }
}

/* ---------------------------------------------------------------------------
 * ka_sad_f32 — // RIG-VERIFY (see file header).
 * -------------------------------------------------------------------------*/
kernel __attribute__((reqd_work_group_size(16, 16, 1))) void ka_sad_f32(
    __global const float* __restrict src0,
    __global const float* __restrict src1,
    int width, int height, int pitch,
    int maxv, __global float* __restrict sum)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    int tid = (int)get_local_id(0) + (int)get_local_id(1) * 16;
    (void)maxv;
    float tmpsum = 0.0f;
    if (x < width && y < height) {
        int off = x + y * pitch;
        tmpsum = fabs(src0[off] - src1[off]);
    }
    __local float sbuf[256];
    sbuf[tid] = tmpsum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sbuf[tid] = sbuf[tid] + sbuf[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        ka_atomic_add_f32(sum, sbuf[0]);
    }
}
