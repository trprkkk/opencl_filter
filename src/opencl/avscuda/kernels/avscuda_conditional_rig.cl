/* ============================================================================
 * RIG-COMPARED (terminal state, 2026-10-28) — device comparison done.
 * avscuda_conditional_rig.cl — faithful transcriptions of the two FLOAT
 * Conditional reductions in `AvsCUDA/filters/ConditionalFunctions.cu`
 * (kl_sum_of_pixels<float4,float,float>, kl_sad<float4,float,float>).
 * NOT covered by `make test` (deliberately — see below).  Both kernels below
 * carry `// RIG-COMPARED 2026-10-28`; results appendix:
 * docs/RIG_HANDOFF_AVSCUDA_CONDITIONAL.md §8.  These kernels stay in this
 * file permanently: their terminal state is deliberately NOT ALG-VERIFIED.
 *
 * Why these cannot be ALG-VERIFIED: the reduction adds floats across
 * work-groups through a global atomicAdd, whose arrival order is UNDEFINED —
 * the CUDA result itself is nondeterministic at rounding level, so no
 * golden can define "the" bit-exact answer.  The ports below are faithful
 * transcriptions compared against CUDA output with a tolerance over many
 * runs sampling arrival orders.
 *
 * Comparison result (2026-10-28, RTX 2060 SUPER, 200 CUDA reps × 10 CL reps;
 * rig/cond_cuda.cu + rig/dev_cond_cuda.py, report
 * rig/RIG_CUDA_TWINS_REPORT.md, appendix in the handoff §8):
 *  - Phase A (exact-arithmetic anchors): 11/11 bit-exact CUDA == CL ==
 *    expected (incl. 1080p 0x497D2000; NaN → 0x7FFFFFFF on both sides).
 *  - Phase B: criterion-1 literal PASS 75/112; the 37 misses are all
 *    ≤36 ulp (typ. 1 ulp) with the RM-gate PASS on every one —
 *    leaf-association rounding, accepted deviation (handoff §3 stage 1 /
 *    §4.3-1).  The as-written criterion-2 bound is violated by the CUDA
 *    twin itself on 3 cells (sum 1080p small/video, sad 1080p small),
 *    confirming it under-covers both toolchains at large N.  Criterion-3
 *    (finiteness) PASS everywhere.  All within the documented CL host-twin
 *    envelope.
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
 * ka_sum_pixels_f32 — // RIG-COMPARED 2026-10-28 Phase A bit-exact 11/11 +
 * Phase B criterion-1 literal 75/112, 37 residuals <=36 ulp (typ. 1), all
 * RM-gate PASS; criterion-2 as-written violated by the CUDA twin itself on
 * 3 cells; criterion-3 PASS — 200 CUDA x 10 CL reps (handoff §8 appendix;
 * rig/RIG_CUDA_TWINS_REPORT.md).
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
 * ka_sad_f32 — // RIG-COMPARED 2026-10-28 Phase A bit-exact 11/11 +
 * Phase B criterion-1 literal 75/112, 37 residuals <=36 ulp (typ. 1), all
 * RM-gate PASS; criterion-2 as-written violated by the CUDA twin itself on
 * 3 cells; criterion-3 PASS — 200 CUDA x 10 CL reps (handoff §8 appendix;
 * rig/RIG_CUDA_TWINS_REPORT.md).
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
