/* ============================================================================
 * nnedi3_prescreen.cl — OpenCL port of the NNEDI3 prescreener in upstream
 * rigaya/NNEDI3 (submodule of AviSynthCUDAFilters @68aef6e),
 * NNEDI3/nnedi3/nnedi3_kernel.cu @01931aa (GPL upstream):
 *   kl_prescreening (:135-238) -> kn_prescreening
 *
 * What it does (per 4-pixel vector): runs the tiny prescreener net over a
 * 12x4 pixel neighbourhood, marks the lanes the net rejects (result <= 0)
 * into a compacted per-group work list (workNN + numblocks) for the
 * expensive kl_compute_nn pass, and writes the cheap bicubic interpolation
 * into dst for every lane of any vector the net did NOT fully reject.
 *
 * Geometry (fixed, from upstream PRE_BLOCK_W/H = 32/16):
 *   local size MUST be exactly (32, 16, 1) — enforced by
 *   reqd_work_group_size; __local staging + the 512-wide scan depend on it.
 *   Grid: (nblocks(width4, 32), nblocks(height, 16)) groups, i.e. one
 *   work-item per 4-pixel output vector, over-covering the frame (guarded).
 *   Group linear id `bid = get_group_id(0) + get_group_id(1)*get_num_groups(0)`
 *   must match the compute_nn pass's view of the same grid.
 *
 * Pointer/layout contract (all set up host-side, verbatim from EvalCUDA:651):
 *   - `ref` points at `(frame_ref - refpitch - 8)` in PIXELS: the kernel's
 *     (xbase,ybase) origin is one row up and 8 pixels (2 vectors) left of
 *     the pixel being interpolated.
 *   - `ref_pitch4`/`dst_pitch4`: pitch in 4-PIXEL VECTORS (upstream's
 *     byte pitch / 4); x indices below are vector indices, and this port
 *     multiplies by 4 internally for its scalar loads.
 *   - `ws`: 64 short4 = 256 shorts, flattened; `ws[i]` component c is
 *     ws_flat[i*4+c].  `wf`: 7 float4 = 28 floats, flattened likewise.
 *     Upstream derives both from one weights0 blob (`wf = &ws[64]`).
 *   - `work_nn`: 2 bytes per entry (upstream uchar2 {x,y}), group slice at
 *     `bid * 32*4*16` entries = `bid * 2048` entries = 4096 bytes.
 *   - `numblocks[bid]`: count of rejected lanes in that group.
 *
 * Determinism of the work list: upstream compacts with an inclusive
 * warp-shuffle add-scan over tid (dev_scan, ReduceKernel.cuh:448) and then
 * subtracts its own count to get an exclusive offset.  This port uses a
 * Hillis-Steele scan in __local instead: integer addition is associative
 * and the tid order is identical, so the resulting offsets — and hence the
 * whole workNN layout — are bit-identical, not merely equivalent.
 *
 * Arithmetic: the neighbourhood dot product is exact int32 (transcribed
 * tap-for-tap, including upstream's asymmetric x==0 / x<4 / x==4 lane
 * split).  The float tail (scale+bias, the t/(|t|+1) "elliott" squash, the
 * 4 weighted accumulations and the final bias) is unfused per-operation
 * float32, accumulated in upstream's order; the host build must disable FP
 * contraction for bit-exactness (nvcc's default --fmad=true may fuse the
 * MUL+ADDs — cross-toolchain device comparison needs rounding tolerance),
 * same caveat as avscuda_resample.cl.
 * // ALG-VERIFIED via python/run_nnedi3_prescreen.py.
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

#define PRE_BLOCK_W 32
#define PRE_BLOCK_H 16
#define PRE_BLOCK_N (PRE_BLOCK_W * PRE_BLOCK_H)

kernel __attribute__((reqd_work_group_size(PRE_BLOCK_W, PRE_BLOCK_H, 1)))
void kn_prescreening(
    __global       PX*    __restrict dst,      int dst_pitch4,
    __global const PX*    __restrict ref,      int ref_pitch4,
    __global const short* __restrict ws,       /* 64 short4, flattened */
    __global const float* __restrict wf,       /* 7  float4, flattened */
    __global       uchar* __restrict work_nn,  /* 2 bytes per entry */
    __global       int*   __restrict numblocks,
    int width4, int height, int val_min, int val_max)
{
    int tx  = get_local_id(0);
    int ty  = get_local_id(1);
    int tid = tx + ty * PRE_BLOCK_W;

    __local short sws[64 * 4];
    __local float swf[7 * 4];

    if (tid < 64) {
        sws[tid * 4 + 0] = ws[tid * 4 + 0];
        sws[tid * 4 + 1] = ws[tid * 4 + 1];
        sws[tid * 4 + 2] = ws[tid * 4 + 2];
        sws[tid * 4 + 3] = ws[tid * 4 + 3];
    }
    if (tid < 7) {
        swf[tid * 4 + 0] = wf[tid * 4 + 0];
        swf[tid * 4 + 1] = wf[tid * 4 + 1];
        swf[tid * 4 + 2] = wf[tid * 4 + 2];
        swf[tid * 4 + 3] = wf[tid * 4 + 3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int xbase = tx + get_group_id(0) * PRE_BLOCK_W;
    int ybase = ty + get_group_id(1) * PRE_BLOCK_H;

    /* upstream inits to {1,1,1,1} so out-of-range items reject nothing */
    float res0 = 1.0f, res1 = 1.0f, res2 = 1.0f, res3 = 1.0f;

    if (xbase < width4 && ybase < height) {
        int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 5; ++x) {
                int vbase = ((x + xbase) + (y + ybase) * ref_pitch4) * 4;
                int vx = (int)ref[vbase + 0];
                int vy = (int)ref[vbase + 1];
                int vz = (int)ref[vbase + 2];
                int vw = (int)ref[vbase + 3];
                if (x == 0) {
                    int a = (0 + y * 16) * 4, b = (1 + y * 16) * 4;
                    sum0 += (int)sws[a + 0] * vz;  sum1 += (int)sws[a + 1] * vz;
                    sum2 += (int)sws[a + 2] * vz;  sum3 += (int)sws[a + 3] * vz;
                    sum0 += (int)sws[b + 0] * vw;  sum1 += (int)sws[b + 1] * vw;
                    sum2 += (int)sws[b + 2] * vw;  sum3 += (int)sws[b + 3] * vw;
                } else if (x < 4) {
                    int a = ((x * 4 - 2) + y * 16) * 4;
                    int b = ((x * 4 - 1) + y * 16) * 4;
                    int c = ((x * 4 + 0) + y * 16) * 4;
                    int d = ((x * 4 + 1) + y * 16) * 4;
                    sum0 += (int)sws[a + 0] * vx;  sum1 += (int)sws[a + 1] * vx;
                    sum2 += (int)sws[a + 2] * vx;  sum3 += (int)sws[a + 3] * vx;
                    sum0 += (int)sws[b + 0] * vy;  sum1 += (int)sws[b + 1] * vy;
                    sum2 += (int)sws[b + 2] * vy;  sum3 += (int)sws[b + 3] * vy;
                    sum0 += (int)sws[c + 0] * vz;  sum1 += (int)sws[c + 1] * vz;
                    sum2 += (int)sws[c + 2] * vz;  sum3 += (int)sws[c + 3] * vz;
                    sum0 += (int)sws[d + 0] * vw;  sum1 += (int)sws[d + 1] * vw;
                    sum2 += (int)sws[d + 2] * vw;  sum3 += (int)sws[d + 3] * vw;
                } else {
                    int a = (14 + y * 16) * 4, b = (15 + y * 16) * 4;
                    sum0 += (int)sws[a + 0] * vx;  sum1 += (int)sws[a + 1] * vx;
                    sum2 += (int)sws[a + 2] * vx;  sum3 += (int)sws[a + 3] * vx;
                    sum0 += (int)sws[b + 0] * vy;  sum1 += (int)sws[b + 1] * vy;
                    sum2 += (int)sws[b + 2] * vy;  sum3 += (int)sws[b + 3] * vy;
                }
            }
        }

        /* t = (float)sum * swf[0] + swf[1]; val = t / (|t| + 1) */
        float t0 = (float)sum0 * swf[0] + swf[4];
        float t1 = (float)sum1 * swf[1] + swf[5];
        float t2 = (float)sum2 * swf[2] + swf[6];
        float t3 = (float)sum3 * swf[3] + swf[7];
        float val0 = t0 / (fabs(t0) + 1.0f);
        float val1 = t1 / (fabs(t1) + 1.0f);
        float val2 = t2 / (fabs(t2) + 1.0f);
        float val3 = t3 / (fabs(t3) + 1.0f);

        /* sumf = swf[2]*val.x + swf[3]*val.y + swf[4]*val.z + swf[5]*val.w
         * (accumulated in that order), then result = sumf + swf[6] */
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        s0 += swf[8 + 0] * val0;  s1 += swf[8 + 1] * val0;
        s2 += swf[8 + 2] * val0;  s3 += swf[8 + 3] * val0;
        s0 += swf[12 + 0] * val1; s1 += swf[12 + 1] * val1;
        s2 += swf[12 + 2] * val1; s3 += swf[12 + 3] * val1;
        s0 += swf[16 + 0] * val2; s1 += swf[16 + 1] * val2;
        s2 += swf[16 + 2] * val2; s3 += swf[16 + 3] * val2;
        s0 += swf[20 + 0] * val3; s1 += swf[20 + 1] * val3;
        s2 += swf[20 + 2] * val3; s3 += swf[20 + 3] * val3;
        res0 = s0 + swf[24 + 0];  res1 = s1 + swf[24 + 1];
        res2 = s2 + swf[24 + 2];  res3 = s3 + swf[24 + 3];
    }

    int num = 0;
    if (res0 <= 0.0f) ++num;
    if (res1 <= 0.0f) ++num;
    if (res2 <= 0.0f) ++num;
    if (res3 <= 0.0f) ++num;

    /* exclusive add-scan of `num` over tid (upstream: dev_scan then -num) */
    __local int sbuf[PRE_BLOCK_N];
    sbuf[tid] = num;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 1; off < PRE_BLOCK_N; off <<= 1) {
        int add = (tid >= off) ? sbuf[tid - off] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        sbuf[tid] += add;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int idx = sbuf[tid] - num;

    int bid = get_group_id(0) + get_group_id(1) * get_num_groups(0);
    int workoff = bid * PRE_BLOCK_W * 4 * PRE_BLOCK_H;

    if (res0 <= 0.0f) {
        work_nn[(workoff + idx) * 2 + 0] = (uchar)(tx * 4 + 0);
        work_nn[(workoff + idx) * 2 + 1] = (uchar)ty;
        ++idx;
    }
    if (res1 <= 0.0f) {
        work_nn[(workoff + idx) * 2 + 0] = (uchar)(tx * 4 + 1);
        work_nn[(workoff + idx) * 2 + 1] = (uchar)ty;
        ++idx;
    }
    if (res2 <= 0.0f) {
        work_nn[(workoff + idx) * 2 + 0] = (uchar)(tx * 4 + 2);
        work_nn[(workoff + idx) * 2 + 1] = (uchar)ty;
        ++idx;
    }
    if (res3 <= 0.0f) {
        work_nn[(workoff + idx) * 2 + 0] = (uchar)(tx * 4 + 3);
        work_nn[(workoff + idx) * 2 + 1] = (uchar)ty;
        ++idx;
    }

    /* last item holds the group total (its exclusive base + its own count) */
    if (tid == PRE_BLOCK_N - 1) numblocks[bid] = idx;

    if (num < 4 && xbase < width4 && ybase < height) {
        /* bicubic: ((s2+s4)*19 - (s3p+s6)*3 + 16) >> 5, clamped */
        int b0 = ((xbase + 2) + (ybase + 0) * ref_pitch4) * 4;
        int b1 = ((xbase + 2) + (ybase + 1) * ref_pitch4) * 4;
        int b2 = ((xbase + 2) + (ybase + 2) * ref_pitch4) * 4;
        int b3 = ((xbase + 2) + (ybase + 3) * ref_pitch4) * 4;
        int dout = (xbase + ybase * dst_pitch4) * 4;
        for (int k = 0; k < 4; ++k) {
            int s3p = (int)ref[b0 + k];
            int s2  = (int)ref[b1 + k];
            int s4  = (int)ref[b2 + k];
            int s6  = (int)ref[b3 + k];
            int tmp = (((s2 + s4) * 19 - (s3p + s6) * 3 + 16) >> 5);
            tmp = clamp(tmp, val_min, val_max);
            dst[dout + k] = (PX)tmp;
        }
    }
}
