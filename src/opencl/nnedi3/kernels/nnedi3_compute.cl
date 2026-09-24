/* ============================================================================
 * nnedi3_compute.cl — OpenCL port of the NNEDI3 predictor network in
 * upstream rigaya/NNEDI3 (submodule of AviSynthCUDAFilters @68aef6e),
 * NNEDI3/nnedi3/nnedi3_kernel.cu @01931aa (GPL upstream):
 *   kl_compute_nn (:331-475) -> kn_compute_nn
 *
 * Runs the real NNEDI3 network on exactly the pixels the prescreener
 * rejected: walks this group's work list (workNN/numblocks produced by
 * kn_prescreening), stages an xdia*ydia neighbourhood per work item row,
 * normalises it by the local mean/σ, evaluates NN neuron pairs (softmax
 * numerator via dev_expf + elliott-squashed value), and writes the
 * weighted average back.
 *
 * -- Template collapse ------------------------------------------------------
 * Upstream instantiates 70 kernels: QUAL in {1,2} x NN in {16,32,64,128,256}
 * x READ in {8x6,16x6,32x6,48x6,8x4,16x4,32x4} (LaunchComputeNN::Get, :547).
 * This port is ONE kernel with qual/nn/xdia/ydia as runtime arguments:
 *   - the loops over QUAL and NN/NN_BLOCK_W are plain runtime loops, and
 *   - all seven ReadPixelNxM policies were checked to produce the SAME
 *     logical tile: every one of them ends up with
 *         B[ty][k] == src[(k % xdia) + (k / xdia) * refpitch],  K = xdia*ydia
 *     (they differ only in which thread loads which element — 8x6/8x4 split
 *     tx into tx>>3 / tx&7, the wider ones stride by 16).  The port uses a
 *     single strided loader; each B element is still written exactly once
 *     from the same address, so the staged VALUES are identical.
 * The `__local` tile is sized for the largest policy (K <= 48*6 = 288).
 *
 * -- Geometry / contracts ---------------------------------------------------
 *   local size MUST be (16, 32, 1) = (NN_BLOCK_W, NN_BLOCK_H); enforced.
 *   Group grid MUST be the same (nblocks(width4,32), nblocks(height,16))
 *   grid kn_prescreening ran on — bid indexes its work list.
 *   Host preconditions (EvalCUDA:665): `ref` is offset to
 *   `frame_ref - (((ydia>>1)-1)*refpitch + ((xdia>>1)-1))` PIXELS, pitches
 *   in pixels; weights are `ws` = NN*K short2 followed by `wf` = float2
 *   pairs (`wf = &ws[NN*K]`), ws_pitch/wf_pitch in short2/float2 units
 *   (upstream: weights1pitch/2 and /4).  work_nn is 2 bytes per entry.
 *   nn must be a multiple of 16 and K <= 288.
 *
 * -- Bit-exactness ----------------------------------------------------------
 * The float reductions are NOT free-order sums: upstream reduces with
 * dev_reduce_warp<16> (ReduceKernel.cuh:40), a shuffle-down butterfly with
 * steps 8,4,2,1, so lane 0 receives a specific binary tree
 *   (((v0+v8)+(v4+v12)) + ((v2+v10)+(v6+v14))) + (...odd lanes...)
 * and float addition is not associative — a sequential sum would differ in
 * the last bits.  This port reproduces the same tree in __local (lanes
 * 8..15 read out of their 16-lane row upstream, but those partial values
 * never flow back into lane 0, so clamping the read is equivalent).
 * dev_expf is upstream's bit-twiddling exp approximation (:318) and is
 * transcribed verbatim — it must NOT be replaced by exp()/native_exp().
 * Everything else is unfused f32 in upstream's operation order; the host
 * build must disable FP contraction (same --fmad caveat as
 * avscuda_resample.cl).  Upstream's two reciprocals are written
 * `(float)(1.0 / (double)K)` and `(float)(1.0 / (double)QUAL)`; the port
 * uses `1.0f / (float)K` so it needs no fp64 support.  That is not a
 * blanket identity (double rounding), so it was checked exhaustively over
 * every reachable value — K in {32,48,64,96,128,192,288} (xdia in
 * {8,16,32,48} x ydia in {4,6}) and qual in {1,2} — where both spellings
 * give the identical float.  Integer accumulators (sum/sumsq and the neuron
 * dot products) are exact int32; upstream relies on the quantised weights
 * keeping them in range.
 * // ALG-VERIFIED via python/run_nnedi3_compute.py.
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

#define NN_BLOCK_W  16
#define NN_BLOCK_H  32
#define PRE_BLOCK_W 32
#define PRE_BLOCK_H 16
#define K_MAX       (48 * 6)

/* upstream dev_expf (:318): clamp to +-80, affine map, reinterpret bits */
static float nn_dev_expf(float f)
{
    const float exp_lo = -80.0f;
    const float exp_hi = +80.0f;
    const float e0_mult = 12102203.161561486f;
    const float e0_bias = 1064866805.0f;
    int i = (int)(fmax(fmin(f, exp_hi), exp_lo) * e0_mult + e0_bias);
    return as_float(i);
}

kernel __attribute__((reqd_work_group_size(NN_BLOCK_W, NN_BLOCK_H, 1)))
void kn_compute_nn(
    __global       PX*    __restrict dst,  int dst_pitch,
    __global const PX*    __restrict ref,  int ref_pitch,
    __global const uchar* __restrict work_nn,
    __global const int*   __restrict numblocks,
    __global const short* __restrict ws,   int ws_pitch,
    __global const float* __restrict wf,   int wf_pitch,
    int val_min, int val_max,
    int qual, int nn, int xdia, int ydia)
{
    int bid = get_group_id(0) + get_group_id(1) * get_num_groups(0);
    int workoff = bid * PRE_BLOCK_W * 4 * PRE_BLOCK_H;

    int xbase = get_group_id(0) * PRE_BLOCK_W * 4;
    int ybase = get_group_id(1) * PRE_BLOCK_H;
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    int K = xdia * ydia;

    __local PX    B[NN_BLOCK_H][K_MAX];
    __local float avg[NN_BLOCK_H];
    __local float var[NN_BLOCK_H];
    __local float invvar[NN_BLOCK_H];
    __local int   ibuf[NN_BLOCK_H][NN_BLOCK_W];
    __local float fbuf[NN_BLOCK_H][NN_BLOCK_W];

    int nb = numblocks[bid];

    for (int b = 0; b < nb; b += NN_BLOCK_H) {
        int x = xbase;
        int y = ybase;
        if (b + ty < nb) {
            x += (int)work_nn[(workoff + b + ty) * 2 + 0];
            y += (int)work_nn[(workoff + b + ty) * 2 + 1];
        }

        /* stage the xdia*ydia tile for this row (see template collapse) */
        __global const PX* src = ref + x + y * ref_pitch;
        for (int k = tx; k < K; k += NN_BLOCK_W) {
            B[ty][k] = src[(k % xdia) + (k / xdia) * ref_pitch];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        /* mean / variance over the tile */
        int sum = 0, sumsq = 0;
        for (int i = 0; i < K / NN_BLOCK_W; ++i) {
            int v = (int)B[ty][tx + i * NN_BLOCK_W];
            sum += v;
            sumsq += v * v;
        }
        ibuf[ty][tx] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int off = 8; off > 0; off >>= 1) {
            int add = (tx + off < NN_BLOCK_W) ? ibuf[ty][tx + off] : 0;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (tx + off < NN_BLOCK_W) ibuf[ty][tx] += add;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        sum = ibuf[ty][0];
        barrier(CLK_LOCAL_MEM_FENCE);
        ibuf[ty][tx] = sumsq;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int off = 8; off > 0; off >>= 1) {
            int add = (tx + off < NN_BLOCK_W) ? ibuf[ty][tx + off] : 0;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (tx + off < NN_BLOCK_W) ibuf[ty][tx] += add;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        sumsq = ibuf[ty][0];
        barrier(CLK_LOCAL_MEM_FENCE);

        if (tx == 0) {
            float scale = 1.0f / (float)K;  /* == (float)(1.0/(double)K) */
            float avg_ = sum * scale;
            float var_ = sumsq * scale - avg_ * avg_;
            float invvar_;
            if (var_ <= FLT_EPSILON) {
                var_ = 0.0f;
                invvar_ = 0.0f;
            } else {
                var_ = sqrt(var_);
                invvar_ = 1.0f / var_;
            }
            avg[ty] = avg_;
            var[ty] = var_;
            invvar[ty] = invvar_;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float result = 0.0f;
        for (int q = 0; q < qual; ++q) {
            float vsum = 0.0f, wsum = 0.0f;
            for (int i = 0; i < nn / NN_BLOCK_W; ++i) {
                int j = i * NN_BLOCK_W + tx;

                int sx = 0, sy = 0;
                for (int k = 0; k < K; ++k) {
                    int v = (int)B[ty][k];
                    int wi = (j + k * nn + q * ws_pitch) * 2;
                    sx += v * (int)ws[wi + 0];
                    sy += v * (int)ws[wi + 1];
                }

                int f1 = (j + q * wf_pitch) * 2;
                int f2 = (j + nn + q * wf_pitch) * 2;
                float res0 = (float)sx * wf[f1 + 0] * invvar[ty] + wf[f2 + 0];
                float res1 = (float)sy * wf[f1 + 1] * invvar[ty] + wf[f2 + 1];

                res0 = nn_dev_expf(res0);

                vsum += res0 * (res1 / (1.0f + fabs(res1)));
                wsum += res0;
            }

            /* float butterfly reduce (order-significant; see header) */
            fbuf[ty][tx] = vsum;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int off = 8; off > 0; off >>= 1) {
                float add = (tx + off < NN_BLOCK_W) ? fbuf[ty][tx + off] : 0.0f;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (tx + off < NN_BLOCK_W) fbuf[ty][tx] = fbuf[ty][tx] + add;
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            vsum = fbuf[ty][0];
            barrier(CLK_LOCAL_MEM_FENCE);
            fbuf[ty][tx] = wsum;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int off = 8; off > 0; off >>= 1) {
                float add = (tx + off < NN_BLOCK_W) ? fbuf[ty][tx + off] : 0.0f;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (tx + off < NN_BLOCK_W) fbuf[ty][tx] = fbuf[ty][tx] + add;
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            wsum = fbuf[ty][0];
            barrier(CLK_LOCAL_MEM_FENCE);

            if (tx == 0) {
                const float min_weight_sum = 1e-10f;
                if (wsum > min_weight_sum)
                    result += ((5.0f * vsum) / wsum) * var[ty] + avg[ty];
                else
                    result += avg[ty];
            }
        }

        if (tx == 0 && b + ty < nb) {
            float scale = 1.0f / (float)qual;  /* == (float)(1.0/(double)qual) */
            int v = (int)(result * scale + 0.5f);
            dst[x + y * dst_pitch] = (PX)min(max(v, val_min), val_max);
        }

        barrier(CLK_LOCAL_MEM_FENCE); /* guard shared memory */
    }
}
