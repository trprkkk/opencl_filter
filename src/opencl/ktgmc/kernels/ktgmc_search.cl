/* ============================================================================
 * ktgmc_search.cl — the KTGMC block-search kernel (kl_search),
 * transliterated from AviSynthCUDAFilters/KTGMC/MVKernel.cu @ 68aef6e
 * (KVersion 0.7.5), verbatim source of record
 * cuda_search/verbatim_search.inc (MVKernel.cu:269..1365, generated).
 *
 * ONLY the shipped CPU_EMU=true semantics are implemented: every reduction is
 * the deterministic serial scan (earlier index wins ties), and the predictor
 * slots follow the CPU_EMU placement (median candidate at pred[3] via the
 * dst-remap).  The warp-shuffle CPU_EMU=false path is not shipped upstream
 * and is out of scope (SEARCH_MEASURE_REPORT.md §5).
 *
 * Host mapping (settles docs/BLOCKSEARCH_MODEL.md §8b.2 — see
 * SEARCH_HOST_CONTRACT.md §4): upstream launches kl_search as
 * dim3(batch, min(nBlkX,nBlkY)) CTAs that work-steal columns through
 * atomicAdd(next) and spin-wait on prog[] for the ANALYZE_SYNC=1 left-column
 * dependency.  Stage 1 measured (SEARCH_MEASURE_REPORT.md §3.4) that prog/
 * next affect SCHEDULING ONLY — results are bit-exactly reproduced by a
 * serial evaluation in a host-chosen order in which the left/up predictors
 * read FRESH own-region values.  This port therefore drops prog/next
 * entirely and uses HOST-SIDE ORDERED DISPATCH:
 *
 *   for each batch (independent), for blkx = 0..nBlkX-1 (ascending):
 *       enqueue kt_search_column (one work-item) for that (batch, blkx)
 *
 * on an in-order command queue.  The single work-item walks blky = 0..nBlkY-1
 * serially inside the kernel, so the left predictor (previous column, same
 * row — finished by the previous enqueue) and the up predictor (same column,
 * previous row — finished earlier in this loop) both read fresh own-region
 * values, exactly the ANALYZE_SYNC=1 dependency.  Batches are independent
 * (disjoint vector rows + planes) and could be dispatched concurrently.
 *
 * Vectors layout (SEARCH_HOST_CONTRACT.md §1): one allocation per batch,
 * vectorsPitch = 4 + 2*nBlkCount short2/int2 entries, addressed relative to
 * the row base at offset +4:
 *   slot -2 = zero-MV sentinel (alloc[2]), slot -1 = global-MV sentinel
 *   (alloc[3]), slots 0..n-1 = own region (per-block results, written here),
 *   slots n..2n-1 = copy region (pre-search snapshot from kt_prepare_search,
 *   read ONLY by the bottom-right predictor of interior blocks).
 * The kernel takes the ALLOCATION base (vec_alloc), not the row base, so
 * every predictor index (including the -2/-1 sentinels) resolves to a
 * non-negative element of the passed buffer — no negative-offset sub-buffer
 * is needed on hosts/drivers where those do not work.
 *
 * REF_VECTOR_INDEX semantics (measured, SEARCH_HOST_CONTRACT.md §2):
 *   data[4]=zero(-2), data[5]=global(-1), data[6]=own blkIdx (incoming),
 *   data[7]=left blkIdx-1 (own region FRESH; ANALYZE_SYNC=1),
 *   data[8]=up blkIdx-nBlkX (own region FRESH; = left slot when by==0),
 *   data[9]=bottom-right blkIdx+nBlkX+1+nBlkCount (copy region, STALE).
 *
 * Refinement semantics (verbatim_search.inc:244-330, 357-440, 811-838):
 *   EXHAUSTIVE (SEARCH=1): snapshot the entry best (bmx,bmy), then
 *     expanding_1 AND expanding_2 both centre on the SAME pre-expand1 point
 *     (expand2 does NOT chain off expand1's result).
 *   HEX (SEARCH=2): hex2 first, then expanding_1 CHAINED off the post-hex2
 *     best.
 *   Each stage: a candidate is pruned iff its lambda-only cost is
 *     >= the stage-ENTRY best (the callee's local writes do not affect the
 *     prune); survivors get cost += sad + ((sad*PENALTY_NEW)>>8); pruned
 *     candidates keep LARGE_COST (INT_MAX); the stage update is CONDITIONAL
 *     (`if (result[0].cost < best.cost) best = result[0]`), so a stage whose
 *     every candidate was pruned leaves the entry best untouched.
 *   Serial reduce: for i=1..n MinCost(best, i) — earlier index wins ties.
 *
 * The per-block SAD is the scalar double loop (identical integer sums; the
 * CUDA __vabsdiff4/__sad packed loads are load optimizations only — same
 * argument as kt_calc_all_sad).  Cost model, penalties (bx<3 REPLACES the
 * cost with sad + (sad*PENALTIES[bx])>>8; bx>=3 ADDS sad), clipping and the
 * cost init (LAMBDA*sq_norm)>>8 are transliterated exactly.
 *
 * Status:
 *   // ALG-VERIFIED (2026-10-28): CPU mirror sim/ktgmc_search_ref.cpp +
 *   // independent Python golden python/run_mv_search.py (the corrected
 *   // scalar model of cuda_search/cuda_search_host.cu, itself validated
 *   // bit-exact vs the CUDA twin on 9 configs — run_s2.txt mism=0).
 *   // Device run: rig/dev_ktgmc_search.py (OpenCL vs golden E2E, both PX,
 *   // plus CUDA-twin dump comparison on the 8 reproduced configs).
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

/* LARGE_COST = INT_MAX (verbatim_search.inc:38). */
#define KS_LARGE_COST 2147483647

/* ---------------------------------------------------------------------------
 * Pure helpers (dev_clip_mv / dev_check_mv / dev_sq_norm / dev_get_ref_block,
 * verbatim_search.inc:45-79).  Same arithmetic as the kt_* helpers in
 * ktgmc_motion.cl; duplicated here because the two files build as separate
 * OpenCL programs (no cross-file linkage).
 * -------------------------------------------------------------------------*/

static void ks_clip_mv(int* x, int* y, __global const int* rect)
{
    *x = (*x > rect[0]) ? rect[0] : ((*x < rect[2]) ? rect[2] : *x);
    *y = (*y > rect[1]) ? rect[1] : ((*y < rect[3]) ? rect[3] : *y);
}

static int ks_check_mv(int x, int y, __global const int* rect)
{
    return (x <= rect[0]) & (y <= rect[1]) & (x >= rect[2]) & (y >= rect[3]);
}

static int ks_sq_norm(int ax, int ay, int bx, int by)
{
    return (ax - bx) * (ax - bx) + (ay - by) * (ay - by);
}

/* median3 as spelled upstream (verbatim_search.inc:719):
 * min(max(min(a,b),c), max(a,b)). */
static int ks_median3(int a, int b, int c)
{
    return min(max(min(a, b), c), max(a, b));
}

/* dev_get_ref_block as an element offset (identical to kt_ref_block_offset
 * in ktgmc_motion.cl).  NPEL in {1,2} are the reachable instantiations; the
 * NPEL==4 branch is kept for the shared-helper equivalence. */
static int ks_ref_offset(int vx, int vy, int nPitch, int nImgPitch, int NPEL)
{
    if (NPEL == 1) {
        return vx + vy * nPitch;
    }
    if (NPEL == 2) {
        int sx = vx & 1;
        int sy = vy & 1;
        int si = sx + sy * 2;
        return (vx >> 1) + (vy >> 1) * nPitch + si * nImgPitch;
    }
    /* NPEL == 4 (unreachable through the upstream dispatcher) */
    {
        int sx = vx & 3;
        int sy = vy & 3;
        int si = sx + sy * 4;
        return (vx >> 2) + (vy >> 2) * nPitch + si * nImgPitch;
    }
}

/* ---------------------------------------------------------------------------
 * Refinement candidate patterns (c_expanding_search_1_area /
 * c_expanding_search_2_area / c_hex2_search_1_area, verbatim_search.inc
 * 230-240 / 333-353 / 432-434).  ORDER IS SEMANTIC: the serial reduce keeps
 * the earlier index on ties.
 * -------------------------------------------------------------------------*/
__constant int ks_e1x[8] = { 0, 0, -1, 1, -1, -1, 1, 1 };
__constant int ks_e1y[8] = { -1, 1, 0, 0, -1, 1, -1, 1 };
__constant int ks_e2x[16] = { -1, -1, 0, 0, 1, 1, -2, 2, -2, 2, -2, 2, -2, -2, 2, 2 };
__constant int ks_e2y[16] = { -2, 2, -2, 2, -2, 2, -1, -1, 0, 0, 1, 1, -2, 2, -2, 2 };
__constant int ks_hx[6]  = { -2, -1, 1, 2, 1, -1 };
__constant int ks_hy[6]  = { 0, 2, 2, 0, -2, -2 };

/* ---------------------------------------------------------------------------
 * Block SAD: src from the BLK_SIZE-strided local tile (dev_read_pixels
 * staging), ref from the plane at block origin + roff (dev_calc_sad).
 * Scalar absolute-difference sums — integer addition is order-independent,
 * so this reproduces the CUDA per-thread split + reduce exactly.
 * -------------------------------------------------------------------------*/
static int ks_tile_sad(
    __local const PX* srcY, __local const PX* srcU, __local const PX* srcV,
    __global const PX* refY, __global const PX* refU, __global const PX* refV,
    int roffY, int roffUV, int BLK, int chroma, int nPitchY, int nPitchUV)
{
    int sad = 0;
    for (int jy = 0; jy < BLK; jy++) {
        for (int jx = 0; jx < BLK; jx++) {
            int a = (int)srcY[jx + jy * BLK];
            int b = (int)refY[roffY + jx + jy * nPitchY];
            int d = a - b; if (d < 0) d = -d;
            sad += d;
        }
    }
    if (chroma) {
        int bs2 = BLK >> 1;
        for (int jy = 0; jy < bs2; jy++) {
            for (int jx = 0; jx < bs2; jx++) {
                int au = (int)srcU[jx + jy * bs2];
                int bu = (int)refU[roffUV + jx + jy * nPitchUV];
                int d = au - bu; if (d < 0) d = -d;
                sad += d;
                int av = (int)srcV[jx + jy * bs2];
                int bv = (int)refV[roffUV + jx + jy * nPitchUV];
                d = av - bv; if (d < 0) d = -d;
                sad += d;
            }
        }
    }
    return sad;
}

/* ---------------------------------------------------------------------------
 * One refinement stage (dev_expanding_search_1/2, dev_hex2_search_1 under
 * CPU_EMU=true).  (cx,cy) is the stage centre; (bcost,bx,by) is the stage-
 * ENTRY best (read for pruning, conditionally updated at the very end —
 * verbatim_search.inc:323/423/496).
 * -------------------------------------------------------------------------*/
static void ks_refine_stage(
    __global const int* data, __global const int* dataf,
    int cx, int cy,
    __constant const int* adx, __constant const int* ady, int nCand,
    int* bcost, int* bx, int* by,
    int NPEL, int BLK, int chroma,
    int nPitchY, int nPitchUV, int nImgPitchY, int nImgPitchUV,
    __local const PX* srcY, __local const PX* srcU, __local const PX* srcV,
    __global const PX* refBY, __global const PX* refBU, __global const PX* refBV)
{
    int cost[16];
    int rx[16];
    int ry[16];

    for (int i = 0; i < nCand; i++) {
        int x = cx + adx[i];
        int y = cy + ady[i];
        int ok = ks_check_mv(x, y, data);
        int c = (dataf[4] * ks_sq_norm(x, y, data[10], data[11])) >> 8;
        /* no additional SAD calculations if partial sum is already above
         * the entry minCost (prune compares the ENTRY best) */
        if (c >= *bcost) {
            ok = 0;
        }
        rx[i] = x;
        ry[i] = y;
        cost[i] = ok ? c : KS_LARGE_COST;
    }

    for (int i = 0; i < nCand; i++) {
        if (cost[i] == KS_LARGE_COST)
            continue;                    /* pruned: keeps LARGE_COST */
        int roffY = ks_ref_offset(rx[i], ry[i], nPitchY, nImgPitchY, NPEL);
        int roffUV = chroma
            ? ks_ref_offset(rx[i] >> 1, ry[i] >> 1, nPitchUV, nImgPitchUV, NPEL)
            : 0;
        int sad = ks_tile_sad(srcY, srcU, srcV, refBY, refBU, refBV,
                              roffY, roffUV, BLK, chroma, nPitchY, nPitchUV);
        cost[i] += sad + ((sad * dataf[3]) >> 8);   /* SAD + penaltyNew */
    }

    /* serial reduce (dev_reduce_result<LEN, CPU_EMU=true>): earlier index
     * wins ties */
    for (int i = 1; i < nCand; i++) {
        if (cost[0] > cost[i]) {
            cost[0] = cost[i];
            rx[0] = rx[i];
            ry[0] = ry[i];
        }
    }

    /* conditional update: the stage keeps the entry best unless its own
     * local minimum is strictly cheaper (an all-pruned stage leaves the
     * entry best untouched — its local min is LARGE_COST) */
    if (cost[0] < *bcost) {
        *bcost = cost[0];
        *bx = rx[0];
        *by = ry[0];
    }
}

/* ---------------------------------------------------------------------------
 * kl_search — per (batch, column) scalar transcription.  One work-item; the
 * host dispatches columns in ascending order per batch on an in-order queue
 * (see file header).  Reads the per-block SearchBlock fields from the flat
 * block_data (stride 12) / block_dataf (stride 5) arrays produced by
 * kt_prepare_search, and reads/writes the batch's vectors row.
 * -------------------------------------------------------------------------*/
kernel void kt_search_column(
    __global const PX* __restrict pSrcY,
    __global const PX* __restrict pSrcU,
    __global const PX* __restrict pSrcV,
    __global const PX* __restrict pRefY,
    __global const PX* __restrict pRefU,
    __global const PX* __restrict pRefV,
    __global int2* __restrict vec_alloc,       /* vectors allocation base */
    __global const int* __restrict block_data, /* nBlkCount * 12 ints */
    __global const int* __restrict block_dataf,/* nBlkCount * 5 ints  */
    int blkx,              /* this invocation's column */
    int nBlkX, int nBlkY, int nPad,
    int BLK_SIZE, int SEARCH, int NPEL, int chroma,
    int nPitchY, int nPitchUV, int nImgPitchY, int nImgPitchUV)
{
    int blkStep = BLK_SIZE >> 1;
    /* row base = allocation + N_CONST_VEC(4); slot i (in [-2, 2n)) is then a
     * non-negative element of vec_alloc for every reachable i */
    __global int2* vrow = vec_alloc + 4;

    /* local src tiles (max BLK_SIZE 32 luma, 16 chroma) */
    __local PX srcY[32 * 32];
    __local PX srcU[16 * 16];
    __local PX srcV[16 * 16];

    for (int blky = 0; blky < nBlkY; blky++) {
        int blkIdx = blky * nBlkX + blkx;
        int offx = nPad + blkx * blkStep;
        int offy = nPad + blky * blkStep;

        __global const int* data  = block_data  + blkIdx * 12;
        __global const int* dataf = block_dataf + blkIdx * 5;

        /* dev_read_pixels staging: BLK_SIZE x BLK_SIZE luma tile,
         * BLK_SIZE/2 x BLK_SIZE/2 U and V tiles */
        for (int jy = 0; jy < BLK_SIZE; jy++)
            for (int jx = 0; jx < BLK_SIZE; jx++)
                srcY[jx + jy * BLK_SIZE] =
                    pSrcY[(offx + jx) + (offy + jy) * nPitchY];
        if (chroma) {
            int b = blkStep;
            int bx0 = offx >> 1;
            int by0 = offy >> 1;
            for (int jy = 0; jy < b; jy++)
                for (int jx = 0; jx < b; jx++) {
                    srcU[jx + jy * b] = pSrcU[(bx0 + jx) + (by0 + jy) * nPitchUV];
                    srcV[jx + jy * b] = pSrcV[(bx0 + jx) + (by0 + jy) * nPitchUV];
                }
        }

        __global const PX* refBY = pRefY + offx + offy * nPitchY;
        __global const PX* refBU = pRefU + (offx >> 1) + (offy >> 1) * nPitchUV;
        __global const PX* refBV = pRefV + (offx >> 1) + (offy >> 1) * nPitchUV;

        /* FetchPredictors (verbatim_search.inc:700-722): 6 raw slots, clipped,
         * CPU_EMU dst-remap (tx>=3 -> tx+1), then median3 of slots 4..6 into
         * slot 3.  7 candidates: 0=zero, 1=global, 2=own, 3=median, 4=left,
         * 5=up, 6=bottom-right. */
        int predX[7];
        int predY[7];
        for (int t = 0; t < 6; t++) {
            int2 vec = vrow[data[4 + t]];
            int x = vec.x;
            int y = vec.y;
            ks_clip_mv(&x, &y, data);
            int dst = (t < 3) ? t : t + 1;   /* CPU_EMU slot remap */
            predX[dst] = x;
            predY[dst] = y;
        }
        predX[3] = ks_median3(predX[4], predX[5], predX[6]);
        predY[3] = ks_median3(predY[4], predY[5], predY[6]);

        /* cost init (line 727) + 7 candidate SADs with the penalty scheme
         * (lines 744-770): bx<3 REPLACES the cost (penaltyZero/global/none),
         * bx>=3 ADDS the SAD to the lambda cost. */
        int cost[8];
        int rx[8];
        int ry[8];
        for (int i = 0; i < 7; i++) {
            rx[i] = predX[i];
            ry[i] = predY[i];
            cost[i] = (dataf[4] * ks_sq_norm(predX[i], predY[i],
                                             data[10], data[11])) >> 8;
        }
        for (int i = 0; i < 7; i++) {
            int roffY = ks_ref_offset(rx[i], ry[i], nPitchY, nImgPitchY, NPEL);
            int roffUV = chroma
                ? ks_ref_offset(rx[i] >> 1, ry[i] >> 1, nPitchUV, nImgPitchUV, NPEL)
                : 0;
            int sad = ks_tile_sad(srcY, srcU, srcV, refBY, refBU, refBV,
                                  roffY, roffUV, BLK_SIZE, chroma,
                                  nPitchY, nPitchUV);
            if (i < 3)
                cost[i] = sad + ((sad * dataf[i]) >> 8);
            else
                cost[i] += sad;
        }

        /* serial reduce over the 7 candidates (earlier index wins) */
        for (int i = 1; i < 7; i++) {
            if (cost[0] > cost[i]) {
                cost[0] = cost[i];
                rx[0] = rx[i];
                ry[0] = ry[i];
            }
        }

        /* Refinement (lines 811-838): EXHAUSTIVE snapshots the centre before
         * expanding_1 and expanding_2 re-uses the SAME pre-expand1 centre;
         * HEX chains hex2 -> expanding_1 off the running best. */
        if (SEARCH == 1) {
            int bmx = rx[0];
            int bmy = ry[0];
            ks_refine_stage(data, dataf, bmx, bmy, ks_e1x, ks_e1y, 8,
                            &cost[0], &rx[0], &ry[0],
                            NPEL, BLK_SIZE, chroma,
                            nPitchY, nPitchUV, nImgPitchY, nImgPitchUV,
                            srcY, srcU, srcV, refBY, refBU, refBV);
            ks_refine_stage(data, dataf, bmx, bmy, ks_e2x, ks_e2y, 16,
                            &cost[0], &rx[0], &ry[0],
                            NPEL, BLK_SIZE, chroma,
                            nPitchY, nPitchUV, nImgPitchY, nImgPitchUV,
                            srcY, srcU, srcV, refBY, refBU, refBV);
        } else { /* SEARCH == 2 (HEX2SEARCH) */
            ks_refine_stage(data, dataf, rx[0], ry[0], ks_hx, ks_hy, 6,
                            &cost[0], &rx[0], &ry[0],
                            NPEL, BLK_SIZE, chroma,
                            nPitchY, nPitchUV, nImgPitchY, nImgPitchUV,
                            srcY, srcU, srcV, refBY, refBU, refBV);
            ks_refine_stage(data, dataf, rx[0], ry[0], ks_e1x, ks_e1y, 8,
                            &cost[0], &rx[0], &ry[0],
                            NPEL, BLK_SIZE, chroma,
                            nPitchY, nPitchUV, nImgPitchY, nImgPitchUV,
                            srcY, srcU, srcV, refBY, refBU, refBV);
        }

        /* writeback (line 839): final MV into the own region slot.  The
         * CUDA __threadfence + atomicExch(&prog[blkx], blky) publish is
         * replaced by the host's ordered dispatch (file header). */
        int2 best = (int2)(rx[0], ry[0]);
        vrow[blkIdx] = best;
    }
}
