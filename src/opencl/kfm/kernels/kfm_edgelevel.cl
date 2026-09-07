/* ============================================================================
 * kfm_edgelevel.cl — OpenCL port of the KFM `KEdgeLevel` filter kernels.
 *
 * Faithful port of rigaya/AviSynthCUDAFilters -> KFM/KDeband.cu (KFM is MIT).
 * KEdgeLevel enhances/visualises edges; it is composed of four device kernels,
 * each transliterated here from the corresponding CUDA __global__ kernel.  The
 * CPU twin of each (cpu_edgelevel, cpu_edgelevel_repair, cpu_el_to444,
 * cpu_el_from444) is the reference the mirrors below cross-check against.
 *
 * Status legend:
 *   // ALG-VERIFIED : arithmetic cross-checked (make test) against a CPU mirror
 *                    + independent Python golden.
 *
 * NOTE on floats: the edge factor / enhancement is computed in 32-bit float
 * (same as CUDA/CPU, both use IEEE float32 with no FMA contraction).  The
 * python golden emulates float32 per-operation so it matches the C mirror
 * bit-for-bit.
 *
 * NOTE on the 4:2:x host seam: in the real KEdgeLevel uv path the host converts
 * subsampled U/V to 4:4:4 (el_to444) and back (el_from444) with sizes chosen by
 * the AviSynth host; the kernels themselves take plain (width,height) source
 * dims + a dst pitch that already accounts for the BW/BH up/down-scaling, and
 * only read in-bounds (border-clamped).  Which width/height the host passes for
 * a given subsampling is a host concern; the kernels below are verified on a
 * self-consistent (width,height = source dims) configuration.
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

/* EdgeLevel marker constants (KDeband.cu) and their maxv-scaled helper:
 *   SCALE(c) = (int)(((float)c / 255.0f) * maxv)
 * Used only in "check" (visualise) mode. */
static int kf_el_scale(int c, int maxv)
{
    return (int)((((float)c) / 255.0f) * (float)maxv);
}

/* ---------------------------------------------------------------------------
 * kf_edgelevel — the edge detection / enhancement core (cpu_edgelevel /
 * kl_edgelevel twin).  Per output pixel it scans a horizontal and a vertical
 * window around (x,y) of half-width (2+selective), records the min/max spread
 * (preferring the axis with the larger spread) and, in selective mode, the max
 * consecutive gradient; derives rdiff and a band-pass factor; then either
 * visualises the edge (check) or pulls the pixel toward the local average by
 * str (enhance), with optional UV processing.
 *
 * Template booleans of the CUDA original (check, selective, uv) are passed as
 * runtime ints here (code-gen only in CUDA).  Grid: 2D (width, height).
 * The border ring [<=1+selective pixels in] is copied through (or, in check
 * mode, set to the NONE marker) — so no out-of-range access ever happens and
 * the whole plane is verifiable.
 * // ALG-VERIFIED (python/run_kfm_edgelevel.py, all 8 check/selective/uv combos,
 * // 8/16-bit, str/thrs sweeps)
 * -------------------------------------------------------------------------*/
kernel void kf_edgelevel(
    __global       PX* __restrict dstY,
    __global       PX* __restrict dstU,
    __global       PX* __restrict dstV,
    __global const PX* __restrict srcY,
    __global const PX* __restrict srcU,
    __global const PX* __restrict srcV,
    int width, int height, int pitch, int maxv,
    int str, int strUV, int thrs,
    int check, int selective, int uv)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    int S = selective ? 1 : 0;

    int offset = y * pitch + x;
    __global const PX* sy = srcY + offset;

    if (y <= (1 + S) || y >= height - (2 + S) ||
        x <= (1 + S) || x >= width - (2 + S)) {
        if (x < width && y < height) {
            dstY[offset] = check ? (PX)kf_el_scale(16, maxv) : sy[0];
            if (uv) {
                dstU[offset] = srcU[offset];
                dstV[offset] = srcV[offset];
            }
        }
        return;
    }

    /* interior */
    int hmax, hmin, vmax, vmin;
    int hdiffmax = 0, vdiffmax = 0;
    int hprev = hmax = hmin = (int)sy[-(2 + S)];
    int vprev = vmax = vmin = (int)sy[-(2 + S) * pitch];

    for (int i = -(1 + S); i < (3 + S); ++i) {
        int hcur = (int)sy[i];
        int vcur = (int)sy[i * pitch];

        if (hcur > hmax) hmax = hcur;
        if (hcur < hmin) hmin = hcur;
        if (vcur > vmax) vmax = vcur;
        if (vcur < vmin) vmin = vcur;

        if (selective) {
            int dh = hcur - hprev; if (dh < 0) dh = -dh;
            int dv = vcur - vprev; if (dv < 0) dv = -dv;
            if (dh > hdiffmax) hdiffmax = dh;
            if (dv > vdiffmax) vdiffmax = dv;
        }
        hprev = hcur;
        vprev = vcur;
    }

    if (hmax - hmin < vmax - vmin) {
        hmax = vmax;
        hmin = vmin;
    }
    if (vdiffmax > hdiffmax) hdiffmax = vdiffmax;

    /* float32 band-pass "factor" (selective only; else exactly 1.0f) */
    float factor = 1.0f;
    if (selective) {
        float rdiff = (float)hdiffmax / (float)(hmax - hmin);
        float a = (0.55f - rdiff) * 10.0f;  if (a < 0.0f) a = 0.0f; else if (a > 1.0f) a = 1.0f;
        float b = (0.35f - rdiff) * 10.0f;  if (b < 0.0f) b = 0.0f; else if (b > 1.0f) b = 1.0f;
        factor = a - b;
    }

    int srcvY = (int)sy[0];
    int dstvY, dstvU, dstvV;

    if (check) {
        if (hmax - hmin > thrs && factor > 0.0f) {
            int avgY = (hmax + hmin) >> 1;
            if (srcvY > avgY)
                dstvY = (factor == 1.0f) ? kf_el_scale(50, maxv) : kf_el_scale(120, maxv);
            else
                dstvY = (factor == 1.0f) ? kf_el_scale(240, maxv) : kf_el_scale(180, maxv);
        } else {
            dstvY = kf_el_scale(16, maxv);
        }
        dstvU = (int)srcU[offset];
        dstvV = (int)srcV[offset];
    } else {
        if (hmax - hmin > thrs && factor > 0.0f) {
            float factorY = ((float)str * factor) * 0.0625f;
            int avgY = (hmax + hmin) >> 1;
            int v = srcvY + (int)(((float)(srcvY - avgY)) * factorY);
            if (v < hmin) v = hmin; else if (v > hmax) v = hmax;
            if (v < 0) v = 0; else if (v > maxv) v = maxv;
            dstvY = v;

            if (uv) {
                /* U: local min/max over the horizontal+vertical window */
                int uoff = offset;
                int Uhmax, Uhmin, Uvmax, Uvmin;
                Uhmax = Uhmin = (int)srcU[uoff - (2 + S)];
                Uvmax = Uvmin = (int)srcU[uoff - (2 + S) * pitch];
                for (int i = -(1 + S); i < (3 + S); ++i) {
                    int hc = (int)srcU[uoff + i];
                    int vc = (int)srcU[uoff + i * pitch];
                    if (hc > Uhmax) Uhmax = hc; if (hc < Uhmin) Uhmin = hc;
                    if (vc > Uvmax) Uvmax = vc; if (vc < Uvmin) Uvmin = vc;
                }
                if (Uhmax - Uhmin < Uvmax - Uvmin) { Uhmax = Uvmax; Uhmin = Uvmin; }
                float factorUV = (float)strUV * 0.0625f;
                int avgU = (Uhmax + Uhmin) >> 1;
                int svU = (int)srcU[uoff];
                int uout = svU + (int)(((float)(svU - avgU)) * factorUV);
                if (uout < Uhmin) uout = Uhmin; else if (uout > Uhmax) uout = Uhmax;
                if (uout < 0) uout = 0; else if (uout > maxv) uout = maxv;
                dstvU = uout;

                int Vhmax, Vhmin, Vvmax, Vvmin;
                Vhmax = Vhmin = (int)srcV[uoff - (2 + S)];
                Vvmax = Vvmin = (int)srcV[uoff - (2 + S) * pitch];
                for (int i = -(1 + S); i < (3 + S); ++i) {
                    int hc = (int)srcV[uoff + i];
                    int vc = (int)srcV[uoff + i * pitch];
                    if (hc > Vhmax) Vhmax = hc; if (hc < Vhmin) Vhmin = hc;
                    if (vc > Vvmax) Vvmax = vc; if (vc < Vvmin) Vvmin = vc;
                }
                if (Vhmax - Vhmin < Vvmax - Vvmin) { Vhmax = Vvmax; Vhmin = Vvmin; }
                int avgV = (Vhmax + Vhmin) >> 1;
                int svV = (int)srcV[uoff];
                int vout = svV + (int)(((float)(svV - avgV)) * factorUV);
                if (vout < Vhmin) vout = Vhmin; else if (vout > Vhmax) vout = Vhmax;
                if (vout < 0) vout = 0; else if (vout > maxv) vout = maxv;
                dstvV = vout;
            }
        } else {
            dstvY = srcvY;
            dstvU = (int)srcU[offset];
            dstvV = (int)srcV[offset];
        }
    }

    dstY[offset] = (PX)dstvY;
    if (uv) {
        dstU[offset] = (PX)dstvU;
        dstV[offset] = (PX)dstvV;
    }
}

/* ---------------------------------------------------------------------------
 * kf_edgelevel_repair — limit the edgelevel result el against the 8 neighbours
 * of src (cpu_edgelevel_repair / kl_edgelevel_repair, N=3 used upstream).
 * Per output pixel it sorts the 8 neighbour values, then clamps el to a range
 * that is min/max of srcv plus the N-th outer/inner sorted neighbour (N is a
 * runtime int, 1..4).  The CUDA/CPU originals read the full 3x3 window with no
 * border guard (borders rely on the padded AviSynth plane), so this kernel
 * likewise reads the window and is verified over the interior (1..width-2,
 * 1..height-2); border pixels need a padded source on the rig.
 * // ALG-VERIFIED over the interior (python/run_kfm_edgelevel.py)
 * -------------------------------------------------------------------------*/
static void kf_sort8(int* a)
{
    /* Batcher odd-even mergesort network, 19 comparators (dev_sort_8elem). */
    int t;
#define CS(p,q) do{ if(a[p]>a[q]){ t=a[p]; a[p]=a[q]; a[q]=t; } }while(0)
    CS(0,1); CS(2,3); CS(4,5); CS(6,7);
    CS(0,2); CS(1,3); CS(4,6); CS(5,7);
    CS(1,2); CS(5,6);
    CS(0,4); CS(1,5); CS(2,6); CS(3,7);
    CS(2,4); CS(3,5);
    CS(1,2); CS(3,4); CS(5,6);
#undef CS
}

kernel void kf_edgelevel_repair(
    __global       PX* __restrict dst,
    __global const PX* __restrict el,
    __global const PX* __restrict src,
    int width, int height, int pitch, int N)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int srcv = (int)src[x + y * pitch];
        int elv  = (int)el[x + y * pitch];
        int dstv = srcv;
        if (elv != srcv) {
            int a[8];
            a[0] = (int)src[(x - 1) + (y - 1) * pitch];
            a[1] = (int)src[(x    ) + (y - 1) * pitch];
            a[2] = (int)src[(x + 1) + (y - 1) * pitch];
            a[3] = (int)src[(x - 1) + (y    ) * pitch];
            a[4] = (int)src[(x + 1) + (y    ) * pitch];
            a[5] = (int)src[(x - 1) + (y + 1) * pitch];
            a[6] = (int)src[(x    ) + (y + 1) * pitch];
            a[7] = (int)src[(x + 1) + (y + 1) * pitch];
            kf_sort8(a);
            int lo = srcv < a[N - 1] ? srcv : a[N - 1];
            int hi = srcv > a[8 - N] ? srcv : a[8 - N];
            /* clamp(elv, min(srcv,a[N-1]), max(srcv,a[8-N])) */
            if (elv < lo) dstv = lo; else if (elv > hi) dstv = hi; else dstv = elv;
        }
        dst[x + y * pitch] = (PX)dstv;
    }
}

/* ---------------------------------------------------------------------------
 * kf_el_to444 / kf_el_from444 — 4:2:x <-> 4:4:4 chroma conversion helpers used
 * by KEdgeLevel's uv path (kl_el_to444 / kl_el_from444 twins).  to444 samples a
 * (width x height) source plane to a (BW*width x BH*height) dst (BW=1<<logUVx,
 * BH=1<<logUVy), bilinear across the chroma samples with a border clamp in the
 * CUDA kernel; from444 is a plain nearest gather at (BW*x, BH*y).
 * Grid: 2D (width, height) = source dims for to444, dst dims for from444.
 * // ALG-VERIFIED on the self-consistent (source-dims) configuration; the
 * // host's actual subsampling size wiring is a seam (see header note).
 * -------------------------------------------------------------------------*/
kernel void kf_el_to444(
    __global       PX* __restrict dst,
    __global const PX* __restrict src,
    int width, int height, int dst_pitch, int src_pitch,
    int logUVx, int logUVy)
{
    int BW = 1 << logUVx;
    int BH = 1 << logUVy;
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int v00 = (int)src[(x + 0) + (y + 0) * src_pitch];
        dst[BW * x + 0 + (BH * y + 0) * dst_pitch] = (PX)v00;
        if (logUVx) {
            int v10 = (x + 1 < width) ? (int)src[(x + 1) + (y + 0) * src_pitch] : v00;
            dst[BW * x + 1 + (BH * y + 0) * dst_pitch] = (PX)((v00 + v10 + 1) >> 1);
            if (logUVy) {
                int v01, v11;
                if (y + 1 < height) {
                    v01 = (int)src[(x + 0) + (y + 1) * src_pitch];
                    v11 = (x + 1 < width) ? (int)src[(x + 1) + (y + 1) * src_pitch] : v01;
                } else {
                    v01 = v00;
                    v11 = (x + 1 < width) ? v10 : v00;
                }
                dst[BW * x + 0 + (BH * y + 1) * dst_pitch] = (PX)((v00 + v01 + 1) >> 1);
                dst[BW * x + 1 + (BH * y + 1) * dst_pitch] = (PX)((v00 + v10 + v01 + v11 + 2) >> 2);
            }
        }
    }
}

kernel void kf_el_from444(
    __global       PX* __restrict dst,
    __global const PX* __restrict src,
    int width, int height, int dst_pitch, int src_pitch,
    int logUVx, int logUVy)
{
    int BW = 1 << logUVx;
    int BH = 1 << logUVy;
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        dst[x + y * dst_pitch] = src[BW * x + BH * y * src_pitch];
    }
}
