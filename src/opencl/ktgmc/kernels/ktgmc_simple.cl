/* ============================================================================
 * ktgmc_simple.cl
 *
 * Faithful OpenCL ports of the KTGMC "simple" (per-plane, no-motion) kernels
 * originally in rigaya/AviSynthCUDAFilters -> KTGMC/Kernel.cu
 * (MIT licensed KTGMC portion).
 *
 * Each CUDA kernel there processed 4 channels per thread and used
 * float4/int4/uchar4 vector types.  Every one of these kernels is *per-pixel
 * separable* (each output element is an independent dot/point operation that
 * never depends on a sibling channel), so the scalar translation below is
 * mathematically bit-identical for the same inputs.  Vectorization can be
 * reintroduced later as a pure performance pass; it does not change results.
 *
 * Arithmetic notes (kept identical to the CUDA source):
 *   - accumulation is done in float (32-bit) in kl_resample_v / kl_resharpen,
 *     exactly like the original.
 *   - every result is clamped to [0, maxval] and then rounded with +0.5
 *     (i.e. floor(x + 0.5)) before storing, matching the CUDA
 *     "cast_to(result + 0.5f)" idiom.  maxval == 255 (PX=uchar) or 65535.
 *   - pixel buffers are plain 1-D arrays of PX with an element pitch
 *     (bytes/px in the AviSynth frame / sizeof(PX)).
 *
 * Two instantiations are used: the host compiles this source twice with
 *   -DPX=uchar   (-DPX_MAX=255)   for 8-bit
 *   -DPX=ushort  (-DPX_MAX=65535) for 16-bit
 * A KTGMC_* AVS-style filter chooses the instantiation from the plane
 * ComponentSize, exactly as the C++ switch (case 1 / case 2) does.
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

/* ---------------------------------------------------------------------------
 * K1. Vertical resampler (kl_resample_v) + ResamplingProgram.
 *     Doubles/interpolates one field along the vertical axis using a
 *     per-output-row FIR filter whose taps come from the host-side
 *     ResamplingProgram (offset[] start row, coef[row*filter_size+i]).
 * -------------------------------------------------------------------------*/
kernel void kt_resample_v(
    __global const PX* __restrict src, int src_pitch,   /* elements/row */
    __global       PX* __restrict dst, int dst_pitch,
    int width, int height,                               /* dims in pixels */
    __global const int*   __restrict offset,
    __global const float* __restrict coef,
    int filter_size)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int   begin = offset[y];
        float acc   = 0.f;
        for (int i = 0; i < filter_size; ++i) {
            acc += convert_float(src[x + (begin + i) * src_pitch]) *
                   coef[y * filter_size + i];
        }
        acc = clamp(acc, 0.f, (float)PX_MAX);
        dst[x + y * dst_pitch] = (PX)(int)(acc + 0.5f);
    }
}

/* ---------------------------------------------------------------------------
 * K2. MakeDiff / AddDiff  (kl_makediff with MakeDiffOp / AddDiffOp).
 *     range_half == 1 << (bits - 1); maxval == PX_MAX.
 * -------------------------------------------------------------------------*/
kernel void kt_makediff(
    __global const PX* __restrict a, int a_pitch,
    __global const PX* __restrict b, int b_pitch,
    __global       PX* __restrict dst, int dst_pitch,
    int width, int height,
    int mode,           /* 0 = make-diff (a-b+half), 1 = add-diff (a+b-half) */
    int range_half)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int va = convert_int(a[x + y * a_pitch]);
        int vb = convert_int(b[x + y * b_pitch]);
        int v  = (mode == 0) ? (va - vb + range_half)
                             : (va + vb - range_half);
        v = clamp(v, 0, PX_MAX);
        dst[x + y * dst_pitch] = (PX)v;
    }
}

/* ---------------------------------------------------------------------------
 * K3. RemoveGrain modes 11/12 & 20 (kl_box3x3_filter).
 *     Interior pixels only; borders are copied unchanged.
 *       RG11 : horizontal [1 2 1], then vertical (a+b*2+c+8)>>4
 *       RG20 : horizontal [1 1 1], then vertical (a+b+c+4)/9
 * -------------------------------------------------------------------------*/
kernel void kt_rg_box3x3(
    __global const PX* __restrict src, int pitch,
    __global       PX* __restrict dst, int pitcho,
    int width, int height,
    int mode)          /* 11 => RG11 blur, 20 => RG20 average */
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int s = convert_int(src[x + y * pitch]);
        int v = s;
        if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
            int ul = convert_int(src[(x - 1) + (y - 1) * pitch]);
            int uc = convert_int(src[(x)     + (y - 1) * pitch]);
            int ur = convert_int(src[(x + 1) + (y - 1) * pitch]);
            int ml = convert_int(src[(x - 1) + y      * pitch]);
            int mc = s;
            int mr = convert_int(src[(x + 1) + y      * pitch]);
            int ll = convert_int(src[(x - 1) + (y + 1) * pitch]);
            int lc = convert_int(src[(x)     + (y + 1) * pitch]);
            int lr = convert_int(src[(x + 1) + (y + 1) * pitch]);
            int vtmp;
            if (mode == 20) {
                /* RG20: (h0+h1+h2 summed) -> vertical (a+b+c+4)/9 */
                int h0 = ul + uc + ur;
                int h1 = ml + mc + mr;
                int h2 = ll + lc + lr;
                vtmp = (h0 + h1 + h2 + 4) / 9;
            } else {
                /* RG11/12: horiz a+b*2+c ; vertical (a+b*2+c+8)>>4 */
                int h0 = ul + 2 * uc + ur;
                int h1 = ml + 2 * mc + mr;
                int h2 = ll + 2 * lc + lr;
                vtmp = (h0 + 2 * h1 + h2 + 8) >> 4;
            }
            v = clamp(vtmp, 0, PX_MAX);
        }
        dst[x + y * pitcho] = (PX)v;
    }
}

/* ---------------------------------------------------------------------------
 * K4. RemoveGrain clip modes 1..4 (kl_rg_clip) 8-neighbour sort network.
 *     n selects which order statistic pair brackets the source:
 *       clamp(s, a[ n-1 ]   , a[ 7-(n-1) ])
 *     Borders copied unchanged.
 * -------------------------------------------------------------------------*/
/* Batcher odd-even 8-element compare-exchange network (19 comparisons). */
inline void sort8(int* a) {
#define CAS(i, j) do { int x=a[i], y=a[j]; a[i]=min(x,y); a[j]=max(x,y); } while(0)
    CAS(0, 1); CAS(2, 3); CAS(4, 5); CAS(6, 7);
    CAS(0, 2); CAS(1, 3); CAS(4, 6); CAS(5, 7);
    CAS(1, 2); CAS(5, 6);
    CAS(0, 4); CAS(1, 5); CAS(2, 6); CAS(3, 7);
    CAS(2, 4); CAS(3, 5);
    CAS(1, 2); CAS(3, 4); CAS(5, 6);
#undef CAS
}

kernel void kt_removegrain_clip(
    __global const PX* __restrict src, int pitch,
    __global       PX* __restrict dst, int pitcho,
    int width, int height,
    int n)          /* 1..4 */
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int s = convert_int(src[x + y * pitch]);
        int v = s;
        if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
            int a[8] = {
                convert_int(src[(x-1)+(y-1)*pitch]),
                convert_int(src[(x  )+(y-1)*pitch]),
                convert_int(src[(x+1)+(y-1)*pitch]),
                convert_int(src[(x-1)+(y  )*pitch]),
                convert_int(src[(x+1)+(y  )*pitch]),
                convert_int(src[(x-1)+(y+1)*pitch]),
                convert_int(src[(x  )+(y+1)*pitch]),
                convert_int(src[(x+1)+(y+1)*pitch]) };
            sort8(a);
            /* N=1 clamp to [a0,a7]; N=2 [a1,a6]; N=3 [a2,a5]; N=4 [a3,a4] */
            v = clamp(s, a[n-1], a[7-(n-1)]);
        }
        dst[x + y * pitcho] = (PX)v;
    }
}

/* ---------------------------------------------------------------------------
 * K5. Repair clip modes 1..4 (kl_repair_clip) 9-neighbour sort network.
 *     Same as RemoveGrain but reads the 3x3 from a *reference* clip, and the
 *     centre of the window is the source pixel value s itself.
 *     Borders copied unchanged.
 * -------------------------------------------------------------------------*/
inline void sort9(int* a) {
#define CAS(i, j) do { int x=a[i], y=a[j]; a[i]=min(x,y); a[j]=max(x,y); } while(0)
    CAS(0, 1); CAS(3, 4); CAS(6, 7);
    CAS(1, 2); CAS(4, 5); CAS(7, 8);
    CAS(0, 1); CAS(3, 4); CAS(6, 7);
    CAS(0, 3); CAS(1, 4); CAS(2, 5);
    CAS(3, 6); CAS(4, 7); CAS(5, 8);
    CAS(0, 3); CAS(1, 4); CAS(2, 5);
    CAS(1, 3); CAS(5, 7); CAS(2, 6); CAS(4, 6);
    CAS(2, 4); CAS(2, 3); CAS(5, 6);
#undef CAS
}

kernel void kt_repair_clip(
    __global const PX* __restrict src, int src_pitch,
    __global const PX* __restrict ref, int ref_pitch,
    __global       PX* __restrict dst, int dst_pitch,
    int width, int height,
    int n)          /* 1..4 */
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int s = convert_int(src[x + y * src_pitch]);
        int v = s;
        if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
            int a[9] = {
                convert_int(ref[(x-1)+(y-1)*ref_pitch]),
                convert_int(ref[(x  )+(y-1)*ref_pitch]),
                convert_int(ref[(x+1)+(y-1)*ref_pitch]),
                convert_int(ref[(x-1)+(y  )*ref_pitch]),
                s,                                       /* centre = src px  */
                convert_int(ref[(x+1)+(y  )*ref_pitch]),
                convert_int(ref[(x-1)+(y+1)*ref_pitch]),
                convert_int(ref[(x  )+(y+1)*ref_pitch]),
                convert_int(ref[(x+1)+(y+1)*ref_pitch]) };
            sort9(a);
            /* N=1 [a0,a8]; N=2 [a1,a7]; N=3 [a2,a6]; N=4 [a3,a5] */
            v = clamp(s, a[n-1], a[8-(n-1)]);
        }
        dst[x + y * dst_pitch] = (PX)v;
    }
}

/* ---------------------------------------------------------------------------
 * K6. Vertical cleaner (vertical median) (kl_vertical_cleaner_median).
 *     median of (top,current,bottom) on interior rows; edge rows unchanged.
 *     tmp = min(max(min(a,b),c), max(a,b))  ==  median(a,b,c) for a<=b?==b.
 * -------------------------------------------------------------------------*/
kernel void kt_vertical_cleaner_median(
    __global const PX* __restrict src, int pitch,
    __global       PX* __restrict dst, int pitcho,
    int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int b = convert_int(src[x + y * pitch]);
        int v = b;
        if (y >= 1 && y < height - 1) {
            int a = convert_int(src[x + (y - 1) * pitch]);
            int c = convert_int(src[x + (y + 1) * pitch]);
            v = min(max(min(a, b), c), max(a, b));
        }
        dst[x + y * pitcho] = (PX)v;
    }
}

/* ---------------------------------------------------------------------------
 * K7. To full range (kl_to_full_range).  luma=false/true below.
 *     Y : (src + (-16))   * (255.0/219.0)
 *     UV: (src + (-128))  * (128.0/112.0) + 128.0
 *     round: clamp(d + 0.5f)  (note: source applies the +0.5 *after* scale)
 * -------------------------------------------------------------------------*/
kernel void kt_to_full_range(
    __global const PX* __restrict src, int pitch,
    __global       PX* __restrict dst, int pitcho,
    int width, int height,
    int is_uv)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        float s = convert_float(src[x + y * pitch]);
        float d = (is_uv) ? (s - 128.0f) * (128.0f / 112.0f) + 128.0f
                          : (s - 16.0f)  * (255.0f / 219.0f);
        d = clamp(d + 0.5f, 0.0f, (float)PX_MAX);
        dst[x + y * pitcho] = (PX)(int)d;
    }
}

/* ---------------------------------------------------------------------------
 * K8. Merge (kl_merge).  weight in [0,1] scaled to 32767 fixed point.
 *     dst = (src0*invweight + src1*weight + 16384) >> 15
 *     (rounded toward -inf by >> of non-negative sum, exactly as CUDA int >>)
 * -------------------------------------------------------------------------*/
kernel void kt_merge(
    __global const PX* __restrict a, int a_pitch,
    __global const PX* __restrict b, int b_pitch,
    __global       PX* __restrict dst, int dst_pitch,
    int width, int height,
    int weight)     /* (int)(w*32767.0f) */
{
    int invweight = 32767 - weight;
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int va = convert_int(a[x + y * a_pitch]);
        int vb = convert_int(b[x + y * b_pitch]);
        int v = (va * invweight + vb * weight + 16384) >> 15;
        /* saturate to pixel range (CUDA vector cast to uchar/ushort) */
        v = clamp(v, 0, PX_MAX);
        dst[x + y * dst_pitch] = (PX)v;
    }
}
