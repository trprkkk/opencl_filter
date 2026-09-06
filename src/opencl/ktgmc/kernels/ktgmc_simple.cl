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

/* ---------------------------------------------------------------------------
 * K9. Horizontal resampler (kl_resample_h) — used by KGaussResize etc.
 *     Per output column x a FIR over source columns [offset[x] .. +filter_size).
 *     offset[]/coef[] come from the host ResamplingProgram.
 * -------------------------------------------------------------------------*/
kernel void kt_resample_h(
    __global const PX* __restrict src, int src_pitch,
    __global       PX* __restrict dst, int dst_pitch,
    int width, int height,
    __global const int*   __restrict offset,
    __global const float* __restrict coef,
    int filter_size)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int   begin = offset[x];
        float acc   = 0.f;
        for (int i = 0; i < filter_size; ++i)
            acc += convert_float(src[(begin + i) + y * src_pitch]) *
                   coef[x * filter_size + i];
        acc = clamp(acc, 0.f, (float)PX_MAX);
        dst[x + y * dst_pitch] = (PX)(int)(acc + 0.5f);
    }
}

/* ---------------------------------------------------------------------------
 * K10. box5 vertical min/max (kl_box5_v_and_border with Min5/Max5) =
 *      Xpand/Expand VerticalX2.  Rows y-2..y+2 at the same x; rows out of the
 *      picture are clamped to the centre row.
 * -------------------------------------------------------------------------*/
kernel void kt_box5_minmax(
    __global const PX* __restrict src, int pitch,
    __global       PX* __restrict dst, int pitcho,
    int width, int height,
    int is_min)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int v2 = convert_int(src[x + y * pitch]);
        int v0 = (y - 2 >= 0)     ? convert_int(src[x + (y - 2) * pitch]) : v2;
        int v1 = (y - 1 >= 0)     ? convert_int(src[x + (y - 1) * pitch]) : v2;
        int v3 = (y + 1 < height) ? convert_int(src[x + (y + 1) * pitch]) : v2;
        int v4 = (y + 2 < height) ? convert_int(src[x + (y + 2) * pitch]) : v2;
        int m;
        if (is_min)
            m = min(min(min(v0, v1), min(v2, v3)), v4);
        else
            m = max(max(max(v0, v1), max(v2, v3)), v4);
        dst[x + y * pitcho] = (PX)m;
    }
}

/* ---------------------------------------------------------------------------
 * K11. logic min / max on two clips (kl_logic2 LogicMin/LogicMax).
 * -------------------------------------------------------------------------*/
kernel void kt_logic_minmax(
    __global const PX* __restrict a, int a_pitch,
    __global const PX* __restrict b, int b_pitch,
    __global       PX* __restrict dst, int dst_pitch,
    int width, int height,
    int is_min)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int va = convert_int(a[x + y * a_pitch]);
        int vb = convert_int(b[x + y * b_pitch]);
        dst[x + y * dst_pitch] = (PX)(is_min ? min(va, vb) : max(va, vb));
    }
}

/* ---------------------------------------------------------------------------
 * K12. Vertical resharpen (kl_box3_v Resharpen, KTGMC_VResharpen).
 *      out = (min(prev,cur,next) + max(prev,cur,next) + 1) >> 1
 *      top/bottom rows clamp to the current row.
 * -------------------------------------------------------------------------*/
kernel void kt_vresharpen(
    __global const PX* __restrict src, int pitch,
    __global       PX* __restrict dst, int pitcho,
    int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int v1 = convert_int(src[x + y * pitch]);
        int v0 = (y == 0)          ? v1 : convert_int(src[x + (y - 1) * pitch]);
        int v2 = (y == height - 1) ? v1 : convert_int(src[x + (y + 1) * pitch]);
        int mn = min(v0, min(v1, v2));
        int mx = max(v0, max(v1, v2));
        dst[x + y * pitcho] = (PX)((mn + mx + 1) >> 1);
    }
}

/* ---------------------------------------------------------------------------
 * K13. Resharpen (kl_resharpen, KTGMC_Resharpen).
 *      lut = src0 + (src0 - src1) * sharpAdj ; out = (int)clamp(lut + 0.5f)
 * -------------------------------------------------------------------------*/
kernel void kt_resharpen(
    __global const PX* __restrict s0, int p0,
    __global const PX* __restrict s1, int p1,
    __global       PX* __restrict dst, int dst_pitch,
    int width, int height,
    float sharpAdj)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        float srcx = convert_float(s0[x + y * p0]);
        float srcy = convert_float(s1[x + y * p1]);
        float lut  = srcx + (srcx - srcy) * sharpAdj;
        lut = clamp(lut + 0.5f, 0.f, (float)PX_MAX);
        dst[x + y * dst_pitch] = (PX)(int)lut;
    }
}

/* ---------------------------------------------------------------------------
 * K14. Limit over sharpen (kl_limit_over_sharpen).
 *      tMin = min(ref, min(compb, compf)); tMax = max(ref, max(compb, compf));
 *      out = clamp(src, tMin - osv, tMax + osv)
 * -------------------------------------------------------------------------*/
kernel void kt_limit_over_sharpen(
    __global const PX* __restrict src, int s_pitch,
    __global const PX* __restrict ref, int r_pitch,
    __global const PX* __restrict cb,   int cb_pitch,
    __global const PX* __restrict cf,   int cf_pitch,
    __global       PX* __restrict dst,  int dst_pitch,
    int width, int height,
    int osv)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int s = convert_int(src[x + y * s_pitch]);
        int r = convert_int(ref[x + y * r_pitch]);
        int b = convert_int(cb  [x + y * cb_pitch]);
        int f = convert_int(cf  [x + y * cf_pitch]);
        int tmin = min(r, min(b, f));
        int tmax = max(r, max(b, f));
        dst[x + y * dst_pitch] = (PX)clamp(s, tmin - osv, tmax + osv);
    }
}

/* ---------------------------------------------------------------------------
 * K15. Lossless proc (kl_lossless_proc).
 *      if ((x-half)*(y-half) < 0)   -> half
 *      else if (fabs(x-half) < fabs(y-half)) -> x
 *      else -> y   ; then clamp to [0,maxval]; out = (int)(no +0.5)
 * -------------------------------------------------------------------------*/
kernel void kt_lossless_proc(
    __global const PX* __restrict xa, int x_pitch,
    __global const PX* __restrict ya, int y_pitch,
    __global       PX* __restrict dst, int dst_pitch,
    int width, int height,
    float half, float maxval)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        float vx = convert_float(xa[x + y * x_pitch]);
        float vy = convert_float(ya[x + y * y_pitch]);
        float v;
        if ((vx - half) * (vy - half) < 0.f)
            v = half;
        else if (fabs(vx - half) < fabs(vy - half))
            v = vx;
        else
            v = vy;
        v = clamp(v, 0.f, maxval);
        dst[x + y * dst_pitch] = (PX)(int)v;
    }
}

/* ---------------------------------------------------------------------------
 * K16. Tweak search clip (kl_tweak_search_clip).
 *      repair,bobbed,blur first scaled to 8-bit units (invscale), then:
 *      tweaked = clamp(bobbed, repair-3, repair+3);
 *      ret = (blur+7)<tweaked ? blur+2
 *          : (blur-7)>tweaked ? blur-2
 *          : (blur*51 + tweaked*49) * (1/100);
 *      return ret*scale ; out=(int)clamp(d+0.5f)
 * -------------------------------------------------------------------------*/
kernel void kt_tweak_search_clip(
    __global const PX* __restrict rep, int rep_pitch,
    __global const PX* __restrict bob, int bob_pitch,
    __global const PX* __restrict blr, int blr_pitch,
    __global       PX* __restrict dst, int dst_pitch,
    int width, int height,
    float scale, float invscale)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        float repair = convert_float(rep[x + y * rep_pitch]) * invscale;
        float bobbed = convert_float(bob[x + y * bob_pitch]) * invscale;
        float blur   = convert_float(blr[x + y * blr_pitch]) * invscale;
        float tweaked = clamp(bobbed, repair - 3.f, repair + 3.f);
        float ret;
        if ((blur + 7.f) < tweaked) ret = blur + 2.f;
        else if ((blur - 7.f) > tweaked) ret = blur - 2.f;
        else ret = (blur * 51.f + tweaked * 49.f) * (1.f / 100.f);
        float d = ret * scale;
        d = clamp(d + 0.5f, 0.f, (float)PX_MAX);
        dst[x + y * dst_pitch] = (PX)(int)d;
    }
}

/* ---------------------------------------------------------------------------
 * K17. Error adjust (kl_error_adjust, KTGMC_ErrorAdjust).
 *      lut = src*(errorAdj+1) - match*errorAdj ; out=(int)clamp(lut+0.5f)
 * -------------------------------------------------------------------------*/
kernel void kt_error_adjust(
    __global const PX* __restrict src, int s_pitch,
    __global const PX* __restrict mt,  int m_pitch,
    __global       PX* __restrict dst, int dst_pitch,
    int width, int height,
    float errorAdj)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        float srcx = convert_float(src[x + y * s_pitch]);
        float matx = convert_float(mt [x + y * m_pitch]);
        float lut = (srcx * (errorAdj + 1.f)) - (matx * errorAdj);
        lut = clamp(lut + 0.5f, 0.f, (float)PX_MAX);
        dst[x + y * dst_pitch] = (PX)(int)lut;
    }
}

/* ---------------------------------------------------------------------------
 * K18. Bob shimmer fixes merge (kl_bobshimmerfixes_merge).
 *      h = 128<<scale ;
 *      diff = diff<(129<<scale) ? diff : (c1<h ? h : c1);
 *      diff = diff>(127<<scale) ? diff : (c2>h ? h : c2);
 *      out = clamp(src + diff - h, 0, PX_MAX)
 * -------------------------------------------------------------------------*/
kernel void kt_bobshimmerfixes_merge(
    __global const PX* __restrict src,  int s_pitch,
    __global const PX* __restrict diff, int d_pitch,
    __global const PX* __restrict c1,   int c1_pitch,
    __global const PX* __restrict c2,   int c2_pitch,
    __global       PX* __restrict dst,  int dst_pitch,
    int width, int height,
    int scale)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int s  = convert_int(src [x + y * s_pitch]);
        int df = convert_int(diff[x + y * d_pitch]);
        int c1v= convert_int(c1  [x + y * c1_pitch]);
        int c2v= convert_int(c2  [x + y * c2_pitch]);
        const int h  = 128 << scale;
        df = (df < (129 << scale)) ? df : ((c1v < h) ? h : c1v);
        df = (df > (127 << scale)) ? df : ((c2v > h) ? h : c2v);
        int v = s + df - h;
        dst[x + y * dst_pitch] = (PX)clamp(v, 0, PX_MAX);
    }
}

/* ---------------------------------------------------------------------------
 * K19/20. Binomial temporal soften 1 & 2 (kl_binomial_temporal_soften_1/2).
 *      Scene-change flags are computed per ref frame upstream (SAD reduction);
 *      if set, that ref is replaced by the current source pixel.
 *   radius1: out = (r0 + 2*src + r1 + 2) >> 2
 *   radius2: out = (r2 + 4*r0 + 6*src + 4*r1 + r3 + 4) >> 4
 *      (refs labelled r0..r3 in the CUDA neighbour order)
 * -------------------------------------------------------------------------*/
kernel void kt_temporal_soften_1(
    __global const PX* __restrict src, int src_pitch,
    __global const PX* __restrict ref0, int ref0_pitch,
    __global const PX* __restrict ref1, int ref1_pitch,
    __global       PX* __restrict dst,  int dst_pitch,
    int width, int height,
    int sc0, int sc1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int s  = convert_int(src [x + y * src_pitch]);
        int r0 = sc0 ? s : convert_int(ref0[x + y * ref0_pitch]);
        int r1 = sc1 ? s : convert_int(ref1[x + y * ref1_pitch]);
        int tmp = (r0 + 2 * s + r1 + 2) >> 2;
        dst[x + y * dst_pitch] = (PX)clamp(tmp, 0, PX_MAX);
    }
}

kernel void kt_temporal_soften_2(
    __global const PX* __restrict src, int src_pitch,
    __global const PX* __restrict ref0, int ref0_pitch,
    __global const PX* __restrict ref1, int ref1_pitch,
    __global const PX* __restrict ref2, int ref2_pitch,
    __global const PX* __restrict ref3, int ref3_pitch,
    __global       PX* __restrict dst,  int dst_pitch,
    int width, int height,
    int sc0, int sc1, int sc2, int sc3)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int s  = convert_int(src [x + y * src_pitch]);
        int r0 = sc0 ? s : convert_int(ref0[x + y * ref0_pitch]);
        int r1 = sc1 ? s : convert_int(ref1[x + y * ref1_pitch]);
        int r2 = sc2 ? s : convert_int(ref2[x + y * ref2_pitch]);
        int r3 = sc3 ? s : convert_int(ref3[x + y * ref3_pitch]);
        int tmp = (r2 + 4 * r0 + 6 * s + 4 * r1 + r3 + 4) >> 4;
        dst[x + y * dst_pitch] = (PX)clamp(tmp, 0, PX_MAX);
    }
}

/* ---------------------------------------------------------------------------
 * K21. Weave two fields into one frame (kl_weave, KDoubleWeave).
 *      dst row 2y   = top   row y ;  dst row 2y+1 = bottom row y
 * -------------------------------------------------------------------------*/
kernel void kt_weave(
    __global const PX* __restrict top,    int top_pitch,
    __global const PX* __restrict bottom, int bot_pitch,
    __global       PX* __restrict dst,    int dst_pitch,
    int width, int height2)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height2) {
        dst[x + (2 * y + 0) * dst_pitch] = top[x + y * top_pitch];
        dst[x + (2 * y + 1) * dst_pitch] = bottom[x + y * bot_pitch];
    }
}

/* ---------------------------------------------------------------------------
 * K22. Copy (kl_copy / CopyFunction).  Straight plane copy honoring pitch.
 * -------------------------------------------------------------------------*/
kernel void kt_copy(
    __global const PX* __restrict src, int src_pitch,
    __global       PX* __restrict dst, int dst_pitch,
    int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height)
        dst[x + y * dst_pitch] = src[x + y * src_pitch];
}

/* ===========================================================================
 * Motion super-sampling filters — from AviSynthCUDAFilters/KTGMC/MVKernel.cu
 * (part of KMSuper's "sharp" interpolation path; stage 2 of the port).
 * These are 1-plane -> 1-plane per-pixel separable filters, so they validate
 * exactly like the kernels above.
 *
 * "so called Wiener interpolation (sharp, similar to Lanczos?) — invariant
 * simplified, 6 taps. Weights: (1,-5,20,20,-5,1)/32 - added by Fizick".
 * =========================================================================*/

/* ---------------------------------------------------------------------------
 * S2a. kl_vertical_wiener.  max_pixel_value == PX_MAX.
 * -------------------------------------------------------------------------*/
kernel void kt_vertical_wiener(
    __global const PX* __restrict src, int src_pitch,
    __global       PX* __restrict dst, int dst_pitch,
    int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width) {
        if (y < 2) {
            dst[x + y * dst_pitch] = (PX)((src[x + y * src_pitch] +
                                           src[x + (y + 1) * src_pitch] + 1) >> 1);
        } else if (y < height - 4) {
            int p0 = convert_int(src[x + (y - 2) * src_pitch]);
            int p1 = convert_int(src[x + (y - 1) * src_pitch]);
            int p2 = convert_int(src[x + (y + 0) * src_pitch]);
            int p3 = convert_int(src[x + (y + 1) * src_pitch]);
            int p4 = convert_int(src[x + (y + 2) * src_pitch]);
            int p5 = convert_int(src[x + (y + 3) * src_pitch]);
            int num = p0 + ((-p1 + 4 * p2 + 4 * p3 - p4) * 5) + p5 + 16;
            int v = clamp(num >> 5, 0, PX_MAX);
            dst[x + y * dst_pitch] = (PX)v;
        } else if (y < height - 1) {
            dst[x + y * dst_pitch] = (PX)((src[x + y * src_pitch] +
                                           src[x + (y + 1) * src_pitch] + 1) >> 1);
        } else { /* last row */
            dst[x + y * dst_pitch] = src[x + y * src_pitch];
        }
    }
}

/* ---------------------------------------------------------------------------
 * S2b. kl_horizontal_wiener.
 * -------------------------------------------------------------------------*/
kernel void kt_horizontal_wiener(
    __global const PX* __restrict src, int src_pitch,
    __global       PX* __restrict dst, int dst_pitch,
    int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (y < height) {
        if (x < 2) {
            dst[x + y * dst_pitch] = (PX)((src[x + y * src_pitch] +
                                           src[(x + 1) + y * src_pitch] + 1) >> 1);
        } else if (x < width - 4) {
            int p0 = convert_int(src[(x - 2) + y * src_pitch]);
            int p1 = convert_int(src[(x - 1) + y * src_pitch]);
            int p2 = convert_int(src[(x + 0) + y * src_pitch]);
            int p3 = convert_int(src[(x + 1) + y * src_pitch]);
            int p4 = convert_int(src[(x + 2) + y * src_pitch]);
            int p5 = convert_int(src[(x + 3) + y * src_pitch]);
            int num = p0 + ((-p1 + 4 * p2 + 4 * p3 - p4) * 5) + p5 + 16;
            int v = clamp(num >> 5, 0, PX_MAX);
            dst[x + y * dst_pitch] = (PX)v;
        } else if (x < width - 1) {
            dst[x + y * dst_pitch] = (PX)((src[x + y * src_pitch] +
                                           src[(x + 1) + y * src_pitch] + 1) >> 1);
        } else { /* last column */
            dst[x + y * dst_pitch] = src[x + y * src_pitch];
        }
    }
}

/* ---------------------------------------------------------------------------
 * Motion-analysis helper (from AviSynthCUDAFilters/KTGMC/Kernel.cu,
 * KBinomialTemporalSoften path): frame/plane-level SAD sum between two planes
 * used to decide scene-change per reference frame.
 *
 * CUDA kl_calculate_sad summed |cur - ref| over a whole plane using __sad /
 * __vabsdiff4 accumulate semantics (sum of per-pixel absolute differences)
 * into a per-plane int, reduced over the plane.  Here it is a straightforward
 * global reduction of abs(a-b).  Host must zero *out* before dispatch (the CUDA
 * side did the same with kl_init_sad).  On a real pipeline, one plane pair is
 * summed for each reference (prv2/prv1/cur/fwd1/fwd2) and the host compares each
 * to scenechange*width*height to build the scN flags consumed by
 * kt_temporal_soften_1/2.
 *
 * NOTE: sum is accumulated in 32-bit int exactly as the CUDA `int sum`.  For
 * 16-bit + very large frames that can overflow; CUDA had the same limitation
 * (frame SAD is only used for the scene-change boolean), so it is preserved.
 * -------------------------------------------------------------------------*/
kernel void kt_plane_sad(
    __global const PX* __restrict a, int a_pitch,
    __global const PX* __restrict b, int b_pitch,
    int width, int height,
    __global int* __restrict out)   /* += sum over whole plane */
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int d = convert_int(a[x + y * a_pitch]) - convert_int(b[x + y * b_pitch]);
        if (d < 0) d = -d;
        atomic_add(out, d);
    }
}
