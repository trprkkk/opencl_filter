/* ============================================================================
 * kfm_filterbase.cl — OpenCL port of the KFM coefficient kernels that live in
 * the shared base `KFMFilterBase.cu` (KFM, MIT) and that the KAnalyzeStatic
 * pipeline (MergeStatic.cu) is built from.
 *
 * KAnalyzeStatic::GetFrameT (MergeStatic.cu) computes a static/coefficient
 * frame whose per-plane dataflow is:
 *   padded = pad(frame n)                       // host glue (VPAD mirror, rig)
 *   CompareFields(padded, flagtmp)  -> kf_calc_combe (Y/U/V)
 *   MergeUVCoefs(flagtmp)           -> kf_merge_uvcoefs (fold UV into Y)
 *   ExtendCoefs(flagtmp, flagc)     -> kf_extend_coef2 (Y only)
 *   GetTemporalDiff(diff, flagtmp)  -> kf_min_frames  (already in
 *                                       kfm_mergestatic.cl)  over 3 diff frames
 *   MergeUVCoefs(flagtmp)           -> kf_merge_uvcoefs
 *   ExtendCoefs(flagtmp, flagd)     -> kf_extend_coef2
 *   AndCoefs(flagc, flagd)          -> kf_and_coefs (already in
 *                                       kfm_mergestatic.cl)  combe ^ diff
 *   ApplyUVCoefs(flagc)             -> kf_apply_uvcoefs_420 (Y -> UV)
 * This file supplies the four kernels defined in KFMFilterBase.cu that the
 * pipeline still needs (kf_min_frames and kf_and_coefs already live in
 * kfm_mergestatic.cl), plus the shared mirror-pad helpers kf_padv/kf_padh
 * (used by KDeblock's DeblockPlane, CombingAnalyze flag planes, ...), the
 * MergeBlock blender kf_merge_block (used by KPatchCombe/KFMSwitch), and the
 * five CombingAnalyze/CompareFields helpers kf_average, kf_max,
 * kf_merge_uvflags, kf_copy_border and kf_analyze_frame.
 * All twelve are per-pixel / per-plane integer ops whose
 * channels are independent, so the CUDA 4-wide vectorisation is equivalent to a
 * scalar translation (bit-identical for plane width a multiple of 4; the
 * scalar twins kf_max / kf_merge_uvflags / kf_copy_border are exact twins at
 * any width).
 *
 * Status:
 *   // ALG-VERIFIED (python/run_kfm_filterbase.py) vs sim/kfm_filterbase_ref.cpp
 *   //   cpu_calc_combe / cpu_merge_uvcoefs / cpu_apply_uvcoefs_420 /
 *   //   cpu_padv / cpu_padh / cpu_merge / cpu_average / cpu_max /
 *   //   cpu_merge_uvflags / cpu_copy_border / cpu_analyze_frame are exact
 *   //   twins; cpu_extend_coef (below) is the CUDA kl_extend_coef2 twin that
 *   //   the .cl transliterates.
 *
 * Fidelity / assembly notes:
 *  - calc_combe / merge_uvcoefs / apply_uvcoefs_420 have identical CUDA and CPU
 *    implementations upstream.
 *  - ExtendCoefs: the real CUDA device path uses kl_extend_coef2 (row index
 *    clamped to [0,height-1]), which is what this .cl transliterates.  The
 *    original CPU *fallback* branch instead runs cpu_extend_coef over the
 *    interior rows plus cpu_copy_border, which copies the extreme rows straight
 *    through (no max with the neighbour); that differs from the device kernel
 *    at row 0 and row height-1.  We follow the device kernel (the OpenCL target
 *    is the GPU), so kf_extend_coef2 is checked against its own transliteration,
 *    and the CPU-vs-CUDA border discrepancy is upstream's, noted here (and
 *    only affects 2 rows, and KAnalyzeStatic's result there anyway flows into
 *    a band-pass coefficient).
 *  - calc_combe reads source rows y-2..y+2 with no border guard: upstream feeds
 *    it a vertically-mirror-padded frame (VPAD=4), so the top/bottom rows are
 *    rig-bound on the padded plane; the arithmetic is verified over the
 *    interior (rows 2..height-3) here.  The host pad layout / offset is a
 *    RIG-VERIFY seam (the calc_combe launch geometry over the padded buffer is
 *    host glue, as with KEdgeLevel's el_to444 host sizing).
 *  - analyze_frame likewise reads source rows y-1/y+1 unguarded from
 *    VPAD-padded planes (interior-origin pointers; the host pad layout/offset
 *    is a RIG-VERIFY seam as with padv/padh) — but its border outputs are
 *    defined by the pad, so all height rows are verified with padded inputs.
 *  - analyze_frame likewise reads source rows y-1/y+1 unguarded from
 *    VPAD-padded planes (interior-origin pointers; the host pad layout/offset
 *    is a RIG-VERIFY seam as with padv/padh) — but its border outputs are
 *    defined by the pad, so all height rows are verified with padded inputs.
 *  - calc_combe's output is clamped to [0,255] regardless of pixel bit depth
 *    (upstream casts the clamped int to the pixel type; the coefficient planes
 *    KAnalyzeStatic uses are 0..128-scaled values, so this matches).
 *  - merge_uvcoefs / apply_uvcoefs_420 assume YV12-style logUVx=logUVy=1
 *    (KAnalyzeStatic enforces 420).
 *  - padv/padh are in-place mirror pads: dst points at the interior origin of
 *    a buffer with vpad/hpad spare rows/columns.  Reads touch only interior
 *    rows/columns and writes only pad rows/columns, so each kernel is race-free
 *    in a single launch; the 2D pad is padv-then-padh (padh over the full
 *    height+2*vpad), sequenced by the host as upstream does.  Preconditions
 *    (faithful host config, always true upstream): vpad <= height,
 *    hpad <= width.  CUDA reads the pad index from the unguarded threadIdx
 *    (blockDim == pad count at every call site); the .cl guards x/y explicitly
 *    over exactly the twins' loop domain.
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

/* CalcCombe(a,b,c,d,e) = abs(a + c*4 + e - (b+d)*3)  (KFMFilterBase.cuh) */
static int kf_calc_combe_val(int a, int b, int c, int d, int e)
{
    int v = a + c * 4 + e - (b + d) * 3;
    if (v < 0) v = -v;
    return v;
}

/* ---------------------------------------------------------------------------
 * kf_calc_combe — CompareFields core (cpu_calc_combe / kl_calc_combe twin).
 * Combing measure at each pixel from the 5 vertical taps y-2..y+2 of a
 * (host-padded) interlaced frame: combe = CalcCombe(...) >> 2, clamped [0,255].
 * Grid: 2D (width,height).  Interior ALG-VERIFIED; border rows need the padded
 * plane (RIG-VERIFY).
 * -------------------------------------------------------------------------*/
kernel void kf_calc_combe(
    __global       PX* __restrict dst,
    __global const PX* __restrict src,
    int width, int height, int pitch)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int off = x + y * pitch;
        int combe = kf_calc_combe_val(
            (int)src[off - 2 * pitch],
            (int)src[off - 1 * pitch],
            (int)src[off + 0 * pitch],
            (int)src[off + 1 * pitch],
            (int)src[off + 2 * pitch]);
        combe >>= 2;
        if (combe < 0) combe = 0; else if (combe > 255) combe = 255;
        dst[off] = (PX)combe;
    }
}

/* ---------------------------------------------------------------------------
 * kf_merge_uvcoefs — MergeUVCoefs core (cpu_merge_uvcoefs / kl_merge_uvcoefs
 * twin).  In-place on the full-size Y coefficient plane: fY = max(fY,
 * max(fU,fV)) where the UV coefficient at (x,y) comes from the subsampled
 * plane offset (x>>logUVx, y>>logUVy).  Grid 2D (width,height)=Y plane dims.
 * YV12 logUVx=logUVy=1.  // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_merge_uvcoefs(
    __global       PX* __restrict fY,
    __global const PX* __restrict fU,
    __global const PX* __restrict fV,
    int width, int height, int pitchY, int pitchUV,
    int logUVx, int logUVy)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int offY = x + y * pitchY;
        int offUV = (x >> logUVx) + (y >> logUVy) * pitchUV;
        int u = (int)fU[offUV], v = (int)fV[offUV];
        int yv = (int)fY[offY];
        if (u > yv) yv = u;
        if (v > yv) yv = v;
        fY[offY] = (PX)yv;
    }
}

/* ---------------------------------------------------------------------------
 * kf_extend_coef2 — ExtendCoefs core = the CUDA kl_extend_coef2 device kernel
 * (see fidelity note re the divergent CPU fallback).  dst[x+y*pitch] = max of
 * src over the 3 vertical rows y-1..y+1 with y clamped to [0,height-1], so
 * borders are handled without an extra pad.  Y plane only.  ALG-VERIFIED.
 * -------------------------------------------------------------------------*/
kernel void kf_extend_coef2(
    __global       PX* __restrict dst,
    __global const PX* __restrict src,
    int width, int height, int pitch)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int y0 = y - 1; if (y0 < 0) y0 = 0;
        int y2 = y + 1; if (y2 > height - 1) y2 = height - 1;
        int off = x + y * pitch;
        int a = (int)src[x + y0 * pitch];
        int b = (int)src[off];
        int c = (int)src[x + y2 * pitch];
        int m = a;
        if (b > m) m = b;
        if (c > m) m = c;
        dst[off] = (PX)m;
    }
}

/* ---------------------------------------------------------------------------
 * kf_apply_uvcoefs_420 — ApplyUVCoefs core (cpu_apply_uvcoefs_420 /
 * kl_apply_uvcoefs_420 twin).  Down-samples the Y coefficient plane into U and
 * V (both equal) as the rounded 2x2 average:
 *   fU=fV = (fY[2x,2y]+fY[2x+1,2y]+fY[2x,2y+1]+fY[2x+1,2y+1] + 2) >> 2
 * Grid 2D (widthUV,heightUV).  Assumes logUVx=logUVy=1 (YV12). ALG-VERIFIED.
 * -------------------------------------------------------------------------*/
kernel void kf_apply_uvcoefs_420(
    __global const PX* __restrict fY,
    __global       PX* __restrict fU,
    __global       PX* __restrict fV,
    int widthUV, int heightUV, int pitchY, int pitchUV)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < widthUV && y < heightUV) {
        int offY0 = (2 * x + 0) + (2 * y + 0) * pitchY;
        int v =
            (int)fY[offY0] +
            (int)fY[offY0 + 1] +
            (int)fY[offY0 + pitchY] +
            (int)fY[offY0 + pitchY + 1];
        int avg = (v + 2) >> 2;
        int offUV = x + y * pitchUV;
        fU[offUV] = (PX)avg;
        fV[offUV] = (PX)avg;
    }
}

/* ---------------------------------------------------------------------------
 * kf_padv — vertical mirror pad (cpu_padv / kl_padv twin, KFMFilterBase.cu).
 * In-place: dst points at the interior (visible) origin inside a buffer with
 * >= vpad spare rows above and below.  For y in [0,vpad), x in [0,width):
 *   row -y-1 <- row y            (top mirror)
 *   row height+y <- row height-y-1 (bottom mirror)
 * Grid: 2D (width, vpad).  // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_padv(
    __global PX* __restrict dst,
    int width, int height, int pitch, int vpad)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < vpad) {
        dst[x + (-y - 1) * pitch] = dst[x + y * pitch];
        dst[x + (height + y) * pitch] = dst[x + (height - y - 1) * pitch];
    }
}

/* ---------------------------------------------------------------------------
 * kf_padh — horizontal mirror pad (cpu_padh / kl_padh twin, KFMFilterBase.cu).
 * In-place: dst points at the interior origin inside a buffer with >= hpad
 * spare columns left and right.  For y in [0,height), x in [0,hpad):
 *   col -x-1 <- col x            (left mirror)
 *   col width+x <- col width-x-1 (right mirror)
 * Grid: 2D (hpad, height).  // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_padh(
    __global PX* __restrict dst,
    int width, int height, int pitch, int hpad)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < hpad && y < height) {
        dst[(-x - 1) + y * pitch] = dst[x + y * pitch];
        dst[(width + x) + y * pitch] = dst[(width - x - 1) + y * pitch];
    }
}

/* ---------------------------------------------------------------------------
 * kf_merge_block — MergeBlock blender (cpu_merge / kl_merge twin,
 * KFMFilterBase.cu; driven per plane by KPatchCombe/KFMSwitch in KFMKernel.cu).
 * Masked blend of the 24/30p base frame with the 60p bob frame under the uchar
 * comb flag (flag is uchar even for 16-bit pixels, as upstream):
 *   combe = flag; inv = 128 - combe
 *   dst = (PX)((combe*src60 + inv*src24 + 64) >> 7)
 * Lanes are independent, so the scalar port is lane-identical to the vector
 * CUDA kernel (bit-identical for plane width a multiple of 4 — CUDA covers
 * width4 lanes and never writes the width%4 trailing columns).  No clamping:
 * production flags are in [0,128], but the transcription (like upstream)
 * applies the formula verbatim over the full uchar flag domain — combe > 128
 * makes inv negative and the >> 7 is an arithmetic shift, with the (PX) cast
 * wrapping mod 256/65536 exactly like VHelper::cast_to.  Grid: 2D
 * (width,height).  // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_merge_block(
    __global       PX* __restrict dst,
    __global const PX* __restrict src24,
    __global const PX* __restrict src60,
    int width, int height, int pitch,
    __global const uchar* __restrict flag, int flag_pitch)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int off = x + y * pitch;
        int combe = (int)flag[x + y * flag_pitch];
        int invcombe = 128 - combe;
        int t = (combe * (int)src60[off] + invcombe * (int)src24[off] + 64) >> 7;
        dst[off] = (PX)t;
    }
}

/* ---------------------------------------------------------------------------
 * kf_average — temporal/pixel mean (cpu_average / kl_average twin,
 * KFMFilterBase.cu; uchar4/ushort4 instantiations, driven by CombingAnalyze).
 * dst = (src0 + src1) >> 1 per pixel (floor mean; sums are non-negative so
 * the shift is exact, and the result is always in range, so
 * VHelper::cast_to is a plain store).  Lanes are independent, so the scalar
 * port is lane-identical to the vector CUDA kernel (bit-identical for plane
 * width a multiple of 4).  Grid: 2D (width,height).  // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_average(
    __global       PX* __restrict dst,
    __global const PX* __restrict src0,
    __global const PX* __restrict src1,
    int width, int height, int pitch)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int off = x + y * pitch;
        int tmp = ((int)src0[off] + (int)src1[off]) >> 1;
        dst[off] = (PX)tmp;
    }
}

/* ---------------------------------------------------------------------------
 * kf_max — per-pixel max (cpu_max / kl_max twin, KFMFilterBase.cu; uint8_t
 * ONLY instantiation — the uchar4 instantiation is commented out upstream —
 * so this uchar kernel is exact parity, at arbitrary width since the twin is
 * scalar).  dst = max(src0, src1).  Faithful oddity: upstream assigns the int
 * tmp straight to the uint8 dst (its VHelper::cast_to line is commented
 * out); the value is always in range so this equals a plain store.  Grid: 2D
 * (width,height).  // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_max(
    __global       uchar* __restrict dst,
    __global const uchar* __restrict src0,
    __global const uchar* __restrict src1,
    int width, int height, int pitch)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int off = x + y * pitch;
        int a = (int)src0[off];
        int b = (int)src1[off];
        dst[off] = (uchar)((a > b) ? a : b);
    }
}

/* ---------------------------------------------------------------------------
 * kf_merge_uvflags — MergeUVFlags core (cpu_merge_uvflags / kl_merge_uvflags
 * twin, KFMFilterBase.cu; uint8_t-only, scalar).  In-place fold of the UV
 * comb flags into the Y flag plane: fY |= ((fU | fV) << 4), with the UV flag
 * read at subsampled offset (x>>logUVx, y>>logUVy).  Race-free: each lane
 * touches only its own fY element (U/V are read-only).  The shift is int
 * arithmetic and the uint8_t |= store wraps mod 256, reproduced here by the
 * (uchar) cast (production flags are small, but the transcription covers the
 * full uchar domain verbatim).  Grid: 2D (width,height).  // ALG-VERIFIED
 * (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_merge_uvflags(
    __global       uchar* __restrict fY,
    __global const uchar* __restrict fU,
    __global const uchar* __restrict fV,
    int width, int height, int pitchY, int pitchUV, int logUVx, int logUVy)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int offUV = (x >> logUVx) + (y >> logUVy) * pitchUV;
        int flagUV = (int)fU[offUV] | (int)fV[offUV];
        int oY = x + y * pitchY;
        fY[oY] = (uchar)((int)fY[oY] | (flagUV << 4));
    }
}

/* ---------------------------------------------------------------------------
 * kf_copy_border — extreme-row copy (cpu_copy_border / kl_copy_border twin,
 * KFMFilterBase.cu; uint8_t/uint16_t instantiations; CPU-fallback helper used
 * by the ExtendCoefs fallback path).  For y in [0,vborder), x in [0,width):
 * copies row y and row height-y-1 from src to dst (extreme rows straight
 * through — this is the twin whose divergence from the kl_extend_coef2 device
 * kernel at rows 0/height-1 is noted in the file header).  Out-of-place;
 * even where vborder*2 > height makes two lanes share a row, both write the
 * identical value (same src read), so the overlap is benign.  CUDA reads y
 * from the unguarded threadIdx.y (blockDim.y == vborder at every call site);
 * the .cl guards x/y explicitly over exactly the twins' loop domain.  Grid:
 * 2D (width,vborder).  // ALG-VERIFIED (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_copy_border(
    __global       PX* __restrict dst,
    __global const PX* __restrict src,
    int width, int height, int pitch, int vborder)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < vborder) {
        dst[x + y * pitch] = src[x + y * pitch];
        dst[x + (height - y - 1) * pitch] = src[x + (height - y - 1) * pitch];
    }
}

/* Diff-flag bits for kf_analyze_frame (KFM.h: MOVE=1, SHIMA=2, LSHIMA=4). */
#define KF_MOVE 1
#define KF_SHIMA 2
#define KF_LSHIMA 4

/* ---------------------------------------------------------------------------
 * kf_analyze_frame — CompareFields flag classifier (cpu_analyze_frame /
 * kl_analyze_frame twin, KFMFilterBase.cu; uchar4/ushort4 source
 * instantiations via LaunchAnalyzeFrame, uchar4 flag output; distinct from
 * the uchar2 block-based kl_analyze_frame that lives in CombingAnalyze.cu).
 * Per pixel, from base rows y-1/y/y+1, sref rows y/y+1, mref row y:
 *   a=base[y-1], b=sref[y], c=base[y], d=sref[y+1], e=base[y+1]
 *   t    = CalcCombe(a,b,c,d,e) = |a + 4c + e - 3(b+d)|   (UNSHIFTED — no
 *            >> 2 here, unlike kf_calc_combe; max 6*PX_MAX, no int overflow)
 *   diff = |mref[y] - c|
 *   flag = (t > threshS ? SHIMA : 0) | (t > threshLS ? LSHIMA : 0)
 *          | (diff > threshM ? MOVE : 0)                      (MakeDiffFlag)
 * The taps are lane-independent, so the scalar port is lane-identical to the
 * vector twin.  Sources point at the interior origin of VPAD-mirror-padded
 * planes (rows y-1/y+1 are read unguarded, as upstream); the host-side pad
 * layout/offset is a RIG-VERIFY seam as with kf_padv/kf_padh — but unlike
 * kf_calc_combe, the border outputs are DEFINED by the pad, so all height
 * rows are verified here with padded inputs.  Grid: 2D (width,height);
 * source row stride pitch, flag row stride dstPitch.  // ALG-VERIFIED
 * (integer)
 * -------------------------------------------------------------------------*/
kernel void kf_analyze_frame(
    __global       uchar* __restrict dst, int dstPitch,
    __global const PX* __restrict base,
    __global const PX* __restrict sref,
    __global const PX* __restrict mref,
    int width, int height, int pitch, int threshM, int threshS, int threshLS)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        int a = (int)base[x + (y - 1) * pitch];
        int b = (int)sref[x + y * pitch];
        int c = (int)base[x + y * pitch];
        int d = (int)sref[x + (y + 1) * pitch];
        int e = (int)base[x + (y + 1) * pitch];
        int t = kf_calc_combe_val(a, b, c, d, e);
        int diff = (int)mref[x + y * pitch] - c;
        if (diff < 0) diff = -diff;
        int flag = 0;
        if (t > threshS) flag |= KF_SHIMA;
        if (t > threshLS) flag |= KF_LSHIMA;
        if (diff > threshM) flag |= KF_MOVE;
        dst[x + y * dstPitch] = (uchar)flag;
    }
}
