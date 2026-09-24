/* ============================================================================
 * nnedi3_pad.cl — OpenCL port of the pad/copy helpers in upstream
 * rigaya/NNEDI3 (submodule of AviSynthCUDAFilters @68aef6e),
 * NNEDI3/nnedi3/nnedi3_kernel.cu @01931aa (GPL upstream):
 *   kl_pad_h                  (:40) -> kn_pad_h
 *   kl_pad_v                  (:57) -> kn_pad_v
 *   kl_copy                   (:73) -> kn_copy
 *   kl_pad_ref_and_copy_half  (:88) -> kn_pad_ref_and_copy_half
 *
 * Scalar port: one work-item per pixel (pad_h/v, copy) or per 4-pixel
 * vector with scalar loads/stores (pad_ref).  Pitches are in PIXELS
 * (elements of PX); upstream's *pitch4/*pitch byte pitches are divided by
 * 4/pixelsize by the host (production is 8-bit: the launch wrappers
 * hardcode `#define pixel_t uint8_t` and ignore `pixelsize`; the port
 * stays PX-generic per repo convention and both PX are verified).
 *
 * Pointer conventions (match upstream; see the CopyPadCUDA caller,
 * nnedi3.cpp:1761-1796, which passes the INTERIOR origin):
 *   - kn_pad_h/v `ptr` points at the interior origin; the kernels index
 *     negative margins exactly as upstream does.
 *   - kn_pad_ref_and_copy_half `ref` points at the interior origin of the
 *     padded buffer (negative x/y index the margins); `dst`/`src` point
 *     at their [0,0].
 *
 * Launch contracts:
 *   - kn_pad_h: exactly 2 groups on x (left/right); LOCAL_X MUST equal
 *     hPad (x = get_local_id(0) is unguarded — same idiom as
 *     kt_pad_frame_h; a runtime value, so no reqd_work_group_size is
 *     possible).  Any y decomposition (guarded).  In-place.
 *   - kn_pad_v: exactly 2 groups on y; LOCAL_Y MUST equal vPad.  In-place.
 *   - kn_copy: any decomposition (2D width x height guards).
 *   - kn_pad_ref_and_copy_half: global must cover AT LEAST
 *     (width4+2*hpad4) x (height+2*vpad) items (upper guards only; low
 *     side is exact by construction); any local size.
 *
 * Liveness note: upstream's CopyPadCUDA/BitBltCUDA callers are currently
 * inside `#if 0` (nnedi3.cpp:1782-1788); the live path is
 * PadRefAndCopyHalfCUDA (hpad=32/vpad=3) feeding prescreening/compute_nn.
 * kn_pad_h/v + kn_copy are ported anyway (one #if flip from live; the
 * generic copy in particular), and all four are verified below.
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

/* ---------------------------------------------------------------------------
 * kn_pad_h — in-place horizontal mirror pad (kl_pad_h twin).
 * Left group (gid0==0):  ptr[-(x+1) + y*pitch] = ptr[(x+1) + y*pitch]
 * Right group (gid0==1): ptr[(width+x) + y*pitch] = ptr[(width-(x+2)) + ...]
 * Reflection (no edge repeat): margin -1 samples interior 1, etc.
 * // ALG-VERIFIED via python/run_nnedi3_pad.py.
 * -------------------------------------------------------------------------*/
kernel void kn_pad_h(
    __global PX* ptr, int pitch, int hPad, int width, int height)
{
    bool isLeft = (get_group_id(0) == 0);
    int  x = get_local_id(0);
    int  y = get_local_id(1) + get_group_id(1) * get_local_size(1);

    if (y < height) {
        if (isLeft)
            ptr[-(x + 1) + y * pitch] = ptr[(x + 1) + y * pitch];
        else
            ptr[(width + x) + y * pitch] = ptr[(width - (x + 2)) + y * pitch];
    }
}

/* ---------------------------------------------------------------------------
 * kn_pad_v — in-place vertical mirror pad (kl_pad_v twin).
 * Top group (gid1==0):    ptr[x - (y+1)*pitch] = ptr[x + (y+1)*pitch]
 * Bottom group (gid1==1): ptr[x + (height+y)*pitch] = ptr[x + (height-(y+2))*...]
 * // ALG-VERIFIED via python/run_nnedi3_pad.py.
 * -------------------------------------------------------------------------*/
kernel void kn_pad_v(
    __global PX* ptr, int pitch, int vPad, int width, int height)
{
    bool isTop = (get_group_id(1) == 0);
    int  x = get_local_id(0) + get_group_id(0) * get_local_size(0);
    int  y = get_local_id(1);

    if (x < width) {
        if (isTop)
            ptr[x - (y + 1) * pitch] = ptr[x + (y + 1) * pitch];
        else
            ptr[x + (height + y) * pitch] =
                ptr[x + (height - (y + 2)) * pitch];
    }
}

/* ---------------------------------------------------------------------------
 * kn_copy — plain 2D copy (kl_copy twin; upstream copies width4 vpixel_t
 * vectors, i.e. width4*4 pixels — pass width in PIXELS here).
 * // ALG-VERIFIED via python/run_nnedi3_pad.py.
 * -------------------------------------------------------------------------*/
kernel void kn_copy(
    __global PX* __restrict dst, int dst_pitch,
    __global const PX* __restrict src, int src_pitch,
    int width, int height)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        dst[x + y * dst_pitch] = src[x + y * src_pitch];
    }
}

/* ---------------------------------------------------------------------------
 * kn_pad_ref_and_copy_half — fused mirror-pad + interior copy
 * (kl_pad_ref_and_copy_half twin).  One work-item per 4-pixel vector (same
 * grid as upstream); the 4 lanes load/store as scalars.  On x-mirror
 * (padx) the lanes reverse (upstream's swap(x,w)+swap(y,z)); ref is
 * written for every item, dst only for interior (!padx && !pady) items.
 * ref points at the interior origin (negative x/y index the margins).
 * // ALG-VERIFIED via python/run_nnedi3_pad.py.
 * -------------------------------------------------------------------------*/
kernel void kn_pad_ref_and_copy_half(
    __global PX* __restrict dst, int dst_pitch,
    __global PX* __restrict ref, int ref_pitch,
    __global const PX* __restrict src, int src_pitch,
    int width4, int height, int hpad4, int vpad)
{
    int x = (int)get_global_id(0) - hpad4;
    int y = (int)get_global_id(1) - vpad;

    if (x < width4 + hpad4 && y < height + vpad) {
        bool padx = true;
        int srcx = x;
        if (srcx < 0) {
            srcx = -srcx - 1;
        } else if (srcx >= width4) {
            srcx = width4 - (srcx - width4) - 1;
        } else {
            padx = false;
        }
        bool pady = true;
        int srcy = y;
        if (srcy < 0) {
            srcy = -srcy - 1;
        } else if (srcy >= height) {
            srcy = height - (srcy - height) - 1;
        } else {
            pady = false;
        }
        PX v0 = src[srcx * 4 + 0 + srcy * src_pitch];
        PX v1 = src[srcx * 4 + 1 + srcy * src_pitch];
        PX v2 = src[srcx * 4 + 2 + srcy * src_pitch];
        PX v3 = src[srcx * 4 + 3 + srcy * src_pitch];
        int ro = x * 4 + y * ref_pitch;
        if (padx) {
            ref[ro + 0] = v3;
            ref[ro + 1] = v2;
            ref[ro + 2] = v1;
            ref[ro + 3] = v0;
        } else {
            ref[ro + 0] = v0;
            ref[ro + 1] = v1;
            ref[ro + 2] = v2;
            ref[ro + 3] = v3;
        }
        if (!padx && !pady) {
            int dout = x * 4 + y * dst_pitch;
            dst[dout + 0] = v0;
            dst[dout + 1] = v1;
            dst[dout + 2] = v2;
            dst[dout + 3] = v3;
        }
    }
}
