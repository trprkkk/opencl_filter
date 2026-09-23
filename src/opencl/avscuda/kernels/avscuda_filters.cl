/* ============================================================================
 * avscuda_filters.cl — OpenCL port of the AvsCUDA Invert kernels in
 * `AvsCUDA/filters/Filters.cu` (rigaya/AviSynthCUDAFilters, MIT).
 *
 * Upstream kl_invert_plane is templated on the vector word (int / int2 /
 * float4) and XORs whole words; the scalar ports below are lane-identical:
 *   int   (planar u8 full invert, YUY2/RGB32 packed): byte x ^= byte (x&3)
 *           of mask0 (logical lane extraction — CUDA `s ^ mask0` ≡ per-byte
 *           XOR; planar path always passes mask0 = 0xFFFFFFFF).
 *   int2  (planar u16, RGB64 packed): element x ^= 16-bit lane of the 8-byte
 *           (mask0,mask1) sequence — word ((x>>1)&1) selects mask0/mask1, bit
 *           (x&1) selects the half.  Planar callers pass a mask64 whose four
 *           16-bit parts are all equal; RGB64 packs per-channel masks.
 *   float4 (planar f32): 1.0f - x, masks ignored (launch passes 0,0).
 *           The C twin invert_plane_float_c has a chroma variant (max = 0)
 *           under FLOAT_CHROMA_IS_ZERO_CENTERED, which is never defined
 *           upstream — so max = 1.0f always and the twin is exact.
 * kl_invert_rgb (RGB24/RGB48 interleaved): element 3*x+c ^= channel mask c,
 *   narrowed to the element width on store (upstream `pf[j] ^= mask` — the
 *   int mask truncates to BYTE/uint16_t; production masks are 0 or all-ones
 *   per channel, partial masks pin the narrowing).
 * All four are IN-PLACE upstream, transcribed faithfully here.
 *
 * Word-overhang note: upstream rounds the row up to whole words
 * ((rowsize+3)>>2 ints, (rowsize+7)>>3 int2s), so a non-multiple rowsize
 * writes up to 3 (u8) / 7 (u16) bytes past rowsize into the pitch padding
 * (harmless with SIMD-aligned production pitches, but technically past the
 * row).  The scalar ports take width = exact pixels/elements and write
 * exactly the row — the only divergence from upstream, and only in the
 * overhang bytes.  Widths that are not a multiple of the word size are
 * covered by the harness.
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

/* ---------------------------------------------------------------------------
 * ka_invert_plane_u8 — in-place byte-lane XOR
 * (kl_invert_plane<int> twin; width/pitch in bytes/pixels).
 * Grid: 2D (width, height).
 * // ALG-VERIFIED via python/run_avscuda_filters.py.
 * -------------------------------------------------------------------------*/
kernel void ka_invert_plane_u8(
    __global uchar* __restrict ptr,
    int width, int height, int pitch, int mask0)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    unsigned lane = ((unsigned)mask0 >> (8u * ((unsigned)x & 3u))) & 0xFFu;
    int off = x + y * pitch;
    ptr[off] = (uchar)((int)ptr[off] ^ (int)lane);
}

/* ---------------------------------------------------------------------------
 * ka_invert_plane_u16 — in-place 16-bit-lane XOR
 * (kl_invert_plane<int2> twin; width/pitch in uint16 elements).
 * Grid: 2D (width, height).
 * // ALG-VERIFIED via python/run_avscuda_filters.py.
 * -------------------------------------------------------------------------*/
kernel void ka_invert_plane_u16(
    __global ushort* __restrict ptr,
    int width, int height, int pitch, int mask0, int mask1)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    unsigned ux = (unsigned)x;
    unsigned m32 = ((((ux >> 1) & 1u) != 0u) ? (unsigned)mask1 : (unsigned)mask0);
    unsigned lane = (m32 >> (16u * (ux & 1u))) & 0xFFFFu;
    int off = x + y * pitch;
    ptr[off] = (ushort)((int)ptr[off] ^ (int)lane);
}

/* ---------------------------------------------------------------------------
 * ka_invert_plane_f32 — in-place 1.0f - x (kl_invert_plane<float4> twin;
 * masks ignored upstream too; width/pitch in float elements).
 * Grid: 2D (width, height).
 * // ALG-VERIFIED via python/run_avscuda_filters.py.
 * -------------------------------------------------------------------------*/
kernel void ka_invert_plane_f32(
    __global float* __restrict ptr,
    int width, int height, int pitch)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    int off = x + y * pitch;
    ptr[off] = 1.0f - ptr[off];
}

/* ---------------------------------------------------------------------------
 * ka_invert_rgb — in-place interleaved 3-element XOR with per-channel masks
 * (kl_invert_rgb twin; pixel_t = BYTE for RGB24, uint16_t for RGB48;
 * el_pitch in elements, width in pixels).  Width-agnostic, so PX-generic.
 * Grid: 2D (width, height).
 * // ALG-VERIFIED via python/run_avscuda_filters.py.
 * -------------------------------------------------------------------------*/
kernel void ka_invert_rgb(
    __global PX* __restrict ptr,
    int width, int height, int el_pitch,
    int bMask, int gMask, int rMask)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    int base = 3 * x + y * el_pitch;
    ptr[base + 0] = (PX)((int)ptr[base + 0] ^ bMask);
    ptr[base + 1] = (PX)((int)ptr[base + 1] ^ gMask);
    ptr[base + 2] = (PX)((int)ptr[base + 2] ^ rMask);
}
