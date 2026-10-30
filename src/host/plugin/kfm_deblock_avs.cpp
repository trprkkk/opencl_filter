// kfm_deblock_avs.cpp — AviSynth+ host plugin for the OpenCL port of KFM's
// KDeblock (upstream: rigaya/AviSynthCUDAFilters KFM/Deblock.cu, MIT).
//
// The SIXTH real AviSynth+ host plugin of the port (after KEdgeLevelOCL,
// KTemporalNROCL, KDebandOCL, KTemporalDiffOCL+KMergeStaticOCL,
// KNoiseClipOCL): it loads the ALG-VERIFIED kernels of
// src/opencl/kfm/kernels/kfm_deblock.cl (embedded into the DLL at build time)
// through the small host runtime in src/host/ocl_host.* and replays the
// upstream filter's host assembly — the FULL KDeblock class of Deblock.cu:
// the QPForDeblock helper (QP table generation incl. the DC-mask dilation and
// the KFM_SourceStart / qpclip / force_qp frame mapping), KDeblock's
// DeblockPlane (pad -> DCT deblock accumulator -> Bayer merge), the show
// paths (QP-flag text, per-block deblock flags) and the optional SharpenFilter
// tail (GaussResize unsharp + QP-driven sharpen / coeff visualisation).
//
// LAUNCH SEQUENCE per GetFrame(n) (upstream verbatim; "dev" = OpenCL launch):
//
//   QPForDeblock (upstream QPForDeblock::GetFrame / QPEntry / QPForPlane):
//     1. pick the QP source frame(s):
//        force_qp > 0            -> constant table (no frame reads)
//        src props KFM_SourceStart
//                                -> qpclip(start) [+ qpclip(start+1) when
//                                   KFM_NumSourceFrames > 1] with the
//                                   frameRateConv*n+0.3 index mapping and the
//                                   +/-10 sanity throw ("Invalid
//                                   KFM_SourceStart")
//        qpclip set              -> qpclip((int)(frameRateConv*n+0.3))
//        else                    -> src frame's own props
//     2. read the frame props QP_Table / QP_Table_Non_B / QP_Stride /
//        QP_ScaleType (and DC_Table when b_adap); missing QP_Table_Non_B =>
//        QP_TABLE_NONE (see STATE MODEL below)
//     3. MakeMask per DC_Table (upstream: kl_copy_pad + kl_max_vh<5>): in this
//        port the 8px mirror pad AND the radius-5 box-max dilation run in the
//        host staging (CPU), producing the bit-identical mask interior — see
//        DEVIATIONS 3.
//     4. dev: ONE kf_make_qp_table launch per plane (Y with qp shift (1,1),
//        U/V with (1-logUVx, 1-logUVy)), macroblock source dims
//        ((srcW+15)>>4, (srcH+15)>>4), element-max of the optional second
//        (frame 2) QP source, norm_qscale + the DC blend
//        qp = max(1, (int)(b*ratio + nonb*(1-ratio) + 0.5f)) into the
//        uint16 table planes (dims calc_qp_size(w) x calc_qp_size(h), chroma
//        planes subsampled; margins zero-filled, upstream leaves them
//        undefined — see DEVIATIONS 4).
//
//   KDeblock::DeblockEntry -> DeblockPlane per plane (Y, U, V):
//     5. pad the plane by 8px on each side (upstream `if (true)` branch: ONE
//        kl_copy_pad mirror-pad launch): in this port the padded plane is
//        assembled in the host staging (CPU; bit-identical mirror semantics —
//        the same expression as kf_copy_pad / Copy+padv+padh), then uploaded;
//        see DEVIATIONS 3.
//     6a. show == 2: dev: ONE kf_deblock_show launch per plane (per-8px-block
//         230/16 deblock-flag visualisation), done — no deblock pass.
//     6b. else:
//         dev: ONE kf_deblock launch per plane over the bw=(w+15)>>3 x
//              bh=(h+15)>>3 block grid: count = 1<<quality offset-DCT ->
//              hard-threshold -> IDCT reconstructions accumulated into the
//              16-bit block-parity accumulator (deblockShift/deblockMaxV from
//              quality+bits-10).
//         dev: ONE kf_merge_deblock launch per plane over the visible pixels:
//              4-parity-slice sum, * (1/(1<<mergeShift)) + Bayer g_ldither
//              /64, min(mergeMaxV), C-truncation into the output plane.
//     7. show == 1: CPU: the upstream DrawText flag message
//        ("KDeblock: Disabled/Using QP table/Constant QP/Using QP table with
//        DC/???") rendered with the upstream 10x20 font (kfm_textout_font.h,
//        mechanically extracted from TextOut.cu) — bit-exact cpu_draw_text
//        twin (luma 235<<shift, chroma 128<<shift, unset pixels dimmed).
//        QP_TABLE_NONE: upstream draws on the (MakeWritable'd) source frame
//        and returns it; this port draws on a fresh copy (same pixels).
//
//   SharpenFilter (only when sharp=true; upstream Create chains
//   Cache -> GaussResize -> SharpenFilter after KDeblock):
//     8. unsharp = env->Invoke("GaussResize",
//        {clip, w, h, 0, 0, w+0.0001, h+0.0001, 2}) on the deblocked clip —
//        the REAL AviSynth+ core filter, invoked exactly like upstream.
//     9. stage the deblocked frame and the unsharp frame; per plane:
//        dev: ONE kf_sharpen_coeff launch (QP>>3 -> 30-entry LUT coeff plane)
//        then dev: ONE kf_sharpen launch (3x3 edge-clamped min/max window
//        incl. the upstream min(x+1,height-1) quirk; manual float32 bilinear
//        coeff read — the documented texture substitution) or, when
//        show == 3, ONE kf_show_sharpen_coeff launch (coeff visualiser).
//
// STATE MODEL (stated like the KMergeStaticOCL/KNoiseClipOCL precedents): the
// upstream KDeblock class holds NO persistent state between GetFrame calls.
// All per-frame data arrives either from the child/qpclip frames or from
// per-frame properties; QPForDeblock's frame mapping is a pure function of
// (n, fixed config) — the KFM_SourceStart/KFM_NumSourceFrames props and
// qpclip frames it reads are themselves per-frame data (no accumulation, no
// previous-frame memory).  Every device buffer here is per-call scratch,
// overwritten (or fill-zeroed where the kernels do not write every element —
// the QP-table margins AND the block-parity accumulators, whose unwritten
// top/left parity-slice cells kf_merge_deblock sums — upstream zero-fills its
// tmpOut the same way) on every GetFrame.  The filter is stateless.
//
// UPSTREAM KERNEL INVENTORY (audited for this plugin): upstream KDeblock +
// QPForDeblock + SharpenFilter launch exactly kl_make_qp_table (x3),
// kl_copy_pad / kl_max_vh<5> (MakeMask), kl_deblock, kl_merge_deblock,
// kl_deblock_show, kl_sharpen_coeff, kl_sharpen / kl_show_sharpen_coeff —
// every one of which has an ALG-VERIFIED OpenCL twin in kfm_deblock.cl.
// kl_copy_pad + kl_max_vh are covered by the CPU-staging substitution
// (DEVIATIONS 3); every other upstream launch uses the OpenCL kernel
// directly.  (QPClip / ShowQP / FrameType of upstream AddFuncDeblock are
// separate host-only filters with no KDeblock device work — QPClip is a
// props pass-through, ShowQP's one kernel kf_scale_qp exists in the .cl but
// its frame assembly is a props viewer, FrameType is CPU-only — see GAPS at
// the file bottom; they are NOT registered by this DLL.)
//
// UPSTREAM REGISTRATION (Deblock.cu AddFuncDeblock, verbatim):
//   env->AddFunction("KDeblock",
//       "c[quality]i[str]f[thr]f[bratio]f[qpclip]c[qp]i[badap]b[sharp]b[show]i",
//       KDeblock::Create, 0);
// This DLL registers KDeblockOCL with that exact signature and defaults
// (quality 3, str 0, thr 0, bratio = badap ? 2 : 0.5, qp -1, badap true,
// sharp false, show 0) and the upstream name "KDeblock" as an alias.
//
// Documented deviations from the upstream host code (none change any pixel
// for the golden-pinned parameter space):
//   1. Device planes use pitch == width plain planes (host staging copies
//      from the frame's real pitch); all kernels are sized to the plain
//      extents.  The 16-bit block accumulator keeps upstream's
//      (width+16) x ((height+16)*4) extent and row stride, but its ORIGIN is
//      the upstream interior (+8 ushorts, +8 rows): the unshifted
//      kf_deblock / kf_merge_deblock pair is SELF-CONSISTENT with this pad
//      convention (padded source at the plane origin, margins at right /
//      bottom; accumulator written unshifted at (bh*off_z + by)*8 rows) and
//      with this plugin's merge arguments (tmp = acc.mem(), tmp_pitch_u4 =
//      pw>>2, tmp_ipitch_rows = bh*8).  Upstream's tmpOut+2+8*pitch merge
//      pre-offset and the golden E-mode +8/+8 placement both reduce to the
//      same net source-row mapping given each side's pad origin, so no +8
//      base offset is needed: every consumed accumulator cell holds the
//      upstream value (pure layout bookkeeping).
//   1b. Accumulator write race, resolved: for odd bh >= 3 the z=0 plane's
//      last tile row (even-bx, by=bh-1) and the z=1 plane's first tile
//      (odd-bx, by=0) overlap in the shared accumulator — upstream CUDA has
//      the same genuine write race (kl_deblock's plain ushort2 stores are
//      undefined at those cells); upstream's non-AVX CPU resolves it by
//      loop order (for by: for bx => the by=bh-1 / z=0 tile wins) and its
//      AVX2 single-plane path never overlaps.  This plugin reproduces the
//      CPU's deterministic resolution by splitting the kf_deblock dispatch:
//      the by=0 row launches first, then by>=1 in a second in-order
//      dispatch, so the by=bh-1 tiles always write last.  No reference
//      change is needed (the in-exe REF already resolves the conflict the
//      same way by its by-outer loop order).
//   2. The SharpenFilter tail is fused into this DLL's filter chain
//      (KDeblockOCL -> [Cache] -> GaussResize -> SharpenFilterOCL) exactly
//      like upstream's Create chain; the QP table is recomputed for the
//      sharpen stage (upstream shares it through AviSynth's Cache — the
//      table is a pure per-frame function, so both computations agree
//      bit-exact).  The upstream Cache(...) wrapper is dropped (internal
//      chain, no caching benefit inside one GetFrame).
//   3. MakeMask (mirror pad + radius-5 box max) and the DeblockPlane 8px
//      mirror pad run in the host staging instead of device kernels
//      (upstream's own CPU fallback path does exactly the same: Copy +
//      cpu_padv + cpu_padh / cpu_max twins).  The produced planes are
//      bit-identical to the device path (kf_copy_pad's documented
//      lane-reversal quirk cancels to the plain pixel mirror; cpu_max_v/h
//      are the exact twins of the device kernels).  The OpenCL kernels
//      kf_copy_pad / kf_padv / kf_padh / kf_max_vh remain ported and
//      ALG-VERIFIED; they are simply not launched by this plugin because
//      those kernels address their planes through an interior origin
//      (negative indices) which a whole-buffer cl_mem argument cannot
//      express on this driver (sub-buffers are broken,
//      docs/RIG_HANDOFF_BRINGUP.md §9.4).
//   4. The QP-table margins outside the per-plane written region
//      ((w+15)>>3 x (h+15)>>3 inside the calc_qp_size frame) are ZERO-FILLED
//      each frame; upstream leaves them undefined (NewVideoFrame garbage).
//      No consumer reads them (kf_deblock / kf_deblock_show /
//      kf_sharpen_coeff stay inside the written region; the sharpen coeff
//      bilinear taps reach at most ((w-1)>>3)+1 <= (w+15)>>3), so this only
//      replaces undefined values with deterministic ones.
//   5. When QP_Table_Non_B is absent (QP_TABLE_NONE) upstream returns a
//      table frame whose planes are NEVER written (undefined memory) and the
//      sharpen tail then reads that undefined table.  This port zero-fills
//      the table planes in that case (constant-QP 0 semantics for the
//      sharpen tail; KDeblock itself passes through unchanged either way).
//   6. Upstream's GetDeviceTypes CUDA device-match checks are CUDA-only and
//      not reproduced (both clips go through this plugin's one queue).
//   7. kf_sharpen / kf_show_sharpen_coeff read the coeff plane through
//      manual float32 bilinear (kf_sharpen_bilinear, the CPU twins' verbatim
//      expression) instead of upstream's CUDA TEXTURE with 16-bit-unorm
//      quantization.  This is the port's pinned semantics (kernel + goldens
//      are the authority, docs/RIG_HANDOFF_KDEBLOCK.md §4.5/§4.6): vs real
//      CUDA the accepted divergence is the characterised ±1 class at
//      truncation boundaries (HW quantizes to round_half_up(v*257)/65535;
//      rig/CUDA_REPORT_kdeblock_texture.md: 7/28,840 px1 + 428/28,040 px2
//      for kf_sharpen, 0 px for kf_show_sharpen_coeff, plus 1 px of
//      upstream's own FMA contraction).  Documented, accepted, not
//      reproduced.
//   8. Pad rows beyond h+16: the padded plane buffers are 8-rounded
//      ((h+7+16)&~7 rows) but the mirror pad fills exactly h+16 rows; the
//      rows in between are ZERO-FILLED each frame (device staging and the
//      in-exe reference both).  kf_deblock reads up to row bh*8+5 for the
//      last block row (by=bh-1, ty offset 6, +7 rows), which exceeds h+15
//      whenever bh is odd (bh*8+5 > h+15); upstream's kl_copy_pad fills
//      exactly h+16 rows and upstream CUDA then reads UNINITIALIZED rows
//      there — undefined upstream, defined-as-zero here.  (Without the
//      zero-fill the device staging write itself read past the staging
//      vector — fixed; see deblockPlaneT.)
//
// GAPS (upstream AddFuncDeblock members NOT registered by this DLL — none of
// them is a KDeblock device kernel):
//   - QPClip("c"): pure host/props pass-through (copies the source's QP
//     props onto a 2x2 Y8 MAGIC_KEY frame); no device kernel upstream.
//   - ShowQP("c[nonb]b[dc]b"): QP-table viewer (kf_scale_qp exists in the
//     .cl, ALG-VERIFIED, but the filter's frame assembly reads QP props and
//     DrawTexts "No QP table ..." fallbacks; a debug view, not part of the
//     KDeblock pipeline).
//   - FrameType("c"): CPU-only run-time environment function (reads
//     current_frame + the FrameType prop; no device work).
//
// The plugin is a plain DLL exporting AvisynthPluginInit3 (no import lib);
// it validates autoload + registration against the installed AviSynth+ 3.7.3
// in test_kfm_deblock.cpp.
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "avisynth.h"
#include "../ocl_host.h"
#include "kfm_deblock_cl.h"  // GENERATED from src/opencl/kfm/kernels/kfm_deblock.cl
#include "kfm_textout_font.h"

namespace {

using oclhost::Buffer;
using oclhost::Context;
using oclhost::Program;

// upstream QP_RESULT_FLAG (Deblock.cu)
enum {
    QP_TABLE_NONE = 0,
    QP_TABLE_ONLY = 1,
    QP_TABLE_CONSTANT = 2,
    QP_TABLE_USING_DC = 3,
};

// upstream norm_qscale (verbatim)
int normQscale(int qscale, int type) {
    switch (type) {
    case 0: return qscale << 2;
    case 1: return qscale << 1;
    case 2: return qscale;
    case 3: return (63 - qscale + 2);
    }
    return qscale;
}

// upstream calc_qp_size (QPForDeblock ctor, verbatim)
int calcQpSize(int len) {
    int a = (len + 7 + 8) >> 3;
    int b = (((len >> 1) + 7 + 8) >> 3) * 2;
    return a > b ? a : b;
}

// upstream Get16BitType (the planar YUV family this port supports; YV411 has
// no 16-bit AviSynth+ type and is rejected in CreateKDeblockOCL)
int pixelType16(int pt) {
    switch (pt) {
    case VideoInfo::CS_YV12: return VideoInfo::CS_YUV420P16;
    case VideoInfo::CS_YV16: return VideoInfo::CS_YUV422P16;
    case VideoInfo::CS_YV24: return VideoInfo::CS_YUV444P16;
    default: return pt;  // already 16-bit (checked by the caller)
    }
}

// One plain plane: a w*h device image, pitch == width (KDebandOCL Plane
// discipline); px = element size in bytes (1 = uchar, 2 = ushort).
struct Plane {
    Buffer buf;
    int w = 0, h = 0, px = 1;
    size_t bytes() const { return (size_t)w * h * px; }
    void alloc(Context& ctx, int w_, int h_, int px_) {
        w = w_; h = h_; px = px_;
        buf = Buffer(ctx, bytes(), /*zero=*/false);
    }
    void fillZero(Context& ctx) {
        buf.fill32(0);
        size_t tail = bytes() & 3;
        if (tail) {  // fill32 clears whole words; sweep the sub-word tail
            static const uint8_t z[3] = { 0, 0, 0 };
            buf.write(z, bytes() - tail, tail);
        }
        ctx.finish();
    }
    cl_mem mem() const { return buf.mem(); }
};

void copyFromFrame(const uint8_t* srcp, int srcPitch, int rowsize, int height, uint8_t* dst,
                   int dstPitch) {
    for (int y = 0; y < height; ++y)
        memcpy(dst + (size_t)y * dstPitch, srcp + (size_t)y * srcPitch, rowsize);
}

void copyToFrame(const uint8_t* srcp, int srcPitch, int rowsize, int height, uint8_t* dstp,
                 int dstPitch) {
    for (int y = 0; y < height; ++y)
        memcpy(dstp + (size_t)y * dstPitch, srcp + (size_t)y * srcPitch, rowsize);
}

Context& sharedContext() {
    static Context ctx;  // thread-safe magic static; one device/queue per process
    return ctx;
}

// ONE shared queue mutex for the whole KDeblock chain (KDeblockOCL,
// SharpenFilterOCL and the QP-forDeblock helper all submit to the same
// in-order queue; the chain is sequential per frame, but different frames
// can run on different threads in MT mode).
std::mutex& queueMutex() {
    static std::mutex mtx;
    return mtx;
}

// kf_copy_pad / kf_mirror_idx mirror index (verbatim semantics)
int mirrorIdx(int x, int width) {
    if (x < 0) return -x - 1;
    if (x >= width) return 2 * width - x - 1;
    return x;
}

// Host-staging mirror pad: build the (w+2*pad)x(h+2*pad) plane with `src`
// (w x h, srcPitch elements) at the interior.  Bit-identical to the device
// kf_copy_pad / Copy+padv+padh path (DEVIATION 3).
template <typename T>
void stageMirrorPad(std::vector<T>& dst, const T* srcp, int srcPitch, int w, int h, int pad) {
    const int pw = w + 2 * pad, ph = h + 2 * pad;
    dst.resize((size_t)pw * ph);
    for (int y = 0; y < ph; ++y) {
        int sy = mirrorIdx(y - pad, h);
        for (int x = 0; x < pw; ++x) {
            int sx = mirrorIdx(x - pad, w);
            dst[(size_t)y * pw + x] = srcp[(size_t)sy * srcPitch + sx];
        }
    }
}

// MakeMask (upstream kl_max_vh<5> over the 8px-padded DC plane): radius-5
// box max, seeded at 0 (upstream uint8_t sum = 0).  Host staging form
// (DEVIATION 3); returns the mask interior (w x h, pitch w).
void makeMaskCpu(std::vector<uint8_t>& mask, const uint8_t* dcp, int dcPitch, int w, int h) {
    std::vector<uint8_t> padded;
    stageMirrorPad(padded, dcp, dcPitch, w, h, 8);
    const int pw = w + 16;
    const int org = 8 + 8 * pw;
    mask.resize((size_t)w * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t best = 0;
            for (int j = -5; j <= 5; ++j)
                for (int i = -5; i <= 5; ++i) {
                    uint8_t v = padded[(size_t)(org + (x + i) + (y + j) * pw)];
                    if (v > best) best = v;
                }
            mask[(size_t)y * w + x] = best;
        }
    }
}

// ---------------------------------------------------------------------------
// cpu_draw_text / DrawText port (upstream TextOut.cu, bit-exact): glyph cell
// 10x20 from kfmtext::kFont; luma set 235<<shift, chroma set 128<<shift;
// unset pixels dimmed (dst + (isUV ? 128<<shift : 0)) >> 1.
// ---------------------------------------------------------------------------
template <typename T>
void drawTextPlane(T* dst, int width, int height, int pitch,
                   const char* charmap, int mapw, int maph, int mappitch,
                   int logx, int logy, int pixelshift, bool isUV) {
    for (int by = 0; by < maph; ++by) {
        for (int bx = 0; bx < mapw; ++bx) {
            for (int ty = 0; ty < 20; ty += (1 << logy)) {
                for (int tx = 0; tx < 10; tx += (1 << logx)) {
                    int x = (tx + bx * 10) >> logx;
                    int y = (ty + by * 20) >> logy;
                    if (x < width && y < height) {
                        int c = charmap[bx + by * mappitch] - ' ';
                        if (c >= 0 && c <= 'z' - ' ') {
                            if (kfmtext::kFont[c][ty] & (1 << (15 - tx))) {
                                dst[(size_t)x + (size_t)y * pitch] =
                                    (T)(isUV ? (128 << pixelshift) : (235 << pixelshift));
                            } else {
                                dst[(size_t)x + (size_t)y * pitch] =
                                    (T)(((int)dst[(size_t)x + (size_t)y * pitch] +
                                         (isUV ? (128 << pixelshift) : 0)) >> 1);
                            }
                        }
                    }
                }
            }
        }
    }
}

// upstream DrawText (single-line form; the flag messages have no '\n')
template <typename T>
void drawTextMessage(PVideoFrame& f, int bits, const std::string& msg) {
    int maxlen = (int)msg.size();
    if (maxlen <= 0) return;
    std::vector<char> charmap((size_t)maxlen, 0);
    memcpy(charmap.data(), msg.data(), msg.size());

    T* dstY = (T*)f->GetWritePtr(PLANAR_Y);
    T* dstU = (T*)f->GetWritePtr(PLANAR_U);
    T* dstV = (T*)f->GetWritePtr(PLANAR_V);
    int width = f->GetRowSize(PLANAR_Y) / (int)sizeof(T);
    if (width > maxlen * 10) width = maxlen * 10;
    int height = f->GetHeight(PLANAR_Y);
    if (height > 20) height = 20;
    int logx = (f->GetRowSize(PLANAR_U) / (int)sizeof(T) < width) ? 1 : 0;
    int logy = (f->GetHeight(PLANAR_U) < height) ? 1 : 0;
    int shift = bits - 8;
    drawTextPlane(dstY, width, height, f->GetPitch(PLANAR_Y) / (int)sizeof(T),
                  charmap.data(), maxlen, 1, maxlen, 0, 0, shift, false);
    drawTextPlane(dstU, width >> logx, height >> logy, f->GetPitch(PLANAR_U) / (int)sizeof(T),
                  charmap.data(), maxlen, 1, maxlen, logx, logy, shift, true);
    drawTextPlane(dstV, width >> logx, height >> logy, f->GetPitch(PLANAR_U) / (int)sizeof(T),
                  charmap.data(), maxlen, 1, maxlen, logx, logy, shift, true);
}

// ---------------------------------------------------------------------------
// QPForDeblockOCL — upstream QPForDeblock host assembly (QP table build).
// Produces the uint16 table planes + the DEBLOCK_QP_FLAG per frame.
// ---------------------------------------------------------------------------
struct QPData {
    int flag = QP_TABLE_NONE;
    Plane tableY, tableU, tableV;  // uint16 planes, dims calcQpSize (chroma subsampled)
};

class QPForDeblockOCL {
public:
    VideoInfo srcvi;
    PClip qpclip;
    float frameRateConv = 1.0f;  // (qpclip fps) / (source fps)
    float b_ratio;
    int force_qp;
    bool b_adap;

    int logUVx = 0, logUVy = 0;
    int qw = 0, qh = 0;  // table frame dims (calc_qp_size)

    QPForDeblockOCL(PClip source, float b_ratio_, PClip qpclip_, int force_qp_, bool b_adap_)
        : qpclip(qpclip_), b_ratio(b_ratio_), force_qp(force_qp_), b_adap(b_adap_) {
        srcvi = source->GetVideoInfo();
        logUVx = srcvi.GetPlaneWidthSubsampling(PLANAR_U);
        logUVy = srcvi.GetPlaneHeightSubsampling(PLANAR_U);
        if (qpclip) {
            const VideoInfo& qpvi = qpclip->GetVideoInfo();
            frameRateConv = (float)((int64_t)qpvi.fps_numerator * srcvi.fps_denominator) /
                            (float)((int64_t)qpvi.fps_denominator * srcvi.fps_numerator);
        } else {
            frameRateConv = 1.0f;
        }
        qw = calcQpSize(srcvi.width);
        qh = calcQpSize(srcvi.height);
    }

    // ---- per-frame compute (upstream GetFrame + QPEntry + QPForPlane) ----
    QPData compute(int n, IScriptEnvironment* env) {
        Context& ctx = sharedContext();
        QPData out;
        out.tableY.alloc(ctx, qw, qh, 2);
        out.tableU.alloc(ctx, qw >> logUVx, qh >> logUVy, 2);
        out.tableV.alloc(ctx, qw >> logUVx, qh >> logUVy, 2);
        // zero-fill: the kernels write only the used region (DEVIATIONS 4/5)
        out.tableY.fillZero(ctx);
        out.tableU.fillZero(ctx);
        out.tableV.fillZero(ctx);

        if (force_qp > 0) {
            // QP forced: constant table (upstream QPEntry(nullptr, ...))
            qpForPlane(out.tableY, srcvi.width, srcvi.height,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       0, 0, 1, 1, ctx);
            qpForPlane(out.tableU, srcvi.width >> logUVx, srcvi.height >> logUVy,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       0, 0, 1 - logUVx, 1 - logUVy, ctx);
            qpForPlane(out.tableV, srcvi.width >> logUVx, srcvi.height >> logUVy,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       0, 0, 1 - logUVx, 1 - logUVy, ctx);
            out.flag = QP_TABLE_CONSTANT;
            ctx.finish();
            return out;
        }

        PVideoFrame src = srcClip->GetFrame(n, env);

        // frame-number mapping via the KFM_SourceStart props
        int error;
        const AVSMap* avsmap = env->getFramePropsRO(src);
        env->propGetInt(avsmap, "KFM_SourceStart", 0, &error);  // existence check
        if (!error) {
            int start = (int)env->propGetInt(avsmap, "KFM_SourceStart", 0, &error);
            int num = (int)env->propGetInt(avsmap, "KFM_NumSourceFrames", 0, &error);
            int qp_n = (int)(frameRateConv * n + 0.3f);
            if (start < qp_n - 10 || start > qp_n + 10)
                env->ThrowError("Invalid KFM_SourceStart");
            PVideoFrame qp0 = getQpFrame(start, env);
            PVideoFrame qp1;
            if (num > 1) qp1 = getQpFrame(start + 1, env);
            qpEntry(out, qp0, qp1, env, ctx);
            ctx.finish();
            return out;
        }

        if (qpclip) {
            int qp_n = (int)(frameRateConv * n + 0.3f);
            PVideoFrame qp0 = getQpFrame(qp_n, env);
            qpEntry(out, qp0, PVideoFrame(), env, ctx);
            ctx.finish();
            return out;
        }

        // QP props on the source frame itself
        qpEntry(out, src, PVideoFrame(), env, ctx);
        ctx.finish();
        return out;
    }

    PClip srcClip;  // the source clip (set by KDeblockOCL after construction)

private:
    Program& program() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "kfm_deblock", kfm_deblock_cl_src, 1);
    }

    PVideoFrame getQpFrame(int n, IScriptEnvironment* env) {
        if (qpclip) return qpclip->GetFrame(n, env);
        return srcClip->GetFrame(n, env);
    }

    // upstream QPEntry(qp0, qp1)
    void qpEntry(QPData& out, PVideoFrame qp0, PVideoFrame qp1,
                 IScriptEnvironment* env, Context& ctx) {
        const AVSMap* avsmap = env->getFramePropsRO(qp0);
        env->propGetInt(avsmap, "QP_Table_Non_B", 0, &error_);  // existence check
        if (error_) {
            // QP table unavailable (DEVIATION 5: planes stay zero-filled)
            out.flag = QP_TABLE_NONE;
            return;
        }
        PVideoFrame qpTable0 = env->propGetFrame(avsmap, "QP_Table", 0, &error_);
        PVideoFrame qpNonB0 = env->propGetFrame(avsmap, "QP_Table_Non_B", 0, &error_);
        int qpStride = (int)env->propGetInt(avsmap, "QP_Stride", 0, &error_);
        int qpScaleType = (int)env->propGetInt(avsmap, "QP_ScaleType", 0, &error_);
        PVideoFrame dc0;
        if (b_adap) dc0 = env->propGetFrame(avsmap, "DC_Table", 0, &error_);

        PVideoFrame qpTable1, qpNonB1, dc1;
        if (qp1) {
            const AVSMap* avsmap1 = env->getFramePropsRO(qp1);
            qpTable1 = env->propGetFrame(avsmap1, "QP_Table", 0, &error_);
            qpNonB1 = env->propGetFrame(avsmap1, "QP_Table_Non_B", 0, &error_);
            if (b_adap) dc1 = env->propGetFrame(avsmap1, "DC_Table", 0, &error_);
        }

        // MakeMask per DC table (host staging form — DEVIATION 3)
        std::vector<uint8_t> mask0, mask1;
        Plane mask0p, mask1p;
        if (dc0) {
            int mw = dc0->GetRowSize(), mh = dc0->GetHeight();
            makeMaskCpu(mask0, dc0->GetReadPtr(), dc0->GetPitch(), mw, mh);
            mask0p.alloc(ctx, mw, mh, 1);
            mask0p.buf.write(mask0.data(), 0, mask0p.bytes());
        }
        if (dc1) {
            int mw = dc1->GetRowSize(), mh = dc1->GetHeight();
            makeMaskCpu(mask1, dc1->GetReadPtr(), dc1->GetPitch(), mw, mh);
            mask1p.alloc(ctx, mw, mh, 1);
            mask1p.buf.write(mask1.data(), 0, mask1p.bytes());
        }

        qpForPlane(out.tableY, srcvi.width, srcvi.height,
                   qpTable0->GetReadPtr(), qpNonB0->GetReadPtr(),
                   dc0 ? &mask0p : nullptr,
                   qp1 ? qpTable1->GetReadPtr() : nullptr,
                   qp1 ? qpNonB1->GetReadPtr() : nullptr,
                   dc1 ? &mask1p : nullptr,
                   qpStride, qpScaleType, 1, 1, ctx);
        qpForPlane(out.tableU, srcvi.width >> logUVx, srcvi.height >> logUVy,
                   qpTable0->GetReadPtr(), qpNonB0->GetReadPtr(),
                   dc0 ? &mask0p : nullptr,
                   qp1 ? qpTable1->GetReadPtr() : nullptr,
                   qp1 ? qpNonB1->GetReadPtr() : nullptr,
                   dc1 ? &mask1p : nullptr,
                   qpStride, qpScaleType, 1 - logUVx, 1 - logUVy, ctx);
        qpForPlane(out.tableV, srcvi.width >> logUVx, srcvi.height >> logUVy,
                   qpTable0->GetReadPtr(), qpNonB0->GetReadPtr(),
                   dc0 ? &mask0p : nullptr,
                   qp1 ? qpTable1->GetReadPtr() : nullptr,
                   qp1 ? qpNonB1->GetReadPtr() : nullptr,
                   dc1 ? &mask1p : nullptr,
                   qpStride, qpScaleType, 1 - logUVx, 1 - logUVy, ctx);

        out.flag = dc0 ? QP_TABLE_USING_DC : QP_TABLE_ONLY;
    }

    // upstream QPForPlane: one kf_make_qp_table launch
    void qpForPlane(Plane& dst, int width, int height,
                    const uint8_t* qpTable0, const uint8_t* qpNonB0, Plane* bmask0,
                    const uint8_t* qpTable1, const uint8_t* qpNonB1, Plane* bmask1,
                    int qpStride, int qpScaleType, int qpShiftX, int qpShiftY,
                    Context& ctx) {
        float dc_coeff = b_ratio / 255.0f;
        int qp_width = (width + 7 + 8) >> 3;
        int qp_height = (height + 7 + 8) >> 3;

        // staging planes for the macroblock QP sources
        int mbw = (srcvi.width + 15) >> 4;
        int mbh = (srcvi.height + 15) >> 4;
        Plane in0, nb0, in1, nb1;
        if (qpTable0) {
            uploadTable(in0, qpTable0, qpStride, mbw, mbh, ctx);
            uploadTable(nb0, qpNonB0, qpStride, mbw, mbh, ctx);
        }
        if (qpTable1) {
            uploadTable(in1, qpTable1, qpStride, mbw, mbh, ctx);
            uploadTable(nb1, qpNonB1, qpStride, mbw, mbh, ctx);
        }

        cl_kernel k = program().kernel("kf_make_qp_table");
        int a = 0;
        oclhost::set_arg(k, a++, mbw);
        oclhost::set_arg(k, a++, mbh);
        oclhost::set_arg(k, a++, qpTable0 ? in0.mem() : (cl_mem)nullptr);
        oclhost::set_arg(k, a++, qpTable0 ? nb0.mem() : (cl_mem)nullptr);
        oclhost::set_arg(k, a++, qpTable1 ? 1 : 0);
        oclhost::set_arg(k, a++, qpTable1 ? in1.mem() : (cl_mem)nullptr);
        oclhost::set_arg(k, a++, qpTable1 ? nb1.mem() : (cl_mem)nullptr);
        oclhost::set_arg(k, a++, mbw);  // in_pitch (staged plane pitch)
        oclhost::set_arg(k, a++, qpTable0 ? qpScaleType : force_qp);
        oclhost::set_arg(k, a++, dc_coeff);
        oclhost::set_arg(k, a++, bmask0 ? 1 : 0);
        oclhost::set_arg(k, a++, bmask0 ? bmask0->mem() : (cl_mem)nullptr);
        oclhost::set_arg(k, a++, bmask1 ? 1 : 0);
        oclhost::set_arg(k, a++, bmask1 ? bmask1->mem() : (cl_mem)nullptr);
        oclhost::set_arg(k, a++, bmask0 ? bmask0->w : 0);  // dc_pitch
        oclhost::set_arg(k, a++, qpShiftX);
        oclhost::set_arg(k, a++, qpShiftY);
        oclhost::set_arg(k, a++, qp_width);
        oclhost::set_arg(k, a++, qp_height);
        oclhost::set_arg(k, a++, dst.mem());
        oclhost::set_arg(k, a++, dst.w);  // out_pitch
        oclhost::dispatch2D(ctx, k, qp_width, qp_height);
    }

    void uploadTable(Plane& p, const uint8_t* hostp, int stride, int mbw, int mbh,
                     Context& ctx) {
        p.alloc(ctx, mbw, mbh, 1);
        stage_.resize((size_t)mbw * mbh);
        for (int y = 0; y < mbh; ++y)
            memcpy(stage_.data() + (size_t)y * mbw, hostp + (size_t)y * stride, mbw);
        p.buf.write(stage_.data(), 0, p.bytes());
    }

    std::vector<uint8_t> stage_;
    int error_ = 0;
};

// ---------------------------------------------------------------------------
// KDeblockOCL — upstream KDeblock (DeblockEntry + DeblockPlane).
// ---------------------------------------------------------------------------
class KDeblockOCL : public GenericVideoFilter {
public:
    std::shared_ptr<QPForDeblockOCL> qp;
    int quality;
    float strength;
    int show;
    float thresh_a;
    float thresh_b;

    int logUVx, logUVy;
    int bits, px;
    int deblockShift, deblockMaxV, mergeShift, mergeMaxV, count;

    // device scratch (per-call, overwritten every GetFrame)
    Plane padY_, padU_, padV_;
    Plane accY_, accU_, accV_;
    Plane dstY_, dstU_, dstV_;

    std::vector<uint8_t> stageB_;    // 8-bit staging (max plane)
    std::vector<uint16_t> stageS_;   // 16-bit staging (max plane)
    std::vector<uint8_t> padB_;      // padded-plane staging, 8-bit
    std::vector<uint16_t> padS_;     // padded-plane staging, 16-bit
    std::mutex* mtx_;

    KDeblockOCL(PClip source, std::shared_ptr<QPForDeblockOCL> qp_,
                int quality_, float strength_, float qp_thresh, int show_,
                IScriptEnvironment* env)
        : GenericVideoFilter(source), qp(qp_), quality(quality_), strength(strength_),
          show(show_) {
        logUVx = vi.GetPlaneWidthSubsampling(PLANAR_U);
        logUVy = vi.GetPlaneHeightSubsampling(PLANAR_U);
        bits = vi.BitsPerComponent();
        px = vi.ComponentSize();

        const float W = 5;
        thresh_a = (qp_thresh + W) / (2 * W);
        thresh_b = (W * W - qp_thresh * qp_thresh) / (2 * W);

        int dshift = quality + bits - 10;
        if (dshift < 0) dshift = 0;
        deblockShift = dshift;
        deblockMaxV = (1 << (bits + 6 - deblockShift)) - 1;
        mergeShift = quality + 6 - deblockShift;
        mergeMaxV = (1 << bits) - 1;
        count = 1 << quality;

        mtx_ = &queueMutex();

        try {
            Context& ctx = sharedContext();
            const int w = vi.width, h = vi.height;
            const int wUV = w >> logUVx, hUV = h >> logUVy;
            const int pw = (w + 7 + 8 * 2) & ~7, ph = (h + 7 + 8 * 2) & ~7;
            const int pwUV = (wUV + 7 + 8 * 2) & ~7, phUV = (hUV + 7 + 8 * 2) & ~7;

            padY_.alloc(ctx, pw, ph, px);
            padU_.alloc(ctx, pwUV, phUV, px);
            padV_.alloc(ctx, pwUV, phUV, px);
            // accumulators: upstream tmpOut dims (pw x ph*4 ushorts); the buffer
            // ORIGIN is the upstream interior (+8,+8) — DEVIATION 1.
            accY_.alloc(ctx, pw, ph * 4, 2);
            accU_.alloc(ctx, pwUV, phUV * 4, 2);
            accV_.alloc(ctx, pwUV, phUV * 4, 2);
            dstY_.alloc(ctx, w, h, px);
            dstU_.alloc(ctx, wUV, hUV, px);
            dstV_.alloc(ctx, wUV, hUV, px);
            stageB_.resize((size_t)w * h * px);
            stageS_.resize((size_t)w * h);
            padB_.resize((size_t)pw * ph);
            padS_.resize((size_t)pw * ph);
            program();  // build the PX instantiation up front (HOST_CONTRACT §1)
        } catch (const std::exception& e) {
            env->ThrowError("[KDeblockOCL] OpenCL init failed: %s", e.what());
        }
    }

    Program& program() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "kfm_deblock", kfm_deblock_cl_src, px);
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            return GetFrameT(n, env_);
        } catch (const std::exception& e) {
            env_->ThrowError("[KDeblockOCL] %s", e.what());
        }
        return PVideoFrame();
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) {
        // mirror upstream KDeblock::SetCacheHints (MT_NICE_FILTER); device work
        // is serialised through the shared in-order queue + plugin mutex.
        if (cachehints == CACHE_GET_MTMODE) return MT_NICE_FILTER;
        return 0;
    }

private:
    // upstream DeblockPlane for one plane (verbatim sequencing)
    template <typename T>
    void deblockPlaneT(Plane& dst, Plane& pad, Plane& acc, Plane& qpTab,
                       int width, int height, const T* srcp, int srcPitch,
                       std::vector<T>& padStage) {
        const int pw = pad.w, ph = pad.h;
        const int bw = (width + 7 + 8) >> 3;
        const int bh = (height + 7 + 8) >> 3;
        Context& ctx = sharedContext();

        // ---- pad (upstream kl_copy_pad): host staging mirror pad ----
        stageMirrorPad<T>(padStage, srcp, srcPitch, width, height, 8);
        // The pad buffer is the 8-rounded plane (pw x ph) but the mirror pad
        // only fills (w+16) x (h+16) rows; zero-fill the remainder so the
        // rows between h+16 and ph that kf_deblock can read for the last
        // block row (by=bh-1 reads up to row bh*8+5 > h+15 for odd bh) are
        // DEFINED zeros on the device (upstream leaves them uninitialized —
        // kl_copy_pad fills exactly h+16 rows; DEVIATION 6).
        padStage.resize((size_t)pw * ph, 0);
        pad.buf.write(padStage.data(), 0, pad.bytes());

        if (show == 2) {
            cl_kernel k = program().kernel("kf_deblock_show");
            int a = 0;
            oclhost::set_arg(k, a++, dst.mem());
            oclhost::set_arg(k, a++, dst.w);
            oclhost::set_arg(k, a++, width);
            oclhost::set_arg(k, a++, height);
            oclhost::set_arg(k, a++, bw);
            oclhost::set_arg(k, a++, bh);
            oclhost::set_arg(k, a++, qpTab.mem());
            oclhost::set_arg(k, a++, qpTab.w);
            oclhost::set_arg(k, a++, thresh_a);
            oclhost::set_arg(k, a++, thresh_b);
            oclhost::dispatch2D(ctx, k, width, height);
            return;
        }

        {
            cl_kernel k = program().kernel("kf_deblock");
            int a = 0;
            oclhost::set_arg(k, a++, pad.mem());
            oclhost::set_arg(k, a++, pw);  // src_pitch
            oclhost::set_arg(k, a++, bw);
            oclhost::set_arg(k, a++, bh);
            oclhost::set_arg(k, a++, acc.mem());
            oclhost::set_arg(k, a++, pw);  // out_pitch (ushort units)
            oclhost::set_arg(k, a++, qpTab.mem());
            oclhost::set_arg(k, a++, qpTab.w);
            oclhost::set_arg(k, a++, count - 1);
            oclhost::set_arg(k, a++, deblockShift);
            oclhost::set_arg(k, a++, deblockMaxV);
            oclhost::set_arg(k, a++, strength);
            oclhost::set_arg(k, a++, thresh_a);
            oclhost::set_arg(k, a++, thresh_b);
            // DETERMINISM FIX (parity-plane overlap): for odd bh >= 3 the z=0
            // plane's last tile row (even-bx, by=bh-1, offy=(0+bh-1)*8, rows
            // (bh-1)*8 .. bh*8+8) overlaps the z=1 plane's first tile (odd-bx,
            // by=0, offy=bh*8, rows bh*8 .. bh*8+16) in rows bh*8..bh*8+8 of
            // the shared accumulator.  Two work items write the same cells;
            // upstream CUDA has the identical write race (kl_deblock's plain
            // ushort2 stores), while upstream's non-AVX CPU resolves it by
            // loop order (for by: for bx => the by=bh-1 / z=0 tile wins).
            // Reproduce that deterministically: launch the by=0 row first,
            // then by>=1 in a second in-order dispatch, so the by=bh-1 tiles
            // always write last.  Within each launch no two work items share
            // accumulator cells (same-z tiles are 16px apart in x and
            // 2*bh*8 >= 48px in y; the only cross-z overlap involves by=0,
            // which is fully in launch 1).
            oclhost::dispatch2DOffset(ctx, k, 0, 0, bw, 1);
            if (bh > 1) oclhost::dispatch2DOffset(ctx, k, 0, 1, bw, bh - 1);
        }
        {
            cl_kernel k = program().kernel("kf_merge_deblock");
            int a = 0;
            oclhost::set_arg(k, a++, acc.mem());
            oclhost::set_arg(k, a++, pw >> 2);  // tmp_pitch_u4
            oclhost::set_arg(k, a++, bh * 8);   // tmp_ipitch_rows
            oclhost::set_arg(k, a++, dst.mem());
            oclhost::set_arg(k, a++, dst.w);    // out_pitch
            oclhost::set_arg(k, a++, width);    // vis_width
            oclhost::set_arg(k, a++, height);   // vis_height
            oclhost::set_arg(k, a++, mergeShift);
            oclhost::set_arg(k, a++, (float)mergeMaxV);
            oclhost::dispatch2D(ctx, k, width, height);
        }
    }

    void deblockPlane(Plane& dst, Plane& pad, Plane& acc, Plane& qpTab,
                      int width, int height, const uint8_t* srcp, int srcPitch) {
        if (px == 1)
            deblockPlaneT<uint8_t>(dst, pad, acc, qpTab, width, height, srcp, srcPitch, padB_);
        else
            deblockPlaneT<uint16_t>(dst, pad, acc, qpTab, width, height,
                                    (const uint16_t*)srcp, srcPitch / 2, padS_);
    }

    PVideoFrame GetFrameT(int n, IScriptEnvironment* env_) {
        const int w = vi.width, h = vi.height;
        const int wUV = w >> logUVx, hUV = h >> logUVy;

        std::lock_guard<std::mutex> lock(*mtx_);

        QPData qpd = qp->compute(n, env_);

        PVideoFrame srcFrame = child->GetFrame(n, env_);

        if (qpd.flag == QP_TABLE_NONE) {
            // upstream: pass the source through (with the flag message when
            // show; upstream DrawText MakeWritable's the source frame in
            // place — this port draws on a fresh copy, same pixels).
            PVideoFrame out = env_->NewVideoFrame(vi);
            copyFromFrame(srcFrame->GetReadPtr(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y),
                          w * px, h, stageB_.data(), w * px);
            copyToFrame(stageB_.data(), w * px, w * px, h,
                        out->GetWritePtr(PLANAR_Y), out->GetPitch(PLANAR_Y));
            copyFromFrame(srcFrame->GetReadPtr(PLANAR_U), srcFrame->GetPitch(PLANAR_U),
                          wUV * px, hUV, stageB_.data(), wUV * px);
            copyToFrame(stageB_.data(), wUV * px, wUV * px, hUV,
                        out->GetWritePtr(PLANAR_U), out->GetPitch(PLANAR_U));
            copyFromFrame(srcFrame->GetReadPtr(PLANAR_V), srcFrame->GetPitch(PLANAR_V),
                          wUV * px, hUV, stageB_.data(), wUV * px);
            copyToFrame(stageB_.data(), wUV * px, wUV * px, hUV,
                        out->GetWritePtr(PLANAR_V), out->GetPitch(PLANAR_V));
            if (show) {
                if (px == 1) drawTextMessage<uint8_t>(out, bits, "KDeblock: Disabled");
                else drawTextMessage<uint16_t>(out, bits, "KDeblock: Disabled");
            }
            return out;
        }

        // ---- deblock the three planes into dstY_/dstU_/dstV_ ----
        // zero the accumulators: kf_merge_deblock reads the parity-slice cells
        // that kf_deblock leaves unwritten (top/left border rows); upstream
        // zero-fills its tmpOut the same way — deterministic output requires it
        accY_.fillZero(sharedContext());
        accU_.fillZero(sharedContext());
        accV_.fillZero(sharedContext());
        deblockPlane(dstY_, padY_, accY_, qpd.tableY, w, h,
                     srcFrame->GetReadPtr(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y));
        deblockPlane(dstU_, padU_, accU_, qpd.tableU, wUV, hUV,
                     srcFrame->GetReadPtr(PLANAR_U), srcFrame->GetPitch(PLANAR_U));
        deblockPlane(dstV_, padV_, accV_, qpd.tableV, wUV, hUV,
                     srcFrame->GetReadPtr(PLANAR_V), srcFrame->GetPitch(PLANAR_V));

        sharedContext().finish();

        // ---- download into a fresh output frame ----
        PVideoFrame dstFrame = env_->NewVideoFrame(vi);
        dstY_.buf.read(stageB_.data(), 0, dstY_.bytes());
        copyToFrame(stageB_.data(), w * px, w * px, h,
                    dstFrame->GetWritePtr(PLANAR_Y), dstFrame->GetPitch(PLANAR_Y));
        dstU_.buf.read(stageB_.data(), 0, dstU_.bytes());
        copyToFrame(stageB_.data(), wUV * px, wUV * px, hUV,
                    dstFrame->GetWritePtr(PLANAR_U), dstFrame->GetPitch(PLANAR_U));
        dstV_.buf.read(stageB_.data(), 0, dstV_.bytes());
        copyToFrame(stageB_.data(), wUV * px, wUV * px, hUV,
                    dstFrame->GetWritePtr(PLANAR_V), dstFrame->GetPitch(PLANAR_V));

        if (show == 1) {
            const char* msg;
            switch (qpd.flag) {
            case QP_TABLE_ONLY: msg = "KDeblock: Using QP table"; break;
            case QP_TABLE_CONSTANT: msg = "KDeblock: Constant QP"; break;
            case QP_TABLE_USING_DC: msg = "KDeblock: Using QP table with DC"; break;
            default: msg = "KDeblock: ???"; break;
            }
            if (px == 1) drawTextMessage<uint8_t>(dstFrame, bits, msg);
            else drawTextMessage<uint16_t>(dstFrame, bits, msg);
        }
        return dstFrame;
    }
};

// ---------------------------------------------------------------------------
// SharpenFilterOCL — upstream SharpenFilter (the sharp=true tail).
// ---------------------------------------------------------------------------
class SharpenFilterOCL : public GenericVideoFilter {
public:
    std::shared_ptr<QPForDeblockOCL> qp;
    PClip unsharpclip;
    bool showCoeff;  // upstream `show` (== show==3 of the KDeblock arg)

    int logUVx, logUVy;
    int px;
    VideoInfo coeffvi;  // QP-table dims, 8-bit

    Plane srcY_, srcU_, srcV_;       // staged deblocked frame
    Plane unsharpY_, unsharpU_, unsharpV_;
    Plane coeffY_, coeffU_, coeffV_;
    Plane dstY_, dstU_, dstV_;

    std::vector<uint8_t> stageB_;
    std::vector<uint16_t> stageS_;
    std::mutex* mtx_ = &queueMutex();

    SharpenFilterOCL(PClip source, std::shared_ptr<QPForDeblockOCL> qp_,
                     PClip unsharp, bool showCoeff_, IScriptEnvironment* env)
        : GenericVideoFilter(source), qp(qp_), unsharpclip(unsharp),
          showCoeff(showCoeff_) {
        logUVx = vi.GetPlaneWidthSubsampling(PLANAR_U);
        logUVy = vi.GetPlaneHeightSubsampling(PLANAR_U);
        px = vi.ComponentSize();

        coeffvi = qp->srcvi;  // upstream: coeffvi = qpclip->GetVideoInfo()
        coeffvi.pixel_type = pixelType16(coeffvi.pixel_type);  // (dims only matter)
        try {
            Context& ctx = sharedContext();
            const int w = vi.width, h = vi.height;
            const int wUV = w >> logUVx, hUV = h >> logUVy;
            srcY_.alloc(ctx, w, h, px);
            srcU_.alloc(ctx, wUV, hUV, px);
            srcV_.alloc(ctx, wUV, hUV, px);
            unsharpY_.alloc(ctx, w, h, px);
            unsharpU_.alloc(ctx, wUV, hUV, px);
            unsharpV_.alloc(ctx, wUV, hUV, px);
            coeffY_.alloc(ctx, qp->qw, qp->qh, 1);
            coeffU_.alloc(ctx, qp->qw >> logUVx, qp->qh >> logUVy, 1);
            coeffV_.alloc(ctx, qp->qw >> logUVx, qp->qh >> logUVy, 1);
            dstY_.alloc(ctx, w, h, px);
            dstU_.alloc(ctx, wUV, hUV, px);
            dstV_.alloc(ctx, wUV, hUV, px);
            stageB_.resize((size_t)w * h * px);
            stageS_.resize((size_t)w * h);
            program();  // build the PX instantiation up front (HOST_CONTRACT §1)
        } catch (const std::exception& e) {
            env->ThrowError("[SharpenFilterOCL] OpenCL init failed: %s", e.what());
        }
    }

    Program& program() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "kfm_deblock", kfm_deblock_cl_src, px);
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            return GetFrameT(n, env_);
        } catch (const std::exception& e) {
            env_->ThrowError("[SharpenFilterOCL] %s", e.what());
        }
        return PVideoFrame();
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) {
        if (cachehints == CACHE_GET_MTMODE) return MT_NICE_FILTER;
        return 0;
    }

private:
    // upstream SharpenFilter::GetFrameT + ProcPlane per plane
    template <typename T>
    void procPlaneT(Plane& dst, Plane& src, Plane& unsharp, Plane& coeff, Plane& qpTab,
                    int width, int height, const T* srcp, int srcPitch,
                    const T* usp, int usPitch, std::vector<T>& stage) {
        int qpw = coeff.w;
        int qph = coeff.h;
        Context& ctx = sharedContext();

        // stage the deblocked src + unsharp planes
        stage.resize((size_t)src.w * src.h);
        for (int y = 0; y < src.h; ++y)
            memcpy(stage.data() + (size_t)y * src.w, srcp + (size_t)y * srcPitch,
                   src.w * sizeof(T));
        src.buf.write(stage.data(), 0, src.bytes());
        stage.resize((size_t)unsharp.w * unsharp.h);
        for (int y = 0; y < unsharp.h; ++y)
            memcpy(stage.data() + (size_t)y * unsharp.w, usp + (size_t)y * usPitch,
                   unsharp.w * sizeof(T));
        unsharp.buf.write(stage.data(), 0, unsharp.bytes());

        // qp -> coeff (upstream kl_sharpen_coeff)
        {
            cl_kernel k = program().kernel("kf_sharpen_coeff");
            int a = 0;
            oclhost::set_arg(k, a++, coeff.mem());
            oclhost::set_arg(k, a++, qpw);
            oclhost::set_arg(k, a++, qph);
            oclhost::set_arg(k, a++, qpw);  // coeff pitch
            oclhost::set_arg(k, a++, qpTab.mem());
            oclhost::set_arg(k, a++, qpTab.w);
            oclhost::dispatch2D(ctx, k, qpw, qph);
        }

        if (showCoeff) {
            cl_kernel k = program().kernel("kf_show_sharpen_coeff");
            int a = 0;
            oclhost::set_arg(k, a++, dst.mem());
            oclhost::set_arg(k, a++, width);
            oclhost::set_arg(k, a++, height);
            oclhost::set_arg(k, a++, dst.w);
            oclhost::set_arg(k, a++, coeff.mem());
            oclhost::set_arg(k, a++, qpw);  // coeff pitch
            oclhost::dispatch2D(ctx, k, width, height);
        } else {
            cl_kernel k = program().kernel("kf_sharpen");
            int a = 0;
            oclhost::set_arg(k, a++, dst.mem());
            oclhost::set_arg(k, a++, width);
            oclhost::set_arg(k, a++, height);
            oclhost::set_arg(k, a++, dst.w);
            oclhost::set_arg(k, a++, src.mem());
            oclhost::set_arg(k, a++, src.w);  // src_pitch
            oclhost::set_arg(k, a++, coeff.mem());
            oclhost::set_arg(k, a++, qpw);    // coeff pitch
            oclhost::set_arg(k, a++, unsharp.mem());
            oclhost::dispatch2D(ctx, k, width, height);
        }
    }

    void procPlane(Plane& dst, Plane& src, Plane& unsharp, Plane& coeff, Plane& qpTab,
                   int width, int height, const uint8_t* srcp, int srcPitch,
                   const uint8_t* usp, int usPitch) {
        if (px == 1)
            procPlaneT<uint8_t>(dst, src, unsharp, coeff, qpTab, width, height,
                                srcp, srcPitch, usp, usPitch, stageB_);
        else
            procPlaneT<uint16_t>(dst, src, unsharp, coeff, qpTab, width, height,
                                 (const uint16_t*)srcp, srcPitch / 2,
                                 (const uint16_t*)usp, usPitch / 2, stageS_);
    }

    PVideoFrame GetFrameT(int n, IScriptEnvironment* env_) {
        const int w = vi.width, h = vi.height;
        const int wUV = w >> logUVx, hUV = h >> logUVy;

        // fetch the child (deblocked) and unsharp frames BEFORE taking the
        // shared queue mutex — the child chain (Cache -> GaussResize ->
        // KDeblockOCL) locks the same mutex itself
        PVideoFrame srcFrame = child->GetFrame(n, env_);       // deblocked frame
        PVideoFrame unsharpFrame = unsharpclip->GetFrame(n, env_);

        std::lock_guard<std::mutex> lock(*mtx_);

        QPData qpd = qp->compute(n, env_);

        procPlane(dstY_, srcY_, unsharpY_, coeffY_, qpd.tableY, w, h,
                  srcFrame->GetReadPtr(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y),
                  unsharpFrame->GetReadPtr(PLANAR_Y), unsharpFrame->GetPitch(PLANAR_Y));
        procPlane(dstU_, srcU_, unsharpU_, coeffU_, qpd.tableU, wUV, hUV,
                  srcFrame->GetReadPtr(PLANAR_U), srcFrame->GetPitch(PLANAR_U),
                  unsharpFrame->GetReadPtr(PLANAR_U), unsharpFrame->GetPitch(PLANAR_U));
        procPlane(dstV_, srcV_, unsharpV_, coeffV_, qpd.tableV, wUV, hUV,
                  srcFrame->GetReadPtr(PLANAR_V), srcFrame->GetPitch(PLANAR_V),
                  unsharpFrame->GetReadPtr(PLANAR_V), unsharpFrame->GetPitch(PLANAR_V));

        sharedContext().finish();

        PVideoFrame dstFrame = env_->NewVideoFrame(vi);
        dstY_.buf.read(stageB_.data(), 0, dstY_.bytes());
        copyToFrame(stageB_.data(), w * px, w * px, h,
                    dstFrame->GetWritePtr(PLANAR_Y), dstFrame->GetPitch(PLANAR_Y));
        dstU_.buf.read(stageB_.data(), 0, dstU_.bytes());
        copyToFrame(stageB_.data(), wUV * px, wUV * px, hUV,
                    dstFrame->GetWritePtr(PLANAR_U), dstFrame->GetPitch(PLANAR_U));
        dstV_.buf.read(stageB_.data(), 0, dstV_.bytes());
        copyToFrame(stageB_.data(), wUV * px, wUV * px, hUV,
                    dstFrame->GetWritePtr(PLANAR_V), dstFrame->GetPitch(PLANAR_V));
        return dstFrame;
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// registration
// ---------------------------------------------------------------------------
AVSValue __cdecl CreateKDeblockOCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        PClip source = args[0].AsClip();
        const VideoInfo& vi = source->GetVideoInfo();
        // upstream ctor checks (QPForDeblock + KDeblock, verbatim semantics)
        if (!vi.IsPlanar() || !vi.IsYUV() || vi.IsYUVA())
            env->ThrowError("[KDeblockOCL] planar YUV input required");
        if (vi.width & 7)
            env->ThrowError("[KDeblock]: width must be multiple of 8");
        if (vi.height & 7)
            env->ThrowError("[KDeblock]: height must be multiple of 8");
        // upstream GetFrame dispatch: only ComponentSize 1/2 are implemented
        if (vi.ComponentSize() != 1 && vi.ComponentSize() != 2)
            env->ThrowError("[KDeblock] Unsupported pixel format");
        if (vi.pixel_type == VideoInfo::CS_YV411)
            env->ThrowError("[KDeblock] Unsupported pixel format");

        int show = (int)args[9].AsInt(0);
        bool b_adap = args[7].AsBool(true);
        auto qpf = std::make_shared<QPForDeblockOCL>(
            source,
            (float)args[4].AsFloat(b_adap ? 2.0f : 0.5f),  // bratio
            args[5].AsClip(),                              // qpclip
            (int)args[6].AsInt(-1),                        // qp (force_qp)
            b_adap);                                       // badap
        // upstream: QPForDeblock's child is the source clip; KDeblock's is too.
        qpf->srcClip = source;

        KDeblockOCL* deblock = new KDeblockOCL(
            source,                    // source
            qpf,
            (int)args[1].AsInt(3),     // quality
            (float)args[2].AsFloat(0), // str
            (float)args[3].AsFloat(0), // thr
            show,                      // show
            env);
        PClip clip = deblock;

        if (args[8].AsBool(false)) {  // sharp
            // upstream: Cache -> GaussResize -> SharpenFilter (the REAL core
            // GaussResize, invoked exactly like upstream)
            AVSValue cached = env->Invoke("Cache", AVSValue(clip));
            clip = cached.AsClip();
            VideoInfo cvi = clip->GetVideoInfo();
            AVSValue gargs[8] = { cached, cvi.width, cvi.height,
                                  0, 0,
                                  (double)cvi.width + 0.0001,
                                  (double)cvi.height + 0.0001,
                                  2 };
            AVSValue unsharp = env->Invoke("GaussResize", AVSValue(gargs, 8));
            clip = new SharpenFilterOCL(clip, qpf, unsharp.AsClip(), show == 3, env);
        }
        return clip;
    } catch (const std::exception& e) {
        env->ThrowError("[KDeblockOCL] %s", e.what());
    }
    return AVSValue();
}

// AviSynth+ linkage: the core passes the function-table pointer as the 2nd
// argument of AvisynthPluginInit3 (no avisynth.lib import needed).
const AVS_Linkage* AVS_linkage{ nullptr };

extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
    AVS_linkage = vectors;
    try {
        // exact upstream signature from Deblock.cu AddFuncDeblock
        env->AddFunction("KDeblockOCL",
                         "c[quality]i[str]f[thr]f[bratio]f[qpclip]c[qp]i[badap]b[sharp]b[show]i",
                         CreateKDeblockOCL, 0);
        // upstream function name as an alias (upstream signature identical)
        env->AddFunction("KDeblock",
                         "c[quality]i[str]f[thr]f[bratio]f[qpclip]c[qp]i[badap]b[sharp]b[show]i",
                         CreateKDeblockOCL, 0);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "KDeblock (OpenCL port of KFM/Deblock.cu KDeblock, MIT; sixth host plugin of the OpenCL port)";
}
