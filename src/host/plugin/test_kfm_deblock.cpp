// test_kfm_deblock.cpp 窶・headless verification of the KDeblockOCL AviSynth+
// plugin (sixth host plugin of the OpenCL port; KFM/Deblock.cu KDeblock +
// QPForDeblock + SharpenFilter tail) against a CPU reference computed in this
// exe.
//
// REFERENCE CHOICE (stated, per the task): the reference is computed HERE, in
// this exe, from the actual input frame pixels and the actual QP-table prop
// frames, transcribing the repo's CPU-mirror/golden semantics 窶・the
// golden-pinned authority the device kernels are ALG-VERIFIED against:
//   - sim/kfm_deblock_ref.cpp      (kf_deblock + kf_merge_deblock core),
//   - sim/kfm_deblock_qp_ref.cpp   (kf_make_qp_table + kf_deblock_show),
//   - sim/kfm_deblock_aux_ref.cpp  (kf_sharpen_coeff / kf_sharpen (device
//     form incl. the min(x+1,height-1) quirk + manual float32 bilinear) /
//     kf_show_sharpen_coeff),
// plus the host-staging twins documented in the plugin header (mirror pad,
// radius-5 box-max MakeMask, upstream norm_qscale / calc_qp_size /
// thresh_a/b, and the bit-exact DrawText port over kfm_textout_font.h).
// All arithmetic is float32 with no FMA (MSVC /fp:precise default), matching
// the .cl's FP_CONTRACT OFF and the goldens.
//
// Coverage (2 frames per case, EVERY pixel of EVERY plane compared):
//   - formats YV12/YV16/YV24 (8-bit) + YUV420P16/YUV422P16/YUV444P16,
//   - QP sources: defaults (QP_TABLE_NONE passthrough), props-on-source
//     frames, qpclip, forced qp (QP_TABLE_CONSTANT), qpclip with DC_Table +
//     badap (QP_TABLE_USING_DC), badap=false, qpclip WITHOUT props (NONE),
//     KFM_SourceStart (num=1 and num=2, frameRateConv != 1, invalid throw),
//   - parameter sweep: quality 0/1/2/6, str/thr boundaries, bratio, show
//     0/1/2/3, sharp=true (incl. the 64x16 quirk config),
//   - alias ("KDeblock") invocation, upstream ctor validation throws
//     (width%8, height%8, non-planar-YUV).
//
// Load path: the exe copies the plugin DLL into the AviSynth+ autoload
// directory (registry HKLM\Software\AviSynth plugins64), creates the script
// environment, verifies autoload via FunctionExists, then invokes
// KDeblockOCL (and the upstream KDeblock alias) and compares every pixel.
//
// Build: see build_host_deblock.bat.  Usage: test_kfm_deblock.exe
// <plugin_dll_path> [plugins_dir]
#include <windows.h>

#include "avisynth.h"
#include "kfm_textout_font.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

typedef IScriptEnvironment* (__stdcall* CSE_t)(int);
const AVS_Linkage* AVS_linkage = nullptr;

static LONG WINAPI eh(EXCEPTION_POINTERS* ep) {
    HMODULE m = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)ep->ExceptionRecord->ExceptionAddress, &m);
    char path[MAX_PATH] = "?";
    GetModuleFileNameA(m, path, MAX_PATH);
    fprintf(stderr, "EXC %08lX at %p in %s\n", ep->ExceptionRecord->ExceptionCode,
            ep->ExceptionRecord->ExceptionAddress, path);
    fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}

static int g_failures = 0;
static int g_cases = 0;
static int g_frame_checks = 0;

// ---------------------------------------------------------------------------
// deterministic source content
// ---------------------------------------------------------------------------
static uint32_t mix32(uint32_t a) {
    a ^= a >> 16; a *= 0x7feb352du;
    a ^= a >> 15; a *= 0x846ca68bu;
    a ^= a >> 16;
    return a;
}

enum SourceKind {
    SRC_SMOOTH,  // smooth 2D gradient
    SRC_NOISY,   // full-range per-pixel noise
    SRC_BLOCKY,  // 8x8 blocky pattern (deblock-relevant)
    SRC_MIXED,   // 16x16 noisy/flat blocks
};

static int patternV(int x, int y, int w, int h, uint32_t salt, SourceKind kind) {
    int v;
    switch (kind) {
    case SRC_SMOOTH:
        v = (int)(((int64_t)x * 127) / (w > 1 ? w - 1 : 1)
                  + ((int64_t)y * 127) / (h > 1 ? h - 1 : 1));
        break;
    case SRC_NOISY:
        v = (int)(mix32((uint32_t)x * 17u + (uint32_t)y * 1013u + salt) & 255u);
        break;
    case SRC_BLOCKY:
        v = ((x / 8) + (y / 8)) & 1
            ? (int)(mix32((uint32_t)(x / 8) * 131u + (uint32_t)(y / 8) * 77u + salt) % 90u) + 40
            : 128 + ((int)(mix32(salt ^ ((uint32_t)x * 3u + (uint32_t)y * 5u)) % 15u) - 7);
        break;
    default:  // SRC_MIXED
        if (((x / 16) + (y / 16)) & 1)
            v = (int)(mix32((uint32_t)x * 31u + (uint32_t)y * 7u + salt) & 255u);
        else
            v = 128 + ((int)(mix32(salt ^ ((uint32_t)x * 3u + (uint32_t)y * 5u)) % 11u) - 5);
        break;
    }
    return v < 0 ? 0 : v > 255 ? 255 : v;
}

template <typename T>
static void fillPatternT(T* dstp, int pitch, int w, int h, uint32_t salt,
                         SourceKind kind, int px) {
    const int scale = px == 2 ? 257 : 1;
    for (int y = 0; y < h; ++y) {
        T* row = dstp + (size_t)y * pitch;
        for (int x = 0; x < w; ++x)
            row[x] = (T)(patternV(x, y, w, h, salt, kind) * scale);
    }
}

// The GetFrame into the source clips needs an IScriptEnvironment; pass it
// through a file-scope pointer (single-threaded test exe).
static IScriptEnvironment* g_env = nullptr;

// ---------------------------------------------------------------------------
// QP-table props machinery
// ---------------------------------------------------------------------------
// Deterministic QP-table bytes: upstream hands KDeblock QP_Table /
// QP_Table_Non_B (and DC_Table when badap) as Y8 frames in the frame props.
// mbw x mbh macroblock-sized frames, QP_Stride = the frame's real pitch.
static void makeQpBytes(std::vector<uint8_t>& in, std::vector<uint8_t>& nonb,
                        std::vector<uint8_t>& dc, int mbw, int mbh, int n,
                        bool withDC) {
    in.resize((size_t)mbw * mbh);
    nonb.resize((size_t)mbw * mbh);
    dc.clear();
    if (withDC) dc.resize((size_t)mbw * mbh);
    for (int y = 0; y < mbh; ++y) {
        for (int x = 0; x < mbw; ++x) {
            size_t i = (size_t)y * mbw + x;
            in[i] = (uint8_t)(2 + mix32((uint32_t)n * 977u + (uint32_t)x * 31u +
                                        (uint32_t)y * 173u + 0xABCDu) % 24u);
            nonb[i] = (uint8_t)(2 + mix32((uint32_t)n * 733u + (uint32_t)x * 57u +
                                          (uint32_t)y * 219u + 0xABCDu + 5u) % 24u);
            if (withDC)
                dc[i] = (uint8_t)(mix32((uint32_t)n * 311u + (uint32_t)x * 91u +
                                        (uint32_t)y * 47u + 0xABCDu + 9u) & 255u);
        }
    }
}

static VideoInfo makeVi8(int w, int h, int numFrames) {
    VideoInfo vi;
    memset(&vi, 0, sizeof vi);
    vi.width = w; vi.height = h;
    vi.pixel_type = VideoInfo::CS_Y8;
    vi.num_frames = numFrames;
    vi.fps_numerator = 1; vi.fps_denominator = 1;
    return vi;
}

// attach the QP props upstream-style to a frame (QP_Table / QP_Table_Non_B /
// QP_Stride / QP_ScaleType [/ DC_Table])
static void attachQpProps(IScriptEnvironment* env, PVideoFrame f, const VideoInfo& vi8,
                          int mbw, int mbh, int n, bool withDC, int scaleType) {
    std::vector<uint8_t> in, nonb, dc;
    makeQpBytes(in, nonb, dc, mbw, mbh, n, withDC);
    PVideoFrame tab = env->NewVideoFrame(vi8);
    PVideoFrame nonbf = env->NewVideoFrame(vi8);
    for (int y = 0; y < mbh; ++y) {
        memcpy(tab->GetWritePtr(PLANAR_Y) + (size_t)y * tab->GetPitch(PLANAR_Y),
               in.data() + (size_t)y * mbw, mbw);
        memcpy(nonbf->GetWritePtr(PLANAR_Y) + (size_t)y * nonbf->GetPitch(PLANAR_Y),
               nonb.data() + (size_t)y * mbw, mbw);
    }
    AVSMap* map = env->getFramePropsRW(f);
    env->propSetFrame(map, "QP_Table", tab, 0);
    env->propSetFrame(map, "QP_Table_Non_B", nonbf, 0);
    env->propSetInt(map, "QP_Stride", tab->GetPitch(PLANAR_Y), 0);
    env->propSetInt(map, "QP_ScaleType", scaleType, 0);
    if (withDC) {
        PVideoFrame dcf = env->NewVideoFrame(vi8);
        for (int y = 0; y < mbh; ++y)
            memcpy(dcf->GetWritePtr(PLANAR_Y) + (size_t)y * dcf->GetPitch(PLANAR_Y),
                   dc.data() + (size_t)y * mbw, mbw);
        env->propSetFrame(map, "DC_Table", dcf, 0);
    }
}

enum QPMode {
    QP_NONE,         // no qpclip, no props          -> QP_TABLE_NONE
    QP_ONFRAME,      // QP props on the source frame -> QP_TABLE_ONLY
    QP_ONFRAME_DC,   // + DC_Table, badap            -> QP_TABLE_USING_DC
    QP_CLIP,         // qpclip with props            -> QP_TABLE_ONLY
    QP_CLIP_DC,      // qpclip with DC_Table + badap -> QP_TABLE_USING_DC
    QP_FORCE,        // qp > 0                       -> QP_TABLE_CONSTANT
    QP_SRCSTART,     // KFM_SourceStart on source frames + qpclip
    QP_SRCSTART2,    // num = 2 (element-max of two QP frames)
    QP_SRCSTART_BAD, // start out of the +-10 window -> GetFrame throws
    QP_BADCLIP,      // qpclip WITHOUT props         -> QP_TABLE_NONE
};

// generic source clip with optional QP props / KFM_SourceStart props
class TestSource : public IClip {
public:
    VideoInfo vi_;
    uint32_t salt_;
    SourceKind kind_;
    QPMode qpm_;
    bool withDC_;
    int scaleType_;
    float conv_;      // frameRateConv used for the KFM_SourceStart mapping
    VideoInfo vi8_;
    int mbw_, mbh_;

    TestSource(VideoInfo vi, uint32_t salt, SourceKind kind, QPMode qpm,
               bool withDC, int scaleType, float conv, int mbw, int mbh)
        : vi_(vi), salt_(salt), kind_(kind), qpm_(qpm), withDC_(withDC),
          scaleType_(scaleType), conv_(conv), mbw_(mbw), mbh_(mbh) {
        vi8_ = makeVi8(mbw, mbh, vi.num_frames + 2);
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) {
        PVideoFrame f = env->NewVideoFrame(vi_);
        const int w = vi_.width, h = vi_.height;
        const int lx = vi_.GetPlaneWidthSubsampling(PLANAR_U);
        const int ly = vi_.GetPlaneHeightSubsampling(PLANAR_U);
        const int wUV = w >> lx, hUV = h >> ly;
        const int px = vi_.ComponentSize();
        uint32_t s = salt_ + (uint32_t)n * 7919u;
        if (px == 1) {
            fillPatternT<uint8_t>((uint8_t*)f->GetWritePtr(PLANAR_Y), f->GetPitch(PLANAR_Y), w, h, s, kind_, 1);
            fillPatternT<uint8_t>((uint8_t*)f->GetWritePtr(PLANAR_U), f->GetPitch(PLANAR_U), wUV, hUV, s + 11u, kind_, 1);
            fillPatternT<uint8_t>((uint8_t*)f->GetWritePtr(PLANAR_V), f->GetPitch(PLANAR_V), wUV, hUV, s + 23u, kind_, 1);
        } else {
            fillPatternT<uint16_t>((uint16_t*)f->GetWritePtr(PLANAR_Y), f->GetPitch(PLANAR_Y) / 2, w, h, s, kind_, 2);
            fillPatternT<uint16_t>((uint16_t*)f->GetWritePtr(PLANAR_U), f->GetPitch(PLANAR_U) / 2, wUV, hUV, s + 11u, kind_, 2);
            fillPatternT<uint16_t>((uint16_t*)f->GetWritePtr(PLANAR_V), f->GetPitch(PLANAR_V) / 2, wUV, hUV, s + 23u, kind_, 2);
        }
        switch (qpm_) {
        case QP_ONFRAME:
        case QP_ONFRAME_DC:
            attachQpProps(env, f, vi8_, mbw_, mbh_, n, withDC_, scaleType_);
            break;
        case QP_SRCSTART:
        case QP_SRCSTART2:
        case QP_SRCSTART_BAD: {
            int qp_n = (int)(conv_ * n + 0.3f);
            int start = qp_n + (qpm_ == QP_SRCSTART_BAD ? 11 : 0);
            AVSMap* map = env->getFramePropsRW(f);
            env->propSetInt(map, "KFM_SourceStart", start, 0);
            env->propSetInt(map, "KFM_NumSourceFrames", qpm_ == QP_SRCSTART2 ? 2 : 1, 0);
            break;
        }
        default:
            break;
        }
        return f;
    }
    bool __stdcall GetParity(int) { return false; }
    void __stdcall GetAudio(void*, int64_t, int64_t, IScriptEnvironment*) {}
    int __stdcall SetCacheHints(int, int) { return 0; }
    const VideoInfo& __stdcall GetVideoInfo() { return vi_; }
};

// qpclip: carrier frames carrying QP props (their own pixels unused)
class QPTableSource : public IClip {
public:
    VideoInfo vi_;   // carrier vi (dims/fps matter: fps drives frameRateConv)
    VideoInfo vi8_;
    int mbw_, mbh_;
    bool withDC_;
    int scaleType_;

    QPTableSource(VideoInfo carrierVi, int mbw, int mbh, bool withDC, int scaleType)
        : vi_(carrierVi), mbw_(mbw), mbh_(mbh), withDC_(withDC), scaleType_(scaleType) {
        vi8_ = makeVi8(mbw, mbh, carrierVi.num_frames + 2);
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) {
        PVideoFrame f = env->NewVideoFrame(vi_);
        attachQpProps(env, f, vi8_, mbw_, mbh_, n, withDC_, scaleType_);
        return f;
    }
    bool __stdcall GetParity(int) { return false; }
    void __stdcall GetAudio(void*, int64_t, int64_t, IScriptEnvironment*) {}
    int __stdcall SetCacheHints(int, int) { return 0; }
    const VideoInfo& __stdcall GetVideoInfo() { return vi_; }
};

// plain source WITHOUT props (QP_BADCLIP / defaults)
class PlainSource : public IClip {
public:
    VideoInfo vi_;
    uint32_t salt_;
    SourceKind kind_;
    PlainSource(VideoInfo vi, uint32_t salt, SourceKind kind)
        : vi_(vi), salt_(salt), kind_(kind) {}
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) {
        PVideoFrame f = env->NewVideoFrame(vi_);
        const int w = vi_.width, h = vi_.height;
        const int lx = vi_.GetPlaneWidthSubsampling(PLANAR_U);
        const int ly = vi_.GetPlaneHeightSubsampling(PLANAR_U);
        const int wUV = w >> lx, hUV = h >> ly;
        const int px = vi_.ComponentSize();
        uint32_t s = salt_ + (uint32_t)n * 7919u;
        if (px == 1) {
            fillPatternT<uint8_t>((uint8_t*)f->GetWritePtr(PLANAR_Y), f->GetPitch(PLANAR_Y), w, h, s, kind_, 1);
            fillPatternT<uint8_t>((uint8_t*)f->GetWritePtr(PLANAR_U), f->GetPitch(PLANAR_U), wUV, hUV, s + 11u, kind_, 1);
            fillPatternT<uint8_t>((uint8_t*)f->GetWritePtr(PLANAR_V), f->GetPitch(PLANAR_V), wUV, hUV, s + 23u, kind_, 1);
        } else {
            fillPatternT<uint16_t>((uint16_t*)f->GetWritePtr(PLANAR_Y), f->GetPitch(PLANAR_Y) / 2, w, h, s, kind_, 2);
            fillPatternT<uint16_t>((uint16_t*)f->GetWritePtr(PLANAR_U), f->GetPitch(PLANAR_U) / 2, wUV, hUV, s + 11u, kind_, 2);
            fillPatternT<uint16_t>((uint16_t*)f->GetWritePtr(PLANAR_V), f->GetPitch(PLANAR_V) / 2, wUV, hUV, s + 23u, kind_, 2);
        }
        return f;
    }
    bool __stdcall GetParity(int) { return false; }
    void __stdcall GetAudio(void*, int64_t, int64_t, IScriptEnvironment*) {}
    int __stdcall SetCacheHints(int, int) { return 0; }
    const VideoInfo& __stdcall GetVideoInfo() { return vi_; }
};

// ---------------------------------------------------------------------------
// CPU reference 窶・golden semantics (see file header)
// ---------------------------------------------------------------------------
static int norm_qscale(int qscale, int type) {
    switch (type) {
    case 0: return qscale << 2;
    case 1: return qscale << 1;
    case 2: return qscale;
    case 3: return (63 - qscale + 2);
    }
    return qscale;
}

static int calc_qp_size(int len) {
    int a = (len + 7 + 8) >> 3;
    int b = (((len >> 1) + 7 + 8) >> 3) * 2;
    return a > b ? a : b;
}

static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static float qp_thresh(int qp, float ta, float tb) { return clampf((float)qp * ta + tb, 0.0f, (float)qp); }

// ---- DCT/IDCT + offset grid (sim/kfm_deblock_ref.cpp verbatim) ----
#define RS1   0.19509032201612825f
#define RC1   0.9807852804032304f
#define RS3   0.5555702330196022f
#define RC3   0.8314696123025452f
#define RS2S6 1.3065629648763766f
#define RS2C6 0.5411961001461971f
#define RS2   1.4142135623730951f

static const int g_offx[127] = {
  0,0,4, 0,2,6,4, 0,5,2,7,4,1,6,3, 0,4,1,5,3,7,2,6,0,4,1,5,3,7,2,6,
  0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,
  0,4,0,4,2,6,2,6,0,4,0,4,2,6,2,6,1,5,1,5,3,7,3,7,1,5,1,5,3,7,3,7,
  0,4,0,4,2,6,2,6,0,4,0,4,2,6,2,6,1,5,1,5,3,7,3,7,1,5,1,5,3,7,3,7,
};
static const int g_offy[127] = {
  0,0,4, 0,2,4,6, 0,1,2,3,4,5,6,7, 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,
  0,2,4,6,1,3,5,7,0,2,4,6,1,3,5,7,0,2,4,6,1,3,5,7,0,2,4,6,1,3,5,7,
  0,4,4,0,2,6,6,2,2,6,6,2,0,4,4,0,1,5,5,1,3,7,7,3,3,7,7,3,1,5,5,1,
  1,5,5,1,3,7,7,3,3,7,7,3,1,5,5,1,0,4,4,0,2,6,6,2,2,6,6,2,0,4,4,0,
};

static void dct8(float* d, int stride) {
    float a0 = d[7 * stride] + d[0 * stride];
    float a1 = d[6 * stride] + d[1 * stride];
    float a2 = d[5 * stride] + d[2 * stride];
    float a3 = d[4 * stride] + d[3 * stride];
    float a4 = d[3 * stride] - d[4 * stride];
    float a5 = d[2 * stride] - d[5 * stride];
    float a6 = d[1 * stride] - d[6 * stride];
    float a7 = d[0 * stride] - d[7 * stride];
    float b0 = a3 + a0, b1 = a2 + a1, b2 = a1 - a2, b3 = a0 - a3;
    float b4 = (RS3 - RC3) * a7 + RC3 * (a4 + a7);
    float b5 = (RS1 - RC1) * a6 + RC1 * (a5 + a6);
    float b6 = -(RC1 + RS1) * a5 + RC1 * (a5 + a6);
    float b7 = -(RC3 + RS3) * a4 + RC3 * (a4 + a7);
    float c0 = b1 + b0, c1 = b0 - b1;
    float c2 = (RS2S6 - RS2C6) * b3 + RS2C6 * (b2 + b3);
    float c3 = -(RS2C6 + RS2S6) * b2 + RS2C6 * (b2 + b3);
    float c4 = b6 + b4, c5 = b7 - b5, c6 = b4 - b6, c7 = b5 + b7;
    float d4 = c7 - c4, d5 = c5 * RS2, d6 = c6 * RS2, d7 = c4 + c7;
    d[0 * stride] = c0; d[4 * stride] = c1; d[2 * stride] = c2; d[6 * stride] = c3;
    d[7 * stride] = d4; d[3 * stride] = d5; d[5 * stride] = d6; d[1 * stride] = d7;
}
static void idct8(float* d, int stride) {
    float c0 = d[0 * stride], c1 = d[4 * stride], c2 = d[2 * stride], c3 = d[6 * stride];
    float d4 = d[7 * stride], d5 = d[3 * stride], d6 = d[5 * stride], d7 = d[1 * stride];
    float c4 = d7 - d4, c5 = d5 * RS2, c6 = d6 * RS2, c7 = d4 + d7;
    float b0 = c1 + c0, b1 = c0 - c1;
    float b2 = -(RS2C6 + RS2S6) * c3 + RS2C6 * (c2 + c3);
    float b3 = (RS2S6 - RS2C6) * c2 + RS2C6 * (c2 + c3);
    float b4 = c6 + c4, b5 = c7 - c5, b6 = c4 - c6, b7 = c5 + c7;
    float a0 = b3 + b0, a1 = b2 + b1, a2 = b1 - b2, a3 = b0 - b3;
    float a4 = -(RC3 + RS3) * b7 + RC3 * (b4 + b7);
    float a5 = -(RC1 + RS1) * b6 + RC1 * (b5 + b6);
    float a6 = (RS1 - RC1) * b5 + RC1 * (b5 + b6);
    float a7 = (RS3 - RC3) * b4 + RC3 * (b4 + b7);
    d[0 * stride] = a7 + a0; d[1 * stride] = a6 + a1; d[2 * stride] = a5 + a2; d[3 * stride] = a4 + a3;
    d[4 * stride] = a3 - a4; d[5 * stride] = a2 - a5; d[6 * stride] = a1 - a6; d[7 * stride] = a0 - a7;
}
static void dct8x8(float* d) { for (int i = 0; i < 8; i++) dct8(d + i * 8, 1); for (int i = 0; i < 8; i++) dct8(d + i, 8); }
static void idct8x8(float* d) { for (int i = 0; i < 8; i++) idct8(d + i, 8); for (int i = 0; i < 8; i++) idct8(d + i * 8, 1); }
static void hardthresh(float* d, float th) {
    for (int i = 1; i < 64; i++) { if (d[i] < -th || d[i] > th) continue; d[i] = 0.0f; }
}

static int mirrorIdx(int x, int width) {
    if (x < 0) return -x - 1;
    if (x >= width) return 2 * width - x - 1;
    return x;
}

template <typename T>
static void stageMirrorPad(std::vector<T>& dst, const T* srcp, int srcPitch, int w, int h, int pad) {
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

// MakeMask (mirror pad + radius-5 box max; plugin DEVIATION 3 / upstream kl_max_vh<5>)
static void makeMaskCpu(std::vector<uint8_t>& mask, const uint8_t* dcp, int dcPitch, int w, int h) {
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

// kf_make_qp_table twin (sim/kfm_deblock_qp_ref.cpp verbatim)
static void ref_make_qp_table(
    int in_width, int in_height,
    const uint8_t* in_table0, const uint8_t* nonb_table0,
    bool has_table1, const uint8_t* in_table1, const uint8_t* nonb_table1,
    int in_pitch, int qp_scale, float dc_coeff,
    bool has_dc0, const uint8_t* dc_table0,
    bool has_dc1, const uint8_t* dc_table1, int dc_pitch,
    int qp_shift_x, int qp_shift_y, int out_width, int out_height,
    uint16_t* out_table, int out_pitch) {
    for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            int qp;
            if (in_table0) {
                int qx = x >> qp_shift_x, qy = y >> qp_shift_y;
                int qp_x = qx < in_width - 1 ? qx : in_width - 1;
                int qp_y = qy < in_height - 1 ? qy : in_height - 1;
                int in_qp = (int)in_table0[qp_x + qp_y * in_pitch];
                int nonb_qp = (int)nonb_table0[qp_x + qp_y * in_pitch];
                int dc = has_dc0 ? (int)dc_table0[qp_x + qp_y * dc_pitch] : 255;
                if (has_table1) {
                    int v = (int)in_table1[qp_x + qp_y * in_pitch];
                    in_qp = in_qp > v ? in_qp : v;
                    v = (int)nonb_table1[qp_x + qp_y * in_pitch];
                    nonb_qp = nonb_qp > v ? nonb_qp : v;
                    v = has_dc1 ? (int)dc_table1[qp_x + qp_y * dc_pitch] : 255;
                    dc = dc > v ? dc : v;
                }
                int b = norm_qscale(in_qp, qp_scale);
                int nonb = norm_qscale(nonb_qp, qp_scale);
                float b_ratio = (float)dc * dc_coeff < 1.0f ? (float)dc * dc_coeff : 1.0f;
                int acc = (int)(b * b_ratio + nonb * (1.0f - b_ratio) + 0.5f);
                qp = acc > 1 ? acc : 1;
            } else {
                qp = qp_scale;
            }
            out_table[x + y * out_pitch] = (uint16_t)qp;
        }
    }
}

// kf_deblock_show twin (sim/kfm_deblock_qp_ref.cpp verbatim)
template <typename T>
static void ref_deblock_show(int width, int height, int dst_pitch, T* dst,
                             const uint16_t* qp_table, int qp_pitch,
                             float thresh_a, float thresh_b) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int bx = (x + 4) >> 3;
            int by = (y + 4) >> 3;
            int qp = (int)qp_table[bx + by * qp_pitch];
            int enabled = (qp_thresh(qp, thresh_a, thresh_b) >= (int)(qp >> 1));
            dst[x + y * dst_pitch] = (T)(enabled ? 230 : 16);
        }
    }
}

// g_ldither (sim/kfm_deblock_aux_ref.cpp; indexed [y&7][X&1][L], X = x>>2)
static const int LDITHER[8][2][4] = {
  { {  0,  48,  12,  60 }, {  3,  51,  15,  63 } },
  { { 32,  16,  44,  28 }, { 35,  19,  47,  31 } },
  { {  8,  56,   4,  52 }, { 11,  59,   7,  55 } },
  { { 40,  24,  36,  20 }, { 43,  27,  39,  23 } },
  { {  2,  50,  14,  62 }, {  1,  49,  13,  61 } },
  { { 34,  18,  46,  30 }, { 33,  17,  45,  29 } },
  { { 10,  58,   6,  54 }, {  9,  57,   5,  53 } },
  { { 42,  26,  38,  22 }, { 41,  25,  37,  21 } },
};

static const int SHARPEN_COEFF[30] = {
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 10,
    50, 90, 120, 150, 160,
    170, 180, 190, 200, 210,
    220, 230, 240, 245, 250,
    255, 255, 255, 255, 255,
};

// kf_sharpen_bilinear twin (manual float32 bilinear 窶・the port's pinned form)
static float ref_sharpen_bilinear(const uint8_t* coeff, int coeff_pitch, float x, float y) {
    float fx = x * (1.0f / 8.0f), fy = y * (1.0f / 8.0f);
    int ix = (int)fx, iy = (int)fy;
    float c00 = (float)coeff[ix + iy * coeff_pitch];
    float c01 = (float)coeff[ix + 1 + iy * coeff_pitch];
    float c10 = (float)coeff[ix + (iy + 1) * coeff_pitch];
    float c11 = (float)coeff[ix + 1 + (iy + 1) * coeff_pitch];
    float fracx = fx - (float)ix, fracy = fy - (float)iy;
    return (c00 * (1.0f - fracx) + c01 * fracx) * (1.0f - fracy)
         + (c10 * (1.0f - fracx) + c11 * fracx) * fracy;
}

// kf_sharpen twin (device form incl. the min(x+1,height-1) quirk;
// sim/kfm_deblock_aux_ref.cpp mode 'P' verbatim)
template <typename T>
static void ref_sharpen(int width, int height, int dst_pitch, T* dst,
                        const T* src, int src_pitch,
                        const uint8_t* coeff, int coeff_pitch,
                        const T* unsharp) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int s = (int)src[x + y * src_pitch];
            int l = s, h = s, vv;
            int xm1 = x - 1 >= 0 ? x - 1 : 0;
            int xp1 = x + 1 <= height - 1 ? x + 1 : height - 1;  // upstream quirk
            int ym1 = y - 1 >= 0 ? y - 1 : 0;
            int yp1 = y + 1 <= height - 1 ? y + 1 : height - 1;
            vv = (int)src[xm1 + ym1 * src_pitch];
            if (vv < l) l = vv; if (vv > h) h = vv;
            vv = (int)src[x + ym1 * src_pitch];
            if (vv < l) l = vv; if (vv > h) h = vv;
            vv = (int)src[xp1 + ym1 * src_pitch];
            if (vv < l) l = vv; if (vv > h) h = vv;
            vv = (int)src[xm1 + y * src_pitch];
            if (vv < l) l = vv; if (vv > h) h = vv;
            vv = (int)src[xp1 + y * src_pitch];
            if (vv < l) l = vv; if (vv > h) h = vv;
            vv = (int)src[xm1 + yp1 * src_pitch];
            if (vv < l) l = vv; if (vv > h) h = vv;
            vv = (int)src[x + yp1 * src_pitch];
            if (vv < l) l = vv; if (vv > h) h = vv;
            vv = (int)src[xp1 + yp1 * src_pitch];
            if (vv < l) l = vv; if (vv > h) h = vv;
            float c = ref_sharpen_bilinear(coeff, coeff_pitch, (float)x, (float)y) * (1.0f / 255.0f);
            int u = (int)unsharp[x + y * dst_pitch];
            float r = (float)s + (float)(s - u) * c + 0.5f;
            if (r < (float)l) r = (float)l;
            else if (r > (float)h) r = (float)h;
            dst[x + y * dst_pitch] = (T)(int)r;
        }
    }
}

// kf_show_sharpen_coeff twin (sim/kfm_deblock_aux_ref.cpp mode 'W')
template <typename T>
static void ref_show_sharpen_coeff(int width, int height, int dst_pitch, T* dst,
                                   const uint8_t* coeff, int coeff_pitch) {
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            dst[x + y * dst_pitch] = (T)(int)ref_sharpen_bilinear(coeff, coeff_pitch, (float)x, (float)y);
}

// ---- DrawText port (plugin verbatim; uses kfm_textout_font.h) ----
template <typename T>
static void ref_drawTextPlane(T* dst, int width, int height, int pitch,
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

// packed-plane DrawText twin of the plugin's drawTextMessage
template <typename T>
static void ref_drawTextMessage(std::vector<T>& Y, std::vector<T>& U, std::vector<T>& V,
                                int W, int H, int lx, int ly, int bits, const char* msg) {
    int maxlen = (int)strlen(msg);
    if (maxlen <= 0) return;
    std::vector<char> charmap((size_t)maxlen, 0);
    memcpy(charmap.data(), msg, (size_t)maxlen);
    int width = W, height = H;
    if (width > maxlen * 10) width = maxlen * 10;
    if (height > 20) height = 20;
    int logx = ((W >> lx) < width) ? 1 : 0;
    int logy = ((H >> ly) < height) ? 1 : 0;
    int shift = bits - 8;
    ref_drawTextPlane<T>(Y.data(), width, height, W, charmap.data(), maxlen, 1, maxlen, 0, 0, shift, false);
    ref_drawTextPlane<T>(U.data(), width >> logx, height >> logy, W >> lx,
                         charmap.data(), maxlen, 1, maxlen, logx, logy, shift, true);
    ref_drawTextPlane<T>(V.data(), width >> logx, height >> logy, W >> lx,
                         charmap.data(), maxlen, 1, maxlen, logx, logy, shift, true);
}

// ---------------------------------------------------------------------------
// reference pipeline: QP table + deblock (+ optional sharpen) per frame
// ---------------------------------------------------------------------------
struct CaseCfg {
    int quality;
    float str;
    float thr;
    float bratio;
    int qp;         // force_qp
    bool badap;
    bool sharp;
    int show;
    QPMode qpm;
    int scaleType;  // QP_ScaleType of the generated tables
    float conv;     // frameRateConv (fps ratio) for the SRCSTART modes
};

struct QPRef {
    int flag;   // QP_TABLE_NONE=0 / ONLY=1 / CONSTANT=2 / USING_DC=3
    bool threw;
    int tw, th, twU, thU;
    std::vector<uint16_t> tabY, tabU, tabV;
    QPRef() : flag(0), threw(false), tw(0), th(0), twU(0), thU(0) {}
};

// upstream QPEntry twin: read props off one/two QP source frames
static void refQpEntry(QPRef& out, const CaseCfg& cfg, int W, int H, int lx, int ly,
                       int mbw, int mbh, PVideoFrame qp0, PVideoFrame qp1,
                       IScriptEnvironment* env) {
    const AVSMap* m0 = env->getFramePropsRO(qp0);
    int error;
    env->propGetInt(m0, "QP_Table_Non_B", 0, &error);
    if (error) { out.flag = 0 /*NONE*/; return; }
    int errT;
    PVideoFrame tab0f = env->propGetFrame(m0, "QP_Table", 0, &errT);
    PVideoFrame nonb0f = env->propGetFrame(m0, "QP_Table_Non_B", 0, &errT);
    int qpStride = (int)env->propGetInt(m0, "QP_Stride", 0, &error);
    int scaleType = (int)env->propGetInt(m0, "QP_ScaleType", 0, &error);
    PVideoFrame dc0f;
    if (cfg.badap) dc0f = env->propGetFrame(m0, "DC_Table", 0, &errT);
    PVideoFrame tab1f, nonb1f, dc1f;
    if (qp1) {
        const AVSMap* m1 = env->getFramePropsRO(qp1);
        tab1f = env->propGetFrame(m1, "QP_Table", 0, &errT);
        nonb1f = env->propGetFrame(m1, "QP_Table_Non_B", 0, &errT);
        if (cfg.badap) dc1f = env->propGetFrame(m1, "DC_Table", 0, &errT);
    }

    // MakeMask per DC table (host staging twin)
    std::vector<uint8_t> mask0, mask1;
    int mask0w = 0, mask1w = 0;
    if (dc0f) {
        mask0w = dc0f->GetRowSize(PLANAR_Y);
        makeMaskCpu(mask0, dc0f->GetReadPtr(PLANAR_Y), dc0f->GetPitch(PLANAR_Y),
                    dc0f->GetRowSize(PLANAR_Y), dc0f->GetHeight(PLANAR_Y));
    }
    if (dc1f) {
        mask1w = dc1f->GetRowSize(PLANAR_Y);
        makeMaskCpu(mask1, dc1f->GetReadPtr(PLANAR_Y), dc1f->GetPitch(PLANAR_Y),
                    dc1f->GetRowSize(PLANAR_Y), dc1f->GetHeight(PLANAR_Y));
    }

    float dc_coeff = cfg.bratio / 255.0f;
    const uint8_t* t0 = tab0f ? tab0f->GetReadPtr(PLANAR_Y) : nullptr;
    const uint8_t* n0 = nonb0f ? nonb0f->GetReadPtr(PLANAR_Y) : nullptr;
    const uint8_t* t1 = tab1f ? tab1f->GetReadPtr(PLANAR_Y) : nullptr;
    const uint8_t* n1 = nonb1f ? nonb1f->GetReadPtr(PLANAR_Y) : nullptr;
    const uint8_t* msk0 = mask0.empty() ? nullptr : mask0.data();
    const uint8_t* msk1 = mask1.empty() ? nullptr : mask1.data();

    // Y plane
    ref_make_qp_table(mbw, mbh, t0, n0, tab1f != nullptr, t1, n1, qpStride,
                      scaleType, dc_coeff, msk0 != nullptr, msk0,
                      msk1 != nullptr, msk1, mask0w,
                      1, 1, (W + 15) >> 3, (H + 15) >> 3, out.tabY.data(), out.tw);
    // chroma planes (U written, V copied 窶・same content)
    ref_make_qp_table(mbw, mbh, t0, n0, tab1f != nullptr, t1, n1, qpStride,
                      scaleType, dc_coeff, msk0 != nullptr, msk0,
                      msk1 != nullptr, msk1, mask0w,
                      1 - lx, 1 - ly, ((W >> lx) + 15) >> 3, ((H >> ly) + 15) >> 3,
                      out.tabU.data(), out.twU);
    out.tabV = out.tabU;
    out.flag = dc0f ? 3 /*USING_DC*/ : 1 /*ONLY*/;
}

static QPRef refComputeQP(const CaseCfg& cfg, PClip srcClip, PClip qpclip,
                          int W, int H, int lx, int ly, int mbw, int mbh, int n,
                          IScriptEnvironment* env) {
    QPRef out;
    out.tw = calc_qp_size(W); out.th = calc_qp_size(H);
    out.twU = out.tw >> lx; out.thU = out.th >> ly;
    out.tabY.assign((size_t)out.tw * out.th, 0);
    out.tabU.assign((size_t)out.twU * out.thU, 0);
    out.tabV.assign((size_t)out.twU * out.thU, 0);

    if (cfg.qp > 0) {
        uint16_t v = (uint16_t)cfg.qp;
        std::fill(out.tabY.begin(), out.tabY.end(), v);
        std::fill(out.tabU.begin(), out.tabU.end(), v);
        std::fill(out.tabV.begin(), out.tabV.end(), v);
        out.flag = 2 /*CONSTANT*/;
        return out;
    }

    PVideoFrame srcf = srcClip->GetFrame(n, env);
    const AVSMap* m = env->getFramePropsRO(srcf);
    int error;
    env->propGetInt(m, "KFM_SourceStart", 0, &error);
    if (!error) {
        int start = (int)env->propGetInt(m, "KFM_SourceStart", 0, &error);
        int num = (int)env->propGetInt(m, "KFM_NumSourceFrames", 0, &error);
        int qp_n = (int)(cfg.conv * n + 0.3f);
        if (start < qp_n - 10 || start > qp_n + 10) { out.threw = true; return out; }
        PVideoFrame qp0 = qpclip ? qpclip->GetFrame(start, env) : srcClip->GetFrame(start, env);
        PVideoFrame qp1;
        if (num > 1) qp1 = qpclip ? qpclip->GetFrame(start + 1, env) : srcClip->GetFrame(start + 1, env);
        refQpEntry(out, cfg, W, H, lx, ly, mbw, mbh, qp0, qp1, env);
        return out;
    }
    if (qpclip) {
        int qp_n = (int)(cfg.conv * n + 0.3f);
        refQpEntry(out, cfg, W, H, lx, ly, mbw, mbh, qpclip->GetFrame(qp_n, env), PVideoFrame(), env);
        return out;
    }
    refQpEntry(out, cfg, W, H, lx, ly, mbw, mbh, srcf, PVideoFrame(), env);
    return out;
}

// one packed plane of the reference deblocked frame
template <typename T>
static void refDeblockPlaneT(std::vector<T>& out, const std::vector<T>& src,
                             int width, int height, const uint16_t* qpTab, int qpPitch,
                             const CaseCfg& cfg, int bits,
                             float thresh_a, float thresh_b) {
    if (qpTab == nullptr) { out = src; return; }  // QP_TABLE_NONE: passthrough

    if (cfg.show == 2) {
        out.resize((size_t)width * height);
        ref_deblock_show<T>(width, height, width, out.data(), qpTab, qpPitch,
                            thresh_a, thresh_b);
        return;
    }

    int dshift = cfg.quality + bits - 10;
    if (dshift < 0) dshift = 0;
    const int deblockShift = dshift;
    const int deblockMaxV = (1 << (bits + 6 - deblockShift)) - 1;
    const int mergeShift = cfg.quality + 6 - deblockShift;
    const int mergeMaxV = (1 << bits) - 1;
    const int count = 1 << cfg.quality;

    // padded plane + accumulator (origin = upstream interior, plugin DEVIATION 1)
    // pad buffer height is the 8-rounded ph_r; rows beyond h+16 are zero
    // (DEVIATION 6 — upstream leaves them uninitialized) and must match the
    // device, which zero-fills them.
    std::vector<T> pad;
    stageMirrorPad(pad, src.data(), width, width, height, 8);
    const int pw = (width + 7 + 8 * 2) & ~7;
    const int bh = (height + 7 + 8) >> 3;
    const int bw = (width + 7 + 8) >> 3;
    pad.resize((size_t)pw * (((height + 7 + 8 * 2) & ~7)), 0);
    std::vector<uint16_t> tmp((size_t)pw * (((height + 7 + 8 * 2) & ~7) * 4), 0);

    int local_out[16][16];
    for (int by = 0; by < bh; by++) for (int bx = 0; bx < bw; bx++) {
        for (int r = 0; r < 16; r++) for (int c = 0; c < 16; c++) local_out[r][c] = 0;
        for (int ty = 0; ty <= count - 1; ty++) {
            int ox0 = g_offx[count - 1 + ty], oy0 = g_offy[count - 1 + ty];
            int ox = bx * 8 + ox0, oy = by * 8 + oy0;
            float d[64];
            for (int y = 0; y < 8; y++) for (int xx = 0; xx < 8; xx++)
                d[xx + y * 8] = (float)pad[(ox + xx) + (oy + y) * pw];
            int qpv = qpTab[bx + by * qpPitch];
            float thresh = qp_thresh(qpv, thresh_a, thresh_b) * ((1 << 2) + cfg.str) - 1.0f;
            dct8x8(d); hardthresh(d, thresh); idct8x8(d);
            int half = (1 << deblockShift) >> 1;
            for (int y = 0; y < 8; y++) for (int xx = 0; xx < 8; xx++) {
                int t = ((int)(d[xx + y * 8] + (float)half)) >> deblockShift;
                if (t < 0) t = 0; else if (t > deblockMaxV) t = deblockMaxV;
                local_out[oy0 + y][ox0 + xx] += t;
            }
        }
        int offz = (bx & 1) + (by & 1) * 2;
        int offx = bx * 8, offy = (bh * offz + by) * 8;
        for (int y = 0; y < 16; y++) for (int xx = 0; xx < 16; xx++)
            tmp[(size_t)(offx + xx + 8) + (size_t)(offy + y + 8) * pw] = (uint16_t)local_out[y][xx];
    }

    // merge (kf_merge_deblock twin, shifted-origin scheme; aux_ref mode 'G')
    out.resize((size_t)width * height);
    float maxv = (float)mergeMaxV;
    float inv = 1.0f / (float)(1 << mergeShift);
    const float sixth = 1.0f / 64.0f;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int X = x >> 2, L = x & 3;
            int sum = 0;
            for (int k = 0; k < 4; k++) {
                int row = bh * 8 * k + y;
                sum += (int)tmp[(size_t)(8 + x) + (size_t)(8 + row) * pw];
            }
            int dth = LDITHER[y & 7][X & 1][L];
            float vv = (float)sum * inv + (float)dth * sixth;
            vv = vv < maxv ? vv : maxv;
            out[(size_t)y * width + x] = (T)(int)vv;
        }
    }
}

// full reference deblocked frame (3 packed planes, text applied)
struct RefFrame {
    std::vector<uint8_t> Y, U, V;          // 8-bit view
    std::vector<uint16_t> Y16, U16, V16;   // 16-bit view
    bool is16;
};

// build the reference deblocked frame for frame n (QP_TABLE_NONE passthrough
// + show text included 窶・matches the plugin's deblocked output exactly)
static bool refDeblockFrame(const CaseCfg& cfg, PClip srcClip, PClip qpclip,
                            const VideoInfo& vi, int W, int H, int lx, int ly,
                            int mbw, int mbh, int n, RefFrame& rf,
                            IScriptEnvironment* env) {
    const int bits = vi.BitsPerComponent();
    const int px = vi.ComponentSize();
    const float W5 = 5;
    const float thresh_a = (cfg.thr + W5) / (2 * W5);
    const float thresh_b = (W5 * W5 - cfg.thr * cfg.thr) / (2 * W5);

    QPRef qp = refComputeQP(cfg, srcClip, qpclip, W, H, lx, ly, mbw, mbh, n, env);
    if (qp.threw) return false;

    rf.is16 = px == 2;
    PVideoFrame srcf = srcClip->GetFrame(n, env);

    const int wUV = W >> lx, hUV = H >> ly;
    const uint16_t* qpY = qp.flag == 0 ? nullptr : qp.tabY.data();
    const uint16_t* qpU = qp.flag == 0 ? nullptr : qp.tabU.data();

    if (px == 1) {
        std::vector<uint8_t> sY((size_t)W * H), sU((size_t)wUV * hUV), sV((size_t)wUV * hUV);
        for (int y = 0; y < H; ++y)
            memcpy(&sY[(size_t)y * W], srcf->GetReadPtr(PLANAR_Y) + (size_t)y * srcf->GetPitch(PLANAR_Y), W);
        for (int y = 0; y < hUV; ++y) {
            memcpy(&sU[(size_t)y * wUV], srcf->GetReadPtr(PLANAR_U) + (size_t)y * srcf->GetPitch(PLANAR_U), wUV);
            memcpy(&sV[(size_t)y * wUV], srcf->GetReadPtr(PLANAR_V) + (size_t)y * srcf->GetPitch(PLANAR_V), wUV);
        }
        refDeblockPlaneT<uint8_t>(rf.Y, sY, W, H, qpY, qp.tw, cfg, bits, thresh_a, thresh_b);
        refDeblockPlaneT<uint8_t>(rf.U, sU, wUV, hUV, qpU, qp.twU, cfg, bits, thresh_a, thresh_b);
        refDeblockPlaneT<uint8_t>(rf.V, sV, wUV, hUV, qpU, qp.twU, cfg, bits, thresh_a, thresh_b);
        if (cfg.show == 1 || (qp.flag == 0 && cfg.show)) {
            const char* msg;
            switch (qp.flag) {
            case 1: msg = "KDeblock: Using QP table"; break;
            case 2: msg = "KDeblock: Constant QP"; break;
            case 3: msg = "KDeblock: Using QP table with DC"; break;
            default: msg = "KDeblock: Disabled"; break;
            }
            ref_drawTextMessage<uint8_t>(rf.Y, rf.U, rf.V, W, H, lx, ly, bits, msg);
        }
    } else {
        std::vector<uint16_t> sY((size_t)W * H), sU((size_t)wUV * hUV), sV((size_t)wUV * hUV);
        for (int y = 0; y < H; ++y)
            memcpy(&sY[(size_t)y * W], srcf->GetReadPtr(PLANAR_Y) + (size_t)y * srcf->GetPitch(PLANAR_Y), (size_t)W * 2);
        for (int y = 0; y < hUV; ++y) {
            memcpy(&sU[(size_t)y * wUV], srcf->GetReadPtr(PLANAR_U) + (size_t)y * srcf->GetPitch(PLANAR_U), (size_t)wUV * 2);
            memcpy(&sV[(size_t)y * wUV], srcf->GetReadPtr(PLANAR_V) + (size_t)y * srcf->GetPitch(PLANAR_V), (size_t)wUV * 2);
        }
        refDeblockPlaneT<uint16_t>(rf.Y16, sY, W, H, qpY, qp.tw, cfg, bits, thresh_a, thresh_b);
        refDeblockPlaneT<uint16_t>(rf.U16, sU, wUV, hUV, qpU, qp.twU, cfg, bits, thresh_a, thresh_b);
        refDeblockPlaneT<uint16_t>(rf.V16, sV, wUV, hUV, qpU, qp.twU, cfg, bits, thresh_a, thresh_b);
        if (cfg.show == 1 || (qp.flag == 0 && cfg.show)) {
            const char* msg;
            switch (qp.flag) {
            case 1: msg = "KDeblock: Using QP table"; break;
            case 2: msg = "KDeblock: Constant QP"; break;
            case 3: msg = "KDeblock: Using QP table with DC"; break;
            default: msg = "KDeblock: Disabled"; break;
            }
            ref_drawTextMessage<uint16_t>(rf.Y16, rf.U16, rf.V16, W, H, lx, ly, bits, msg);
        }
    }
    return true;
}

// IClip serving the reference deblocked frames (feeds the REAL core
// GaussResize, exactly like the plugin's chain)
class RefClip : public IClip {
public:
    CaseCfg cfg;
    PClip srcClip, qpclip;
    VideoInfo vi_;
    int W, H, lx, ly, mbw, mbh;

    RefClip(const CaseCfg& c, PClip src, PClip qp, const VideoInfo& vi,
            int W_, int H_, int lx_, int ly_, int mbw_, int mbh_)
        : cfg(c), srcClip(src), qpclip(qp), vi_(vi), W(W_), H(H_), lx(lx_),
          ly(ly_), mbw(mbw_), mbh(mbh_) {}

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) {
        RefFrame rf;
        if (!refDeblockFrame(cfg, srcClip, qpclip, vi_, W, H, lx, ly, mbw, mbh, n, rf, env))
            env->ThrowError("refDeblockFrame threw (KFM_SourceStart)");
        PVideoFrame f = env->NewVideoFrame(vi_);
        const int wUV = W >> lx, hUV = H >> ly;
        if (!rf.is16) {
            for (int y = 0; y < H; ++y)
                memcpy(f->GetWritePtr(PLANAR_Y) + (size_t)y * f->GetPitch(PLANAR_Y),
                       &rf.Y[(size_t)y * W], W);
            for (int y = 0; y < hUV; ++y) {
                memcpy(f->GetWritePtr(PLANAR_U) + (size_t)y * f->GetPitch(PLANAR_U),
                       &rf.U[(size_t)y * wUV], wUV);
                memcpy(f->GetWritePtr(PLANAR_V) + (size_t)y * f->GetPitch(PLANAR_V),
                       &rf.V[(size_t)y * wUV], wUV);
            }
        } else {
            for (int y = 0; y < H; ++y)
                memcpy(f->GetWritePtr(PLANAR_Y) + (size_t)y * f->GetPitch(PLANAR_Y),
                       &rf.Y16[(size_t)y * W], (size_t)W * 2);
            for (int y = 0; y < hUV; ++y) {
                memcpy(f->GetWritePtr(PLANAR_U) + (size_t)y * f->GetPitch(PLANAR_U),
                       &rf.U16[(size_t)y * wUV], (size_t)wUV * 2);
                memcpy(f->GetWritePtr(PLANAR_V) + (size_t)y * f->GetPitch(PLANAR_V),
                       &rf.V16[(size_t)y * wUV], (size_t)wUV * 2);
            }
        }
        return f;
    }
    bool __stdcall GetParity(int) { return false; }
    void __stdcall GetAudio(void*, int64_t, int64_t, IScriptEnvironment*) {}
    int __stdcall SetCacheHints(int, int) { return 0; }
    const VideoInfo& __stdcall GetVideoInfo() { return vi_; }
};

// reference sharpen tail + compare helpers ----------------------------------

template <typename T>
static void readPlanePacked(std::vector<T>& out, PVideoFrame f, int pl, int pw, int ph) {
    out.resize((size_t)pw * ph);
    const uint8_t* fp = f->GetReadPtr(pl);
    int pitch = f->GetPitch(pl);
    for (int y = 0; y < ph; ++y)
        memcpy(&out[(size_t)y * pw], fp + (size_t)y * pitch, (size_t)pw * sizeof(T));
}

// kf_sharpen_coeff twin over the FULL qp-table dims (zero margins included)
static void refSharpenCoeff(std::vector<uint8_t>& cf, const std::vector<uint16_t>& tab,
                            int tw, int th) {
    cf.resize((size_t)tw * th);
    for (int y = 0; y < th; ++y)
        for (int x = 0; x < tw; ++x) {
            int q = (int)tab[(size_t)x + y * tw] >> 3;
            cf[(size_t)x + y * tw] = (uint8_t)(q >= 25 ? 255 : SHARPEN_COEFF[q]);
        }
}

// full reference sharpen output (3 packed planes) for frame n
template <typename T>
static void refSharpenAll(std::vector<T>& dY, std::vector<T>& dU, std::vector<T>& dV,
                              const RefFrame& rf, PVideoFrame unsharpf, const QPRef& qp,
                              int W, int H, int lx, int ly, const CaseCfg& cfg) {
    const int wUV = W >> lx, hUV = H >> ly;
    const int twU = qp.tw >> lx, thU = qp.th >> ly;
    std::vector<uint8_t> cfY, cfU;
    refSharpenCoeff(cfY, qp.tabY, qp.tw, qp.th);
    refSharpenCoeff(cfU, qp.tabU, twU, thU);

    // unsharp planes (REAL GaussResize output)
    std::vector<T> uY, uU, uV;
    if constexpr (sizeof(T) == 1) {
        readPlanePacked<uint8_t>(uY, unsharpf, PLANAR_Y, W, H);
        readPlanePacked<uint8_t>(uU, unsharpf, PLANAR_U, wUV, hUV);
        readPlanePacked<uint8_t>(uV, unsharpf, PLANAR_V, wUV, hUV);
    } else {
        readPlanePacked<uint16_t>(uY, unsharpf, PLANAR_Y, W, H);
        readPlanePacked<uint16_t>(uU, unsharpf, PLANAR_U, wUV, hUV);
        readPlanePacked<uint16_t>(uV, unsharpf, PLANAR_V, wUV, hUV);
    }

    const T* rY = sizeof(T) == 1 ? (const T*)rf.Y.data() : (const T*)rf.Y16.data();
    const T* rU = sizeof(T) == 1 ? (const T*)rf.U.data() : (const T*)rf.U16.data();
    const T* rV = sizeof(T) == 1 ? (const T*)rf.V.data() : (const T*)rf.V16.data();

    dY.assign((size_t)W * H, 0);
    dU.assign((size_t)wUV * hUV, 0);
    dV.assign((size_t)wUV * hUV, 0);
    if (cfg.show == 3) {
        ref_show_sharpen_coeff<T>(W, H, W, dY.data(), cfY.data(), qp.tw);
        ref_show_sharpen_coeff<T>(wUV, hUV, wUV, dU.data(), cfU.data(), twU);
        ref_show_sharpen_coeff<T>(wUV, hUV, wUV, dV.data(), cfU.data(), twU);
    } else {
        ref_sharpen<T>(W, H, W, dY.data(), rY, W, cfY.data(), qp.tw, uY.data());
        ref_sharpen<T>(wUV, hUV, wUV, dU.data(), rU, wUV, cfU.data(), twU, uU.data());
        ref_sharpen<T>(wUV, hUV, wUV, dV.data(), rV, wUV, cfU.data(), twU, uV.data());
    }
}

template <typename T>
static bool comparePlane(const char* tag, int n, const char* plane, PVideoFrame f, int pl,
                         const std::vector<T>& exp, int pw, int ph) {
    const uint8_t* fp = f->GetReadPtr(pl);
    int pitch = f->GetPitch(pl);
    int bad = 0;
    for (int y = 0; y < ph; ++y) {
        for (int x = 0; x < pw; ++x) {
            int got = sizeof(T) == 1 ? fp[(size_t)y * pitch + x]
                                     : (int)((uint16_t*)(fp + (size_t)y * pitch))[x];
            if (got != (int)exp[(size_t)y * pw + x]) {
                if (bad < 10)
                    printf("  MISMATCH %s frame %d %s: px(%d,%d) got=%d exp=%d\n",
                           tag, n, plane, x, y, got, (int)exp[(size_t)y * pw + x]);
                ++bad;
            }
        }
    }
    if (bad) {
        printf("  total mismatches: %d / %d in %s\n", bad, pw * ph, plane);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// case runner
// ---------------------------------------------------------------------------
static VideoInfo makeSrcVi(int W, int H, int pixelType, int numFrames,
                           int fpsNum, int fpsDen) {
    VideoInfo vi;
    memset(&vi, 0, sizeof vi);
    vi.width = W; vi.height = H;
    vi.pixel_type = pixelType;
    vi.num_frames = numFrames;
    vi.fps_numerator = fpsNum; vi.fps_denominator = fpsDen;
    return vi;
}

static void runCase(IScriptEnvironment* env, const char* tag, const CaseCfg& cfg,
                    PClip srcClip, PClip qpclip, const VideoInfo& vi,
                    int W, int H, int lx, int ly, int mbw, int mbh,
                    bool useAlias, bool expectThrow) {
    AVSValue args[10] = {
        AVSValue(srcClip),
        cfg.quality,
        (double)cfg.str,
        (double)cfg.thr,
        (double)cfg.bratio,
        qpclip ? AVSValue(qpclip) : AVSValue(),
        cfg.qp,
        cfg.badap,
        cfg.sharp,
        cfg.show,
    };
    PClip filtered;
    try {
        AVSValue ret = env->Invoke(useAlias ? "KDeblock" : "KDeblockOCL", AVSValue(args, 10));
        if (!ret.IsClip()) { printf("[FAIL] %s: Invoke did not return a clip\n", tag); ++g_failures; return; }
        filtered = ret.AsClip();
    } catch (const AvisynthError& e) {
        printf("[FAIL] %s: Invoke threw: %s\n", tag, e.msg);
        ++g_failures;
        return;
    }

    // reference deblocked clip (feeds the REAL core GaussResize for sharp)
    CaseCfg rcfg = cfg;
    PClip refDeblock = new RefClip(rcfg, srcClip, qpclip, vi, W, H, lx, ly, mbw, mbh);

    bool ok = true;
    for (int n = 0; n < 2 && ok; ++n) {
        // reference
        RefFrame rf;
        if (!refDeblockFrame(cfg, srcClip, qpclip, vi, W, H, lx, ly, mbw, mbh, n, rf, env)) {
            // reference says the frame mapping must throw
            try {
                filtered->GetFrame(n, env);
                printf("  GetFrame did NOT throw (expected Invalid KFM_SourceStart)\n");
                ok = false;
            } catch (const AvisynthError&) {
                ok = true;  // expected
            }
            if (ok) { ++g_frame_checks; continue; }
            break;
        }

        std::vector<uint8_t> eY, eU, eV;
        std::vector<uint16_t> eY16, eU16, eV16;
        if (cfg.sharp) {
            // unsharp = REAL core GaussResize on the reference deblocked clip,
            // invoked exactly like the plugin chain
            AVSValue gargs[8] = { AVSValue(refDeblock), W, H, 0, 0,
                                  (double)W + 0.0001, (double)H + 0.0001, 2 };
            AVSValue unsharpV;
            try {
                unsharpV = env->Invoke("GaussResize", AVSValue(gargs, 8));
                if (!unsharpV.IsClip()) throw AvisynthError("no clip");
            } catch (const AvisynthError& e) {
                printf("  GaussResize(ref) threw: %s\n", e.msg);
                ok = false;
                break;
            }
            PVideoFrame uf = unsharpV.AsClip()->GetFrame(n, env);
            QPRef qp = refComputeQP(cfg, srcClip, qpclip, W, H, lx, ly, mbw, mbh, n, env);
            if (rf.is16)
                refSharpenAll<uint16_t>(eY16, eU16, eV16, rf, uf, qp, W, H, lx, ly, cfg);
            else
                refSharpenAll<uint8_t>(eY, eU, eV, rf, uf, qp, W, H, lx, ly, cfg);
        } else {
            eY = rf.Y; eU = rf.U; eV = rf.V;
            eY16 = rf.Y16; eU16 = rf.U16; eV16 = rf.V16;
        }

        // plugin output
        PVideoFrame outf;
        try {
            outf = filtered->GetFrame(n, env);
        } catch (const AvisynthError& e) {
            printf("  GetFrame threw unexpectedly: %s\n", e.msg);
            ok = false;
            break;
        }
        ++g_frame_checks;
        const int wUV = W >> lx, hUV = H >> ly;
        if (rf.is16) {
            ok = comparePlane<uint16_t>(tag, n, "Y", outf, PLANAR_Y, eY16, W, H)
              && comparePlane<uint16_t>(tag, n, "U", outf, PLANAR_U, eU16, wUV, hUV)
              && comparePlane<uint16_t>(tag, n, "V", outf, PLANAR_V, eV16, wUV, hUV);
        } else {
            ok = comparePlane<uint8_t>(tag, n, "Y", outf, PLANAR_Y, eY, W, H)
              && comparePlane<uint8_t>(tag, n, "U", outf, PLANAR_U, eU, wUV, hUV)
              && comparePlane<uint8_t>(tag, n, "V", outf, PLANAR_V, eV, wUV, hUV);
        }
    }
    ++g_cases;
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", tag);
    if (!ok) ++g_failures;
}

// build source/qpclip pair for a QP mode and run the case
static void runModeCase(IScriptEnvironment* env, const char* tag, const CaseCfg& cfg,
                        int W, int H, int pixelType, SourceKind kind, bool useAlias,
                        bool expectThrow = false) {
    const int numFrames = 4;
    int fpsNum = 50, fpsDen = 1;
    int qpFpsNum = fpsNum, qpFpsDen = fpsDen;
    if (cfg.conv != 1.0f) { qpFpsNum = 25; qpFpsDen = 1; }
    VideoInfo vi = makeSrcVi(W, H, pixelType, numFrames, fpsNum, fpsDen);
    int lx = vi.GetPlaneWidthSubsampling(PLANAR_U);
    int ly = vi.GetPlaneHeightSubsampling(PLANAR_U);
    int mbw = (W + 15) >> 4, mbh = (H + 15) >> 4;

    bool needQpclip = cfg.qpm == QP_CLIP || cfg.qpm == QP_CLIP_DC || cfg.qpm == QP_BADCLIP
                   || cfg.qpm == QP_SRCSTART || cfg.qpm == QP_SRCSTART2 || cfg.qpm == QP_SRCSTART_BAD;
    bool withDC = cfg.qpm == QP_ONFRAME_DC || cfg.qpm == QP_CLIP_DC;

    PClip qpclip;
    if (needQpclip) {
        VideoInfo qvi = makeSrcVi(W, H, VideoInfo::CS_Y8, numFrames + 4, qpFpsNum, qpFpsDen);
        qpclip = new QPTableSource(qvi, mbw, mbh, withDC && cfg.qpm != QP_BADCLIP, cfg.scaleType);
    }

    QPMode srcMode = cfg.qpm;
    if (srcMode == QP_CLIP || srcMode == QP_CLIP_DC || srcMode == QP_BADCLIP)
        srcMode = QP_NONE;  // no props on the source frames themselves

    PClip srcClip = new TestSource(vi, 0xC0FFEEu ^ ((uint32_t)kind << 20), kind,
                                   srcMode, withDC, cfg.scaleType, cfg.conv, mbw, mbh);
    // the plugin computes frameRateConv from the fps ratio; mirror it into cfg
    CaseCfg c = cfg;
    if (needQpclip)
        c.conv = (float)((int64_t)qpFpsNum * fpsDen) / (float)((int64_t)qpFpsDen * fpsNum);
    else
        c.conv = 1.0f;

    runCase(env, tag, c, srcClip, needQpclip ? qpclip : PClip(), vi,
            W, H, lx, ly, mbw, mbh, useAlias, expectThrow);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    AddVectoredExceptionHandler(1, eh);
    const char* pluginDll = argc > 1 ? argv[1] : nullptr;
    std::string pluginsDir = argc > 2 ? argv[2]
        : "Q:\\Downloads\\openclfilter\\sdk\\avs\\plugins64";

    // ---- install the plugin into the autoload directory ----
    if (pluginDll) {
        std::string dst = pluginsDir + "\\kfm_deblock_avs.dll";
        if (!CopyFileA(pluginDll, dst.c_str(), FALSE)) {
            printf("cannot copy plugin '%s' -> '%s' (%lu)\n", pluginDll, dst.c_str(), GetLastError());
            return 1;
        }
        printf("plugin installed: %s\n", dst.c_str());
    }

    // ---- in-process AviSynth host (verified smoke pattern) ----
    HMODULE dll = LoadLibraryA("AviSynth.dll");
    if (!dll) { printf("cannot load AviSynth.dll (%lu)\n", GetLastError()); return 1; }
    FARPROC pv = GetProcAddress(dll, "?AVS_linkage@@3QEBUAVS_Linkage@@EB");
    if (!pv) { printf("AVS_linkage export not found\n"); return 1; }
    AVS_linkage = *(const AVS_Linkage**)pv;
    CSE_t cse = (CSE_t)GetProcAddress(dll, "CreateScriptEnvironment");
    if (!cse) { printf("CreateScriptEnvironment not exported\n"); return 1; }
    IScriptEnvironment* env = cse(AVISYNTH_INTERFACE_VERSION);
    if (!env) { printf("env creation failed\n"); return 1; }
    g_env = env;

    // ---- autoload validation ----
    if (!env->FunctionExists("KDeblockOCL")) {
        printf("[FAIL] autoload: KDeblockOCL not registered after env creation\n");
        return 1;
    }
    if (!env->FunctionExists("KDeblock")) {
        printf("[FAIL] autoload: KDeblock alias not registered\n");
        return 1;
    }
    printf("autoload OK (KDeblockOCL + KDeblock alias registered)\n");

    // ---- case matrix ----
    struct Fmt { const char* name; int pt; };
    static const Fmt g_formats[] = {
        { "YV12",      VideoInfo::CS_YV12 },
        { "YV16",      VideoInfo::CS_YV16 },
        { "YV24",      VideoInfo::CS_YV24 },
        { "YUV420P16", VideoInfo::CS_YUV420P16 },
        { "YUV422P16", VideoInfo::CS_YUV422P16 },
        { "YUV444P16", VideoInfo::CS_YUV444P16 },
    };
    const int W = 64, H = 48;

    for (const Fmt& fm : g_formats) {
        char tag[160];

        // defaults (no QP source at all) -> QP_TABLE_NONE passthrough
        sprintf_s(tag, "%s defaults (QP_TABLE_NONE passthrough)", fm.name);
        runModeCase(env, tag, CaseCfg{ 3, 0, 0, 0.5f, -1, true, false, 0, QP_NONE, 0, 1.0f },
                    W, H, fm.pt, SRC_MIXED, false);

        // props on source frames -> QP_TABLE_ONLY
        sprintf_s(tag, "%s props-on-frame q3", fm.name);
        runModeCase(env, tag, CaseCfg{ 3, 0, 0, 0.5f, -1, false, false, 0, QP_ONFRAME, 0, 1.0f },
                    W, H, fm.pt, SRC_BLOCKY, false);

        // qpclip -> QP_TABLE_ONLY, quality/str/thr sweep
        sprintf_s(tag, "%s qpclip q1 str2 thr5 scaleType0", fm.name);
        runModeCase(env, tag, CaseCfg{ 1, 2, 5, 0.5f, -1, false, false, 0, QP_CLIP, 0, 1.0f },
                    W, H, fm.pt, SRC_NOISY, false);

        sprintf_s(tag, "%s qpclip q0 scaleType2", fm.name);
        runModeCase(env, tag, CaseCfg{ 0, 0, 0, 0.5f, -1, false, false, 0, QP_CLIP, 2, 1.0f },
                    W, H, fm.pt, SRC_SMOOTH, false);

        // qpclip with different fps -> frameRateConv 0.5 mapping
        sprintf_s(tag, "%s qpclip conv=0.5 q2", fm.name);
        runModeCase(env, tag, CaseCfg{ 2, 1, 0, 0.5f, -1, false, false, 0, QP_CLIP, 0, 0.5f },
                    W, H, fm.pt, SRC_NOISY, false);

        // qpclip + DC + badap -> QP_TABLE_USING_DC
        sprintf_s(tag, "%s qpclip+DC badap bratio2 q2", fm.name);
        runModeCase(env, tag, CaseCfg{ 2, 0, 0, 2.0f, -1, true, false, 0, QP_CLIP_DC, 0, 1.0f },
                    W, H, fm.pt, SRC_BLOCKY, false);

        // badap=false with DC props present (DC ignored)
        sprintf_s(tag, "%s DC-props badap=false", fm.name);
        runModeCase(env, tag, CaseCfg{ 2, 0, 0, 0.5f, -1, false, false, 0, QP_CLIP_DC, 0, 1.0f },
                    W, H, fm.pt, SRC_MIXED, false);

        // forced qp -> QP_TABLE_CONSTANT
        sprintf_s(tag, "%s force_qp=6 q0", fm.name);
        runModeCase(env, tag, CaseCfg{ 0, 0, 0, 0.5f, 6, true, false, 0, QP_FORCE, 0, 1.0f },
                    W, H, fm.pt, SRC_NOISY, false);

        sprintf_s(tag, "%s force_qp=30 q6 str1 thr-10", fm.name);
        runModeCase(env, tag, CaseCfg{ 6, 1, -10, 0.5f, 30, false, false, 0, QP_FORCE, 0, 1.0f },
                    W, H, fm.pt, SRC_MIXED, false);

        // show = 1 variants (all four flag messages)
        sprintf_s(tag, "%s show=1 qpclip (Using QP table)", fm.name);
        runModeCase(env, tag, CaseCfg{ 2, 0, 0, 0.5f, -1, false, false, 1, QP_CLIP, 0, 1.0f },
                    W, H, fm.pt, SRC_BLOCKY, false);
        sprintf_s(tag, "%s show=1 force_qp (Constant QP)", fm.name);
        runModeCase(env, tag, CaseCfg{ 2, 0, 0, 0.5f, 12, false, false, 1, QP_FORCE, 0, 1.0f },
                    W, H, fm.pt, SRC_SMOOTH, false);
        sprintf_s(tag, "%s show=1 qpclip+DC (Using QP table with DC)", fm.name);
        runModeCase(env, tag, CaseCfg{ 2, 0, 0, 2.0f, -1, true, false, 1, QP_CLIP_DC, 0, 1.0f },
                    W, H, fm.pt, SRC_NOISY, false);
        sprintf_s(tag, "%s show=1 badclip (Disabled)", fm.name);
        runModeCase(env, tag, CaseCfg{ 2, 0, 0, 0.5f, -1, false, false, 1, QP_BADCLIP, 0, 1.0f },
                    W, H, fm.pt, SRC_MIXED, false);

        // show = 2 (per-block deblock flags)
        sprintf_s(tag, "%s show=2 qpclip", fm.name);
        runModeCase(env, tag, CaseCfg{ 2, 0, 0, 0.5f, -1, false, false, 2, QP_CLIP, 0, 1.0f },
                    W, H, fm.pt, SRC_BLOCKY, false);
        sprintf_s(tag, "%s show=2 force_qp thr=10", fm.name);
        runModeCase(env, tag, CaseCfg{ 2, 0, 10, 0.5f, 20, false, false, 2, QP_FORCE, 0, 1.0f },
                    W, H, fm.pt, SRC_NOISY, false);

        // sharp = true
        sprintf_s(tag, "%s sharp qpclip q2 str3", fm.name);
        runModeCase(env, tag, CaseCfg{ 2, 3, 0, 0.5f, -1, false, true, 0, QP_CLIP, 0, 1.0f },
                    W, H, fm.pt, SRC_BLOCKY, false);
        sprintf_s(tag, "%s sharp force_qp=8 q1 show=3", fm.name);
        runModeCase(env, tag, CaseCfg{ 1, 0, 0, 0.5f, 8, false, true, 3, QP_FORCE, 0, 1.0f },
                    W, H, fm.pt, SRC_NOISY, false);
        sprintf_s(tag, "%s sharp qpclip+DC badap show=3", fm.name);
        runModeCase(env, tag, CaseCfg{ 1, 2, 5, 2.0f, -1, true, true, 3, QP_CLIP_DC, 0, 1.0f },
                    W, H, fm.pt, SRC_MIXED, false);

        // KFM_SourceStart (frame mapping), conv=1
        sprintf_s(tag, "%s KFM_SourceStart num=1", fm.name);
        runModeCase(env, tag, CaseCfg{ 2, 0, 0, 0.5f, -1, false, false, 0, QP_SRCSTART, 0, 1.0f },
                    W, H, fm.pt, SRC_BLOCKY, false);
        // KFM_SourceStart with num=2 and frameRateConv 0.5
        sprintf_s(tag, "%s KFM_SourceStart num=2 conv=0.5", fm.name);
        runModeCase(env, tag, CaseCfg{ 2, 0, 0, 0.5f, -1, false, false, 0, QP_SRCSTART2, 0, 0.5f },
                    W, H, fm.pt, SRC_NOISY, false);
        // invalid KFM_SourceStart -> GetFrame throws
        sprintf_s(tag, "%s KFM_SourceStart invalid (expect throw)", fm.name);
        runModeCase(env, tag, CaseCfg{ 2, 0, 0, 0.5f, -1, false, false, 0, QP_SRCSTART_BAD, 0, 1.0f },
                    W, H, fm.pt, SRC_BLOCKY, false, true);

        // alias invocation (upstream name)
        sprintf_s(tag, "%s KDeblock alias q3 str2", fm.name);
        runModeCase(env, tag, CaseCfg{ 3, 2, 0, 0.5f, -1, false, false, 0, QP_CLIP, 0, 1.0f },
                    W, H, fm.pt, SRC_MIXED, true);
    }

    // ---- 64x24 quirk config for the sharpen tail (kf_sharpen window quirk
    // bites at x >= height-1; 64x16 is impossible because the REAL core
    // GaussResize rejects chroma height < 9 — same as upstream's chain) ----
    {
        char tag[160];
        sprintf_s(tag, "YV12 64x24 sharp force_qp q2");
        runModeCase(env, tag, CaseCfg{ 2, 2, 0, 0.5f, 10, false, true, 0, QP_FORCE, 0, 1.0f },
                    64, 24, VideoInfo::CS_YV12, SRC_BLOCKY, false);
        sprintf_s(tag, "YUV420P16 64x24 sharp qpclip show=3");
        runModeCase(env, tag, CaseCfg{ 2, 0, 0, 0.5f, -1, false, true, 3, QP_CLIP, 0, 1.0f },
                    64, 24, VideoInfo::CS_YUV420P16, SRC_NOISY, false);
    }

    // ---- validation: width%%8 must throw (upstream ctor) ----
    {
        try {
            AVSValue srcV = env->Invoke("Eval",
                AVSValue("BlankClip(width=60, height=48, pixel_type=\"YV12\", length=2)"));
            AVSValue args[2] = { srcV };
            try {
                env->Invoke("KDeblockOCL", AVSValue(args, 1));
                printf("[FAIL] width%%8!=0 accepted (expected throw)\n");
                ++g_failures;
            } catch (const AvisynthError&) {
                printf("[PASS] width=60 rejected (width must be multiple of 8)\n");
            }
        } catch (const AvisynthError& e) {
            printf("[FAIL] width-validation setup: %s\n", e.msg);
            ++g_failures;
        }
        ++g_cases;
    }

    // ---- validation: height%%8 must throw (upstream ctor) ----
    {
        try {
            AVSValue srcV = env->Invoke("Eval",
                AVSValue("BlankClip(width=64, height=50, pixel_type=\"YV12\", length=2)"));
            AVSValue args[2] = { srcV };
            try {
                env->Invoke("KDeblockOCL", AVSValue(args, 1));
                printf("[FAIL] height%%8!=0 accepted (expected throw)\n");
                ++g_failures;
            } catch (const AvisynthError&) {
                printf("[PASS] height=50 rejected (height must be multiple of 8)\n");
            }
        } catch (const AvisynthError& e) {
            printf("[FAIL] height-validation setup: %s\n", e.msg);
            ++g_failures;
        }
        ++g_cases;
    }

    // ---- validation: RGB32 must throw (planar YUV required) ----
    {
        try {
            AVSValue srcV = env->Invoke("Eval",
                AVSValue("BlankClip(width=64, height=48, pixel_type=\"RGB32\", length=2)"));
            AVSValue args[2] = { srcV };
            try {
                env->Invoke("KDeblockOCL", AVSValue(args, 1));
                printf("[FAIL] RGB32 accepted (expected throw)\n");
                ++g_failures;
            } catch (const AvisynthError&) {
                printf("[PASS] RGB32 rejected (planar YUV input required)\n");
            }
        } catch (const AvisynthError& e) {
            printf("[FAIL] RGB32-validation setup: %s\n", e.msg);
            ++g_failures;
        }
        ++g_cases;
    }

    printf("\n=== KDeblockOCL: %d cases (%d frame checks), %d failures ===\n",
           g_cases, g_frame_checks, g_failures);
    env->DeleteScriptEnvironment();
    // NOTE: deliberately not FreeLibrary 窶・AviSynth+ keeps internal threads
    // referencing the DLL at shutdown.
    return g_failures ? 1 : 0;
}
