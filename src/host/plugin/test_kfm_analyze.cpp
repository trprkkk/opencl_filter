// test_kfm_analyze.cpp — headless verification of the KFM CombingAnalyze.cu +
// DecombeUCF.cu analysis/reduction AviSynth+ plugin (the seventh host plugin
// of the OpenCL port) against CPU references computed in this exe.
//
// REFERENCE CHOICE (stated, per the task): the reference is computed HERE from
// the same deterministic source-pixel generator the input clips are filled
// with, transcribing the upstream host classes' per-frame assembly
// (CombingAnalyze.cu / DecombeUCF.cu) plus the ALG-VERIFIED kernel semantics
// (python/run_kfm_combinganalyze.py + python/run_kfm_decombeucf.py goldens and
// the sim/*_ref.cpp mirrors).  The reference never reads the plugin's staging:
// it recomputes the arithmetic straight from the source bytes.
//
// Coverage (every pixel/counter of every plane compared where deterministic):
//   KCleanSuper, KPreCycleAnalyze, KPreCycleAnalyzeShow, KContainsCombe,
//   KCombeMask, KRemoveCombe (8/16-bit where upstream), KCFieldDiff,
//   KCFrameDiffDup (conditional), KAnalyzeNoise, KFMSuper, plus alias
//   invocations and ctor-validation throws.
//
// Load path: the exe copies the plugin DLL into the AviSynth+ autoload dir
// (registry HKLM\Software\AviSynth plugins64), creates the script env,
// verifies autoload via FunctionExists, then invokes each filter.
//
// Build: see build_host_analyze.bat.  Usage:
//   test_kfm_analyze.exe <plugin_dll_path> [plugins_dir]
#include <windows.h>

#include "avisynth.h"
#include "kfm_textout_font.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

typedef IScriptEnvironment* (__stdcall* CSE_t)(int);
const AVS_Linkage* AVS_linkage = nullptr;

static LONG WINAPI eh(EXCEPTION_POINTERS* ep) {
    HMODULE m = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)ep->ExceptionRecord->ExceptionAddress, &m);
    char path[MAX_PATH] = "?";
    GetModuleFileNameA(m, path, MAX_PATH);
    fprintf(stderr, "EXC %08lX at %p in %s\n",
            ep->ExceptionRecord->ExceptionCode,
            ep->ExceptionRecord->ExceptionAddress, path);
    fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}

static int g_failures = 0;
static int g_cases = 0;
static int g_checks = 0;

// ---------------------------------------------------------------------------
// deterministic source content (shared by the clips and the reference)
// ---------------------------------------------------------------------------
static uint32_t mix32(uint32_t a) {
    a ^= a >> 16; a *= 0x7feb352du;
    a ^= a >> 15; a *= 0x846ca68bu;
    a ^= a >> 16;
    return a;
}

enum Kind { K_SMOOTH, K_NOISY, K_COMBED, K_FLAT };

static int gen8(int kind, int plane, int x, int y, int n, int w, int h,
                uint32_t salt) {
    uint32_t s = salt + (uint32_t)plane * 0x9E3779B9u + (uint32_t)n * 0x85EBCA6Bu;
    switch (kind) {
    case K_SMOOTH:
        return (int)(((int64_t)x * 255) / (w > 1 ? w - 1 : 1) +
                     ((int64_t)y * 255) / (h > 1 ? h - 1 : 1)) / 2;
    case K_NOISY:
        return (int)(mix32((uint32_t)x * 2654435761u + (uint32_t)y * 40503u + s) & 255u);
    case K_COMBED: {
        int base = (y & 1) ? 205 : 50;
        int d = (int)(mix32((uint32_t)x * 131u + (uint32_t)y * 17u + s) & 63u) - 31;
        int v = base + d;
        return v < 0 ? 0 : (v > 255 ? 255 : v);
    }
    default:
        return 128;
    }
}

static int planeW(const VideoInfo& vi, int plane) {
    return vi.width >> (plane == PLANAR_Y ? 0 : vi.GetPlaneWidthSubsampling(PLANAR_U));
}
static int planeH(const VideoInfo& vi, int plane) {
    return vi.height >> (plane == PLANAR_Y ? 0 : vi.GetPlaneHeightSubsampling(PLANAR_U));
}

static VideoInfo mkvi(int w, int h, int pt, int nf) {
    VideoInfo vi;
    memset(&vi, 0, sizeof vi);
    vi.width = w; vi.height = h; vi.pixel_type = pt; vi.num_frames = nf;
    vi.fps_numerator = 50; vi.fps_denominator = 1;
    return vi;
}

// PLANAR_Y=1, PLANAR_U=2, PLANAR_V=4: NEVER iterate numerically.
static const int kPlanes[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };

class PatternClip : public IClip {
public:
    VideoInfo vi_;
    uint32_t salt_;
    int kind_;
    bool parity_;
    PatternClip(VideoInfo vi, uint32_t salt, int kind, bool parity = false)
        : vi_(vi), salt_(salt), kind_(kind), parity_(parity) {}

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) {
        PVideoFrame f = env->NewVideoFrame(vi_);
        const int px = vi_.ComponentSize();
        const int scale = (px == 2) ? 257 : 1;
        for (int pi_ = 0; pi_ < 3; ++pi_) for (int plane = kPlanes[pi_], q_ = 0; q_ < 1; ++q_) {
            if (vi_.IsY8() && plane != PLANAR_Y) continue;
            const int w = planeW(vi_, plane), h = planeH(vi_, plane);
            BYTE* base = f->GetWritePtr(plane);
            if (!base) continue;
            const int pitch = f->GetPitch(plane);
            for (int y = 0; y < h; ++y) {
                if (px == 1) {
                    uint8_t* row = base + (size_t)y * pitch;
                    for (int x = 0; x < w; ++x)
                        row[x] = (uint8_t)(gen8(kind_, plane, x, y, n, w, h, salt_) * scale);
                } else {
                    uint16_t* row = (uint16_t*)(base + (size_t)y * pitch);
                    for (int x = 0; x < w; ++x)
                        row[x] = (uint16_t)(gen8(kind_, plane, x, y, n, w, h, salt_) * scale);
                }
            }
        }
        return f;
    }
    bool __stdcall GetParity(int) { return parity_; }
    void __stdcall GetAudio(void*, int64_t, int64_t, IScriptEnvironment*) {}
    int __stdcall SetCacheHints(int, int) { return 0; }
    const VideoInfo& __stdcall GetVideoInfo() { return vi_; }
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static int rd(const PVideoFrame& f, int plane, int x, int y, int px) {
    const BYTE* b = f->GetReadPtr(plane) + (size_t)y * f->GetPitch(plane);
    return px == 1 ? (int)b[x] : (int)((const uint16_t*)b)[x];
}

static int rdPair(const PVideoFrame& f, int plane, int e, int r, int lane) {
    const uint8_t* row = f->GetReadPtr(plane) + (size_t)r * f->GetPitch(plane);
    return (int)row[2 * e + lane];
}

static int mirrorIdx(int v, int n) {
    if (v < 0) return -v - 1;
    if (v >= n) return 2 * n - v - 1;
    return v;
}

// source sample at logical (x,y): rows are mirror-mapped (upstream's VPAD
// mirror ring) so out-of-range vertical taps reproduce the pad;
static int srcS(int kind, uint32_t salt, const VideoInfo& vi, int plane,
                int x, int y, int n) {
    const int w = planeW(vi, plane), h = planeH(vi, plane);
    const int scale = (vi.ComponentSize() == 2) ? 257 : 1;
    int yy = mirrorIdx(y, h);
    if (x < 0) x = 0;
    if (x >= w) x = w - 1;
    return gen8(kind, plane, x, yy, n, w, h, salt) * scale;
}

static int clampU8(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }
static int nblocks(int a, int b) { return (a + b - 1) / b; }

static bool chk(const char* tag, const char* what, long long got, long long exp,
                int ctx) {
    ++g_checks;
    if (got != exp) {
        printf("[FAIL] %s %s ctx=%d got=%lld exp=%lld\n", tag, what, ctx, got, exp);
        return false;
    }
    return true;
}

static void report(bool ok, const char* tag) {
    ++g_cases;
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", tag);
    if (!ok) ++g_failures;
}

// ===========================================================================
// KCleanSuper
// ===========================================================================
static bool testCleanSuper(IScriptEnvironment* env, int pt, int kind, int thY,
                           int thC, bool alias) {
    const int W = 32, H = 16, NF = 4;
    VideoInfo vi = mkvi(W, H, pt, NF);
    const uint32_t salt = 0x1234u + (uint32_t)pt + (uint32_t)thY;
    PClip clip = new PatternClip(vi, salt, kind);

    AVSValue args[3] = { AVSValue(clip), thY, thC };
    PClip out;
    try {
        out = env->Invoke(alias ? "KCleanSuper" : "KCleanSuperOCL", AVSValue(args, 3)).AsClip();
    } catch (const AvisynthError& e) {
        printf("[FAIL] KCleanSuper invoke: %s\n", e.msg);
        return false;
    }
    bool ok = true;
    for (int n = 1; n <= 2 && ok; ++n) {
        PVideoFrame f = out->GetFrame(n, env);
        for (int pi_ = 0; pi_ < 3 && ok; ++pi_) for (int plane = kPlanes[pi_], q_ = 0; q_ < 1; ++q_) {
            const int w = planeW(vi, plane), h = planeH(vi, plane);
            const int ew = w / 2;
            const int thr = (plane == PLANAR_Y) ? thY : thC;
            for (int r = 0; r < h && ok; ++r)
                for (int e = 0; e < ew && ok; ++e) {
                    int cx = srcS(kind, salt, vi, plane, 2 * e, r, n);
                    int cy = srcS(kind, salt, vi, plane, 2 * e + 1, r, n);
                    int py = srcS(kind, salt, vi, plane, 2 * e + 1, r, n - 1);
                    int gx = (py <= thr && cy <= thr) ? 0 : cx;
                    int gox = rdPair(f, plane, e, r, 0);
                    int goy = rdPair(f, plane, e, r, 1);
                    if (!chk("KCleanSuper", "x", gox, gx, n * 100 + plane)) ok = false;
                    if (!chk("KCleanSuper", "y", goy, cy, n * 100 + plane)) ok = false;
                }
        }
    }
    return ok;
}

// ===========================================================================
// KPreCycleAnalyze
// ===========================================================================
static bool testPreCycle(IScriptEnvironment* env, int kind, int parity, int thMY,
                         int thSY, int thMC, int thSC, bool alias) {
    const int W = 32, H = 16, NF = 4;
    VideoInfo vi = mkvi(W, H, VideoInfo::CS_YV12, NF);
    const uint32_t salt = 0x5A5Au + (uint32_t)kind + (uint32_t)thMY;
    PClip clip = new PatternClip(vi, salt, kind, parity != 0);

    AVSValue args[5] = { AVSValue(clip), thMY, thSY, thMC, thSC };
    PClip out;
    try {
        out = env->Invoke(alias ? "KPreCycleAnalyze" : "KPreCycleAnalyzeOCL",
                          AVSValue(args, 5)).AsClip();
    } catch (const AvisynthError& e) {
        printf("[FAIL] KPreCycleAnalyze invoke: %s\n", e.msg);
        return false;
    }
    // FMCount[2] census over cells (1..W-1, 1..H-1) of the pair-packed planes.
    long long fm[6] = { 0, 0, 0, 0, 0, 0 };
    const int np = parity ? 0 : 1;  // !parity
    for (int i = 0; i < 2; ++i) {
        const int frame = i;  // n = 0 -> frames 0 and 1
        const int slot = i ^ np;
        for (int pi = 0; pi < 3; ++pi) {
            const int plane = kPlanes[pi];
            int thM = (plane == PLANAR_Y) ? thMY : thMC;
            int thS = (plane == PLANAR_Y) ? thSY : thSC;
            int thLS = (plane == PLANAR_Y) ? thSY * 3 : thSC * 3;
            const int ew = planeW(vi, plane) / 2;  // uchar2 element width
            const int eh = planeH(vi, plane);
            for (int by = 1; by < eh; ++by)
                for (int bx = 1; bx < ew; ++bx) {
                    int ex = srcS(kind, salt, vi, plane, 2 * bx + 0, by, frame);
                    int ey = srcS(kind, salt, vi, plane, 2 * bx + 1, by, frame);
                    if (ey >= thM) fm[slot * 3 + 0]++;
                    if (ex >= thS) fm[slot * 3 + 1]++;
                    if (ex >= thLS) fm[slot * 3 + 2]++;
                }
        }
    }
    bool ok = true;
    PVideoFrame f = out->GetFrame(0, env);
    const int* got = (const int*)f->GetReadPtr();
    for (int k = 0; k < 6 && ok; ++k)
        if (!chk("KPreCycleAnalyze", "fm", got[k], fm[k], k)) ok = false;
    return ok;
}

// ===========================================================================
// KPreCycleAnalyzeShow (text overlay; bit-exact font port)
// ===========================================================================
static void drawTextPlaneRef(uint8_t* dst, int width, int height, int pitch,
                             const char* charmap, int mapw, int logx, int logy,
                             int shift, bool isUV) {
    for (int bx = 0; bx < mapw; ++bx)
        for (int ty = 0; ty < 20; ty += (1 << logy))
            for (int tx = 0; tx < 10; tx += (1 << logx)) {
                int x = (tx + bx * 10) >> logx;
                int y = (ty) >> logy;
                if (x < width && y < height) {
                    int c = charmap[bx] - ' ';
                    if (c >= 0 && c <= 'z' - ' ') {
                        if (kfmtext::kFont[c][ty] & (1 << (15 - tx)))
                            dst[x + (size_t)y * pitch] = (uint8_t)(isUV ? (128 << shift) : (235 << shift));
                        else
                            dst[x + (size_t)y * pitch] =
                                (uint8_t)(((int)dst[x + (size_t)y * pitch] +
                                           (isUV ? (128 << shift) : 0)) >> 1);
                    }
                }
            }
}

static bool testPreCycleShow(IScriptEnvironment* env, bool alias) {
    const int W = 64, H = 32, NF = 3;
    VideoInfo vi = mkvi(W, H, VideoInfo::CS_YV12, NF);
    PClip srcClip = new PatternClip(vi, 0xBEEFu, K_SMOOTH);
    VideoInfo fvi = mkvi(6, 1, VideoInfo::CS_BGR32, NF);
    class FmClip : public IClip {
    public:
        VideoInfo vi_;
        FmClip(VideoInfo v) : vi_(v) {}
        PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) {
            PVideoFrame f = env->NewVideoFrame(vi_);
            int* p = (int*)f->GetWritePtr();
            p[0] = 11 + n; p[1] = 22; p[2] = 33;
            p[3] = 44; p[4] = 55; p[5] = 66;
            return f;
        }
        bool __stdcall GetParity(int) { return false; }
        void __stdcall GetAudio(void*, int64_t, int64_t, IScriptEnvironment*) {}
        int __stdcall SetCacheHints(int, int) { return 0; }
        const VideoInfo& __stdcall GetVideoInfo() { return vi_; }
    };
    PClip fm = new FmClip(fvi);
    AVSValue args[2] = { AVSValue(fm), AVSValue(srcClip) };
    PClip out;
    try {
        out = env->Invoke(alias ? "KPreCycleAnalyzeShow" : "KPreCycleAnalyzeShowOCL",
                          AVSValue(args, 2)).AsClip();
    } catch (const AvisynthError& e) {
        printf("[FAIL] KPreCycleAnalyzeShow invoke: %s\n", e.msg);
        return false;
    }
    bool ok = true;
    for (int n = 0; n < 2 && ok; ++n) {
        PVideoFrame f = out->GetFrame(n, env);
        char buf[100];
        sprintf(buf, "Pre: [0](%d,%d,%d) [1](%d,%d,%d)", 11 + n, 22, 33, 44, 55, 66);
        int maxlen = (int)strlen(buf);
        int textW = vi.width; if (textW > maxlen * 10) textW = maxlen * 10;
        int textH = vi.height; if (textH > 20) textH = 20;
        int logx = (planeW(vi, PLANAR_U) < textW) ? 1 : 0;
        int logy = (planeH(vi, PLANAR_U) < textH) ? 1 : 0;
        for (int pi_ = 0; pi_ < 3 && ok; ++pi_) for (int plane = kPlanes[pi_], q_ = 0; q_ < 1; ++q_) {
            const int w = planeW(vi, plane), h = planeH(vi, plane);
            std::vector<uint8_t> ref((size_t)w * h);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    ref[(size_t)y * w + x] = (uint8_t)srcS(K_SMOOTH, 0xBEEFu, vi, plane, x, y, n);
            if (plane == PLANAR_Y)
                drawTextPlaneRef(ref.data(), textW, textH, w, buf, maxlen, 0, 0, 0, false);
            else
                drawTextPlaneRef(ref.data(), textW >> logx, textH >> logy, w, buf, maxlen,
                                 logx, logy, 0, true);
            for (int y = 0; y < h && ok; ++y)
                for (int x = 0; x < w && ok; ++x)
                    if (!chk("KPreCycleAnalyzeShow", "px", rd(f, plane, x, y, 1),
                             ref[(size_t)y * w + x], n * 100 + plane)) ok = false;
        }
    }
    return ok;
}

// ===========================================================================
// KContainsCombe
// ===========================================================================
static bool testContainsCombe(IScriptEnvironment* env, int kind, int special,
                              bool alias) {
    const int W = 20, H = 12, NF = 3;
    VideoInfo vi = mkvi(W, H, VideoInfo::CS_Y8, NF);
    const uint32_t salt = 0xC0DEu + (uint32_t)kind;
    PClip clip = new PatternClip(vi, salt, kind);
    if (special == 1) {  // all zero
        clip = new PatternClip(vi, salt, K_FLAT);  // flat=128 -> not zero; use custom
        class ZeroClip : public IClip {
        public:
            VideoInfo vi_;
            ZeroClip(VideoInfo v) : vi_(v) {}
            PVideoFrame __stdcall GetFrame(int, IScriptEnvironment* env) {
                PVideoFrame f = env->NewVideoFrame(vi_);
                memset(f->GetWritePtr(), 0, (size_t)f->GetPitch() * f->GetHeight());
                return f;
            }
            bool __stdcall GetParity(int) { return false; }
            void __stdcall GetAudio(void*, int64_t, int64_t, IScriptEnvironment*) {}
            int __stdcall SetCacheHints(int, int) { return 0; }
            const VideoInfo& __stdcall GetVideoInfo() { return vi_; }
        };
        clip = new ZeroClip(vi);
    } else if (special == 2) {  // border-only hit at (0,0)
        class BorderClip : public IClip {
        public:
            VideoInfo vi_;
            BorderClip(VideoInfo v) : vi_(v) {}
            PVideoFrame __stdcall GetFrame(int, IScriptEnvironment* env) {
                PVideoFrame f = env->NewVideoFrame(vi_);
                memset(f->GetWritePtr(), 0, (size_t)f->GetPitch() * f->GetHeight());
                f->GetWritePtr()[0] = 200;
                return f;
            }
            bool __stdcall GetParity(int) { return false; }
            void __stdcall GetAudio(void*, int64_t, int64_t, IScriptEnvironment*) {}
            int __stdcall SetCacheHints(int, int) { return 0; }
            const VideoInfo& __stdcall GetVideoInfo() { return vi_; }
        };
        clip = new BorderClip(vi);
    }

    AVSValue args[1] = { AVSValue(clip) };
    PClip out;
    try {
        out = env->Invoke(alias ? "KContainsCombe" : "KContainsCombeOCL",
                          AVSValue(args, 1)).AsClip();
    } catch (const AvisynthError& e) {
        printf("[FAIL] KContainsCombe invoke: %s\n", e.msg);
        return false;
    }
    bool ok = true;
    for (int n = 0; n < 2 && ok; ++n) {
        PVideoFrame f = out->GetFrame(n, env);
        int got = *(const int*)f->GetReadPtr();
        int exp = 0;
        if (special == 2) exp = 1;
        else if (special == 0)
            for (int y = 0; y < H && !exp; ++y)
                for (int x = 0; x < W && !exp; ++x)
                    if (srcS(kind, salt, vi, PLANAR_Y, x, y, n)) exp = 1;
        if (!chk("KContainsCombe", "int", got, exp, n)) ok = false;
    }
    return ok;
}

// ===========================================================================
// KCombeMask
// ===========================================================================
static void bilinRef(const std::vector<uint8_t>& pad, int pw, int fwidth, int fheight,
                     int scalew, int scaleh, std::vector<uint8_t>& outv, int dw, int dh) {
    const int shw = (scalew == 8) ? 3 : 2;
    const int shh = (scaleh == 8) ? 3 : 2;
    const int hw = scalew / 2, hh = scaleh / 2;
    const int dwW = scalew * fwidth;
    std::vector<uint8_t> hout((size_t)dwW * (fheight + 2));
    for (int y = 0; y < fheight + 2; ++y)
        for (int x = 0; x < dwW; ++x) {
            int x0 = (x - hw) >> shw;
            int c0 = ((x0 + 1) << shw) - (x - hw);
            int c1 = scalew - c0;
            hout[(size_t)y * dwW + x] =
                (uint8_t)((pad[(size_t)y * pw + x0 + 1] * c0 +
                           pad[(size_t)y * pw + x0 + 2] * c1 + hw) >> shw);
        }
    outv.assign((size_t)dw * dh, 0);
    for (int y = 0; y < dh; ++y)
        for (int x = 0; x < dw; ++x) {
            int y0 = (y - hh) >> shh;
            int c0 = ((y0 + 1) << shh) - (y - hh);
            int c1 = scaleh - c0;
            outv[(size_t)y * dw + x] =
                (uint8_t)((hout[(size_t)(y0 + 1) * dwW + x] * c0 +
                           hout[(size_t)(y0 + 2) * dwW + x] * c1 + hh) >> shh);
        }
}

static bool testCombeMask(IScriptEnvironment* env, int pt, int kind, bool alias) {
    const int W = 64, H = 48, NF = 3;
    VideoInfo vi = mkvi(W, H, pt, NF);
    const int lx = vi.GetPlaneWidthSubsampling(PLANAR_U);
    const int ly = vi.GetPlaneHeightSubsampling(PLANAR_U);
    const int fwidth = W / 8, fheight = H / 8;
    const int FW = fwidth + 8, FH = fheight + 4;
    VideoInfo fvi = mkvi(FW, FH, VideoInfo::CS_Y8, NF);
    const uint32_t fsalt = 0xF00Du + (uint32_t)kind;
    PClip srcClip = new PatternClip(vi, 0xAAAAu, K_SMOOTH);
    PClip flagClip = new PatternClip(fvi, fsalt, kind);

    AVSValue args[2] = { AVSValue(srcClip), AVSValue(flagClip) };
    PClip out;
    try {
        out = env->Invoke(alias ? "KCombeMask" : "KCombeMaskOCL",
                          AVSValue(args, 2)).AsClip();
    } catch (const AvisynthError& e) {
        printf("[FAIL] KCombeMask invoke: %s\n", e.msg);
        return false;
    }
    std::vector<uint8_t> roi((size_t)fwidth * fheight);
    for (int y = 0; y < fheight; ++y)
        for (int x = 0; x < fwidth; ++x)
            roi[(size_t)y * fwidth + x] =
                (uint8_t)srcS(kind, fsalt, fvi, PLANAR_Y, x + 4, y + 2, 0);
    const int pw = fwidth + 2;
    std::vector<uint8_t> pad((size_t)pw * (fheight + 2));
    for (int r = 0; r < fheight + 2; ++r)
        for (int c = 0; c < pw; ++c)
            pad[(size_t)r * pw + c] =
                roi[(size_t)mirrorIdx(r - 1, fheight) * fwidth + mirrorIdx(c - 1, fwidth)];

    std::vector<uint8_t> eY, eU, eV;
    bilinRef(pad, pw, fwidth, fheight, 8, 8, eY, W, H);
    const int wUV = W >> lx, hUV = H >> ly;
    const int suw = wUV / fwidth, suh = hUV / fheight;
    bilinRef(pad, pw, fwidth, fheight, suw, suh, eU, wUV, hUV);
    bilinRef(pad, pw, fwidth, fheight, suw, suh, eV, wUV, hUV);

    bool ok = true;
    PVideoFrame f = out->GetFrame(0, env);
    for (int y = 0; y < H && ok; ++y)
        for (int x = 0; x < W && ok; ++x)
            if (!chk("KCombeMask", "Y", rd(f, PLANAR_Y, x, y, 1), eY[(size_t)y * W + x], y * 100 + x)) ok = false;
    for (int y = 0; y < hUV && ok; ++y)
        for (int x = 0; x < wUV && ok; ++x) {
            if (!chk("KCombeMask", "U", rd(f, PLANAR_U, x, y, 1), eU[(size_t)y * wUV + x], y * 100 + x)) ok = false;
            if (!chk("KCombeMask", "V", rd(f, PLANAR_V, x, y, 1), eV[(size_t)y * wUV + x], y * 100 + x)) ok = false;
        }
    return ok;
}

// ===========================================================================
// KRemoveCombe
// ===========================================================================
static bool testRemoveCombe(IScriptEnvironment* env, int pt, int kind, float thY,
                            float thC, bool alias) {
    const int W = 32, Hout = 24, NF = 3;
    VideoInfo padvi = mkvi(W, Hout + 8, pt, NF);
    VideoInfo supvi = mkvi(W / 2, Hout, VideoInfo::CS_YV12, NF);
    const uint32_t psalt = 0x1111u + (uint32_t)kind, ssalt = 0x2222u;
    PClip padClip = new PatternClip(padvi, psalt, kind);
    PClip supClip = new PatternClip(supvi, ssalt, kind);
    const int px = padvi.ComponentSize();

    AVSValue args[4] = { AVSValue(padClip), AVSValue(supClip), (double)thY, (double)thC };
    PClip out;
    try {
        out = env->Invoke(alias ? "KRemoveCombe" : "KRemoveCombeOCL", AVSValue(args, 4)).AsClip();
    } catch (const AvisynthError& e) {
        printf("[FAIL] KRemoveCombe invoke: %s\n", e.msg);
        return false;
    }
    const int ly = (pt == VideoInfo::CS_YV12 || pt == VideoInfo::CS_YUV420P16) ? 1 : 0;
    bool ok = true;
    for (int n = 0; n < 2 && ok; ++n) {
        PVideoFrame f = out->GetFrame(n, env);
        for (int pi_ = 0; pi_ < 3 && ok; ++pi_) for (int plane = kPlanes[pi_], q_ = 0; q_ < 1; ++q_) {
            const int plx = (plane == PLANAR_Y) ? 0 : (pt == VideoInfo::CS_YV24 ||
                            pt == VideoInfo::CS_YUV444P16 ? 0 : 1);
            const int w = W >> plx, h = Hout >> (plane == PLANAR_Y ? 0 : ly);
            const int voff = (plane == PLANAR_Y) ? 4 : (4 >> ly);
            const int thr = (int)((plane == PLANAR_Y) ? thY : thC);
            for (int y = 0; y < h && ok; ++y)
                for (int x = 0; x < w && ok; ++x) {
                    // combe: .x lane of the pair-packed super plane element (x>>2, y>>2)
                    int score = srcS(kind, ssalt, supvi, plane, 2 * (x >> 2), y >> 2, n);
                    int v;
                    if (score >= thr) {
                        int a = srcS(kind, psalt, padvi, plane, x, y - 1 + voff, n);
                        int b = srcS(kind, psalt, padvi, plane, x, y + voff, n);
                        int c = srcS(kind, psalt, padvi, plane, x, y + 1 + voff, n);
                        v = (a + 2 * b + c + 2) >> 2;
                    } else {
                        v = srcS(kind, psalt, padvi, plane, x, y + voff, n);
                    }
                    if (!chk("KRemoveCombe", "px", rd(f, plane, x, y, px), v, n * 100 + plane)) ok = false;
                }
        }
    }
    return ok;
}

// ===========================================================================
// KCFieldDiff / KCFrameDiffDup (conditional)
// ===========================================================================
static int scaleParam(float th, int bits) {
    return (int)(th * (1 << (bits - 8)) + 0.5f);
}

static bool testFieldDiff(IScriptEnvironment* env, int pt, int kind, float nt,
                          bool chroma, bool alias) {
    const int W = 32, H = 24, NF = 4;
    VideoInfo vi = mkvi(W, H, pt, NF);
    const uint32_t salt = 0x77u + (uint32_t)kind;
    PClip clip = new PatternClip(vi, salt, kind);
    const int bits = vi.BitsPerComponent();
    const int nt6 = scaleParam(nt * 6, bits);

    bool ok = true;
    for (int n = 0; n < 2 && ok; ++n) {
        env->SetGlobalVar("current_frame", AVSValue(n));
        AVSValue args[3] = { AVSValue(clip), (double)nt, chroma };
        AVSValue rv;
        try {
            rv = env->Invoke(alias ? "KCFieldDiff" : "KCFieldDiffOCL", AVSValue(args, 3));
        } catch (const AvisynthError& e) {
            printf("[FAIL] KCFieldDiff invoke: %s\n", e.msg);
            return false;
        }
        unsigned long long raw = 0;
        for (int pi_ = 0; pi_ < 3; ++pi_) for (int plane = kPlanes[pi_], q_ = 0; q_ < 1; ++q_) {
            if (plane != PLANAR_Y && !chroma) continue;
            const int w = planeW(vi, plane), h = planeH(vi, plane);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    int a = srcS(kind, salt, vi, plane, x, y - 2, n);
                    int b = srcS(kind, salt, vi, plane, x, y - 1, n);
                    int c = srcS(kind, salt, vi, plane, x, y, n);
                    int d = srcS(kind, salt, vi, plane, x, y + 1, n);
                    int e = srcS(kind, salt, vi, plane, x, y + 2, n);
                    int combe = abs(a + 4 * c + e - 3 * (b + d));
                    if (combe > nt6) raw += (unsigned)combe;
                }
        }
        raw /= 6;
        double exp = (double)(raw >> (bits - 8));
        double got = rv.AsFloat();
        if (!chk("KCFieldDiff", "val", (long long)(got * 1e6 + 0.5),
                 (long long)(exp * 1e6 + 0.5), n)) ok = false;
    }
    return ok;
}

static bool testFrameDiffDup(IScriptEnvironment* env, int pt, int kind, bool chroma,
                             int blocksize, bool alias) {
    const int W = 48, H = 32, NF = 4;
    VideoInfo vi = mkvi(W, H, pt, NF);
    const uint32_t salt = 0x99u + (uint32_t)kind;
    PClip clip = new PatternClip(vi, salt, kind);
    const int bits = vi.BitsPerComponent();
    const int lu = vi.GetPlaneWidthSubsampling(PLANAR_U);
    const int lv = vi.GetPlaneHeightSubsampling(PLANAR_U);
    const int bs = blocksize;
    const int blocks_w = nblocks(W, bs), blocks_h = nblocks(H, bs);
    const int th_z = 256 / (bs * (bs / 4));
    const int th_uv_z = th_z * (chroma ? (1 << (lu + lv)) : 1);
    const int block_align = std::max(4, th_uv_z);
    const int block_pitch = nblocks(blocks_w, block_align) * block_align;
    const int len = block_pitch * blocks_h;

    bool ok = true;
    for (int n = 0; n < 2 && ok; ++n) {
        env->SetGlobalVar("current_frame", AVSValue(n));
        AVSValue args[3] = { AVSValue(clip), chroma, blocksize };
        AVSValue rv;
        try {
            rv = env->Invoke(alias ? "KCFrameDiffDup" : "KCFrameDiffDupOCL", AVSValue(args, 3));
        } catch (const AvisynthError& e) {
            printf("[FAIL] KCFrameDiffDup invoke: %s\n", e.msg);
            return false;
        }
        std::vector<int> ab((size_t)len, 0), sg((size_t)len, 0);
        const int n0 = std::max(0, std::min(n - 1, NF - 1));
        const int n1 = std::max(0, std::min(n, NF - 1));
        for (int pi_ = 0; pi_ < 3; ++pi_) for (int plane = kPlanes[pi_], q_ = 0; q_ < 1; ++q_) {
            if (plane != PLANAR_Y && !chroma) continue;
            const int w = planeW(vi, plane), h = planeH(vi, plane);
            for (int by = 0; by < blocks_h; ++by)
                for (int bx = 0; bx < blocks_w; ++bx) {
                    int asum = 0, ssum = 0;
                    for (int ty = 0; ty < bs; ++ty) {
                        int yy = by * bs + ty;
                        for (int tx = 0; tx < bs; ++tx) {
                            int xx = bx * bs + tx;
                            if (xx >= w || yy >= h) continue;
                            int s0 = srcS(kind, salt, vi, plane, xx, yy, n0);
                            int s1 = srcS(kind, salt, vi, plane, xx, yy, n1);
                            asum += abs(s0 - s1);
                            ssum += s0 - s1;
                        }
                    }
                    ab[bx + by * block_pitch] += asum;
                    sg[bx + by * block_pitch] += ssum;
                }
        }
        int best = 0;
        const int blocks_w4 = blocks_w >> 2;
        const int block_pitch4 = block_pitch >> 2;
        for (int y = 0; y < blocks_h; ++y)
            for (int x = 0; x < blocks_w4; ++x) {
                int c4 = (x + y * block_pitch4) * 4;
                for (int k = 0; k < 4; ++k) {
                    int m = ab[c4 + k] + sg[c4 + k] * 4;
                    if (m > best) best = m;
                }
            }
        double exp = best / (2.0 * (235 << (bits - 8)) * bs * bs) * 100.0;
        double got = rv.AsFloat();
        if (!chk("KCFrameDiffDup", "val", (long long)(got * 1e6 + 0.5),
                 (long long)(exp * 1e6 + 0.5), n * 100 + blocksize)) ok = false;
    }
    return ok;
}

// ===========================================================================
// KAnalyzeNoise
// ===========================================================================
struct NoiseResultRef {
    unsigned long long noise0, noise1, noiseR0, noiseR1, diff0, diff1;
};

static bool testAnalyzeNoise(IScriptEnvironment* env, int kind, bool useSuper,
                             bool alias) {
    const int W = 32, H = 24, NF = 6;
    VideoInfo vi = mkvi(W, H, VideoInfo::CS_YV12, NF);
    VideoInfo nvi = mkvi(W, H, VideoInfo::CS_YV12, 2 * NF + 3);
    VideoInfo pvi = mkvi(W, H + 8, VideoInfo::CS_YV12, NF);
    const uint32_t ssalt = 0x3333u, nsalt = 0x4444u, psalt = 0x5555u;
    PClip srcClip = new PatternClip(vi, ssalt, kind);
    PClip noiseClip = new PatternClip(nvi, nsalt, kind);
    PClip padClip; if (useSuper) padClip = new PatternClip(pvi, psalt, kind);

    AVSValue args[3] = { AVSValue(srcClip), AVSValue(noiseClip),
                         useSuper ? AVSValue(padClip) : AVSValue() };
    PClip out;
    try {
        out = env->Invoke(alias ? "KAnalyzeNoise" : "KAnalyzeNoiseOCL", AVSValue(args, 3)).AsClip();
    } catch (const AvisynthError& e) {
        printf("[FAIL] KAnalyzeNoise invoke: %s\n", e.msg);
        return false;
    }
    const int n = 0;
    NoiseResultRef exp[2];
    memset(exp, 0, sizeof exp);

    // AnalyzeNoise (noise clip is NOT padded; raw frames)
    for (int pi_ = 0; pi_ < 3; ++pi_) for (int plane = kPlanes[pi_], q_ = 0; q_ < 1; ++q_) {
        int rec = (plane == PLANAR_Y) ? 0 : 1;
        const int w = planeW(vi, plane), h = planeH(vi, plane);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                int a = gen8(kind, plane, x, y, 2 * n + 0, planeW(nvi, plane), planeH(nvi, plane), nsalt);
                int b = gen8(kind, plane, x, y, 2 * n + 1, planeW(nvi, plane), planeH(nvi, plane), nsalt);
                int c = gen8(kind, plane, x, y, 2 * n + 2, planeW(nvi, plane), planeH(nvi, plane), nsalt);
                exp[rec].noise0 += abs(a - 128);
                exp[rec].noise1 += abs(b - 128);
                exp[rec].noiseR0 += abs(b - a);
                exp[rec].noiseR1 += abs(c - b);
            }
    }
    // AnalyzeDiff
    for (int pi_ = 0; pi_ < 3; ++pi_) for (int plane = kPlanes[pi_], q_ = 0; q_ < 1; ++q_) {
        int rec = (plane == PLANAR_Y) ? 0 : 1;
        const int w = planeW(vi, plane), h = planeH(vi, plane);
        const int ly = vi.GetPlaneHeightSubsampling(PLANAR_U);
        const int voff = (plane == PLANAR_Y) ? 4 : (4 >> ly);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                int a, b, c, d, e, a2, b2, c2, d2, e2;
                if (useSuper) {
                    a  = srcS(kind, psalt, pvi, plane, x, y - 2 + voff, n);
                    b  = srcS(kind, psalt, pvi, plane, x, y - 1 + voff, n);
                    c  = srcS(kind, psalt, pvi, plane, x, y + voff, n);
                    d  = srcS(kind, psalt, pvi, plane, x, y + 1 + voff, n);
                    e  = srcS(kind, psalt, pvi, plane, x, y + 2 + voff, n);
                    a2 = srcS(kind, psalt, pvi, plane, x, y - 2 + voff, n + 1);
                    b2 = srcS(kind, psalt, pvi, plane, x, y - 1 + voff, n + 1);
                    c2 = srcS(kind, psalt, pvi, plane, x, y + voff, n + 1);
                    d2 = srcS(kind, psalt, pvi, plane, x, y + 1 + voff, n + 1);
                    e2 = srcS(kind, psalt, pvi, plane, x, y + 2 + voff, n + 1);
                } else {
                    a  = srcS(kind, ssalt, vi, plane, x, y - 2, n);
                    b  = srcS(kind, ssalt, vi, plane, x, y - 1, n);
                    c  = srcS(kind, ssalt, vi, plane, x, y, n);
                    d  = srcS(kind, ssalt, vi, plane, x, y + 1, n);
                    e  = srcS(kind, ssalt, vi, plane, x, y + 2, n);
                    a2 = srcS(kind, ssalt, vi, plane, x, y - 2, n + 1);
                    b2 = srcS(kind, ssalt, vi, plane, x, y - 1, n + 1);
                    c2 = srcS(kind, ssalt, vi, plane, x, y, n + 1);
                    d2 = srcS(kind, ssalt, vi, plane, x, y + 1, n + 1);
                    e2 = srcS(kind, ssalt, vi, plane, x, y + 2, n + 1);
                }
                exp[rec].diff0 += (unsigned)abs(a + 4 * c + e - 3 * (b + d));
                int q0, q1, q2, q3, q4;
                if (y & 1) { q0 = a; q1 = b2; q2 = c; q3 = d2; q4 = e; }
                else       { q0 = a2; q1 = b; q2 = c2; q3 = d; q4 = e2; }
                exp[rec].diff1 += (unsigned)abs(q0 + 4 * q2 + q4 - 3 * (q1 + q3));
            }
    }
    PVideoFrame f = out->GetFrame(n, env);
    const NoiseResultRef* got = (const NoiseResultRef*)f->GetReadPtr();
    bool ok = true;
    for (int rec = 0; rec < 2 && ok; ++rec) {
        if (!chk("KAnalyzeNoise", "n0", got[rec].noise0, exp[rec].noise0, rec)) ok = false;
        if (!chk("KAnalyzeNoise", "n1", got[rec].noise1, exp[rec].noise1, rec)) ok = false;
        if (!chk("KAnalyzeNoise", "nR0", got[rec].noiseR0, exp[rec].noiseR0, rec)) ok = false;
        if (!chk("KAnalyzeNoise", "nR1", got[rec].noiseR1, exp[rec].noiseR1, rec)) ok = false;
        if (!chk("KAnalyzeNoise", "d0", got[rec].diff0, exp[rec].diff0, rec)) ok = false;
        if (!chk("KAnalyzeNoise", "d1", got[rec].diff1, exp[rec].diff1, rec)) ok = false;
    }
    return ok;
}

// ===========================================================================
// KFMSuper
// ===========================================================================
static int combe8(int L[8]) {
    int diff8 = abs(L[0] - L[7]);
    int dT = abs(L[0]-L[1])+abs(L[1]-L[2])+abs(L[2]-L[3])+abs(L[3]-L[4])+
             abs(L[4]-L[5])+abs(L[5]-L[6])+abs(L[6]-L[7]) - diff8;
    int dE = abs(L[0]-L[2])+abs(L[2]-L[4])+abs(L[4]-L[6])+abs(L[6]-L[7]) - diff8;
    int dO = abs(L[0]-L[1])+abs(L[1]-L[3])+abs(L[3]-L[5])+abs(L[5]-L[7]) - diff8;
    return dT - dE - dO;
}
static int diff8(int L[8]) {
    return abs(L[0]-L[1])+abs(L[2]-L[3])+abs(L[4]-L[5])+abs(L[6]-L[7]);
}

// Packed flag element grids for one plane/frame: geometry from the SOURCE vi
// (nBlkX = srcPlaneW/4, nBlkY = srcPlaneH/4), pixels from the pad clip (row
// offset VPAD / VPAD>>ly).
static void superRefPlane(int kind, uint32_t psalt, const VideoInfo& svi,
                          const VideoInfo& pvi, int plane, int n, int parity,
                          int shift, std::vector<uint8_t>& fx,
                          std::vector<uint8_t>& fy, int& nbx, int& nby,
                          int& fpx, int& fpy) {
    const int w = planeW(svi, plane), h = planeH(svi, plane);
    const int ly = svi.GetPlaneHeightSubsampling(PLANAR_U);
    const int voff = (plane == PLANAR_Y) ? 4 : (4 >> ly);
    nbx = w / 4; nby = h / 4;
    fpx = w / 4; fpy = h / 2;
    fx.assign((size_t)fpx * fpy, 0);
    fy.assign((size_t)fpx * fpy, 0);
    for (int by = 0; by < nby - 1; ++by)
        for (int bx = 0; bx < nbx - 1; ++bx) {
            int x0 = bx * 4, y0 = by * 4;
            int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int tx = 0; tx < 8; ++tx) {
                int x = x0 + tx;
                int f0[8], f1[8];
                for (int k = 0; k < 8; ++k) {
                    f0[k] = srcS(kind, psalt, pvi, plane, x, y0 + k + voff, n);
                    f1[k] = srcS(kind, psalt, pvi, plane, x, y0 + k + voff, n + 1);
                }
                int t0, t2;
                if (parity) {
                    t0 = combe8(f0);
                    int m[8] = { f1[0], f0[1], f1[2], f0[3], f1[4], f0[5], f1[6], f0[7] };
                    t2 = combe8(m);
                } else {
                    t2 = combe8(f0);
                    int m[8] = { f0[0], f1[1], f0[2], f1[3], f0[4], f1[5], f0[6], f1[7] };
                    t0 = combe8(m);
                }
                int a[8] = { f0[0], f1[0], f0[2], f1[2], f0[4], f1[4], f0[6], f1[6] };
                int b[8] = { f0[1], f1[1], f0[3], f1[3], f0[5], f1[5], f0[7], f1[7] };
                s0 += t0; s1 += diff8(a); s2 += t2; s3 += diff8(b);
            }
            int c = bx + 1;
            int r0 = 2 * (by + 1), r1 = r0 + 1;
            fx[(size_t)c + (size_t)r0 * fpx] = (uint8_t)clampU8(s0 >> shift);
            fy[(size_t)c + (size_t)r0 * fpx] = (uint8_t)clampU8(s1 >> shift);
            fx[(size_t)c + (size_t)r1 * fpx] = (uint8_t)clampU8(s2 >> shift);
            fy[(size_t)c + (size_t)r1 * fpx] = (uint8_t)clampU8(s3 >> shift);
        }
}

static bool testFMSuper(IScriptEnvironment* env, int pt, int kind, bool parity,
                        bool alias) {
    const int W = 32, H = 32, NF = 4;
    VideoInfo vi = mkvi(W, H, pt, NF);
    VideoInfo pvi = mkvi(W, H + 8, pt, NF);
    const uint32_t psalt = 0x6666u;
    PClip mainClip = new PatternClip(vi, 0u, kind, parity);
    PClip padClip = new PatternClip(pvi, psalt, kind, parity);
    AVSValue args[2] = { AVSValue(mainClip), AVSValue(padClip) };
    PClip out;
    try {
        out = env->Invoke(alias ? "KFMSuper" : "KFMSuperOCL", AVSValue(args, 2)).AsClip();
    } catch (const AvisynthError& e) {
        printf("[FAIL] KFMSuper invoke: %s\n", e.msg);
        return false;
    }
    const int bits = vi.BitsPerComponent();
    const int shift = bits - 8 + 4;
    const int par = parity ? 1 : 0;
    bool ok = true;
    const int n = 0;
    std::vector<uint8_t> fxY, fyY, fxU, fyU, fxV, fyV;
    int nbx, nby, fpx, fpy, nbxU, nbyU, fpxU, fpyU;
    superRefPlane(kind, psalt, vi, pvi, PLANAR_Y, n, par, shift, fxY, fyY, nbx, nby, fpx, fpy);
    superRefPlane(kind, psalt, vi, pvi, PLANAR_U, n, par, shift, fxU, fyU, nbxU, nbyU, fpxU, fpyU);
    superRefPlane(kind, psalt, vi, pvi, PLANAR_V, n, par, shift, fxV, fyV, nbxU, nbyU, fpxU, fpyU);

    PVideoFrame f0 = out->GetFrame(2 * n, env);
    PVideoFrame f1 = out->GetFrame(2 * n + 1, env);
    struct PL { int plane; std::vector<uint8_t>*fx; std::vector<uint8_t>*fy; int fpx; int nbx; int nby; };
    PL pls[3] = {
        { PLANAR_Y, &fxY, &fyY, fpx, nbx, nby },
        { PLANAR_U, &fxU, &fyU, fpxU, nbxU, nbyU },
        { PLANAR_V, &fxV, &fyV, fpxU, nbxU, nbyU },
    };
    bool matched = false, any = false;
    for (int order = 0; order < 2 && !matched; ++order) {
        PVideoFrame fa = order ? f1 : f0;  // even packed rows
        PVideoFrame fb = order ? f0 : f1;  // odd packed rows
        bool good = true;
        for (int pi = 0; pi < 3 && good; ++pi) {
            PL& pl = pls[pi];
            for (int by = 1; by <= pl.nby - 1 && good; ++by)
                for (int c = 1; c <= pl.nbx - 1 && good; ++c) {
                    int evenX = (*pl.fx)[(size_t)c + (size_t)(2 * by) * pl.fpx];
                    int evenY = (*pl.fy)[(size_t)c + (size_t)(2 * by) * pl.fpx];
                    int oddX = (*pl.fx)[(size_t)c + (size_t)(2 * by + 1) * pl.fpx];
                    int oddY = (*pl.fy)[(size_t)c + (size_t)(2 * by + 1) * pl.fpx];
                    if (rdPair(fa, pl.plane, c, by, 0) != evenX ||
                        rdPair(fa, pl.plane, c, by, 1) != evenY ||
                        rdPair(fb, pl.plane, c, by, 0) != oddX ||
                        rdPair(fb, pl.plane, c, by, 1) != oddY)
                        good = false;
                    any = true;
                }
        }
        if (good) matched = true;
    }
    if (!any) { printf("[FAIL] KFMSuper: no written cells checked\n"); ok = false; }
    else if (!matched) { printf("[FAIL] KFMSuper: written flag cells mismatch\n"); ok = false; }
    return ok;
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    AddVectoredExceptionHandler(1, eh);
    const char* pluginDll = argc > 1 ? argv[1] : nullptr;
    std::string pluginsDir = argc > 2 ? argv[2]
        : "Q:\\Downloads\\openclfilter\\sdk\\avs\\plugins64";

    if (pluginDll) {
        std::string dst = pluginsDir + "\\kfm_analyze_avs.dll";
        if (!CopyFileA(pluginDll, dst.c_str(), FALSE)) {
            printf("cannot copy plugin '%s' -> '%s' (%lu)\n", pluginDll, dst.c_str(),
                   GetLastError());
            return 1;
        }
        printf("plugin installed: %s\n", dst.c_str());
    }

    HMODULE dll = LoadLibraryA("AviSynth.dll");
    if (!dll) { printf("cannot load AviSynth.dll (%lu)\n", GetLastError()); return 1; }
    FARPROC pv = GetProcAddress(dll, "?AVS_linkage@@3QEBUAVS_Linkage@@EB");
    if (!pv) { printf("AVS_linkage export not found\n"); return 1; }
    AVS_linkage = *(const AVS_Linkage**)pv;
    CSE_t cse = (CSE_t)GetProcAddress(dll, "CreateScriptEnvironment");
    if (!cse) { printf("CreateScriptEnvironment not exported\n"); return 1; }
    IScriptEnvironment* env = cse(AVISYNTH_INTERFACE_VERSION);
    if (!env) { printf("env creation failed\n"); return 1; }

    const char* names[] = { "KFMSuperOCL", "KCleanSuperOCL", "KPreCycleAnalyzeOCL",
                            "KPreCycleAnalyzeShowOCL", "KContainsCombeOCL",
                            "KCombeMaskOCL", "KRemoveCombeOCL", "KCFieldDiffOCL",
                            "KCFrameDiffDupOCL", "KAnalyzeNoiseOCL" };
    for (const char* nm : names)
        if (!env->FunctionExists(nm)) { printf("[FAIL] missing %s\n", nm); return 1; }
    printf("autoload OK\n");

    try {

    // ---- KCleanSuper ----
    {
        struct F { const char* n; int pt; };
        F fmts[] = { { "YV12", VideoInfo::CS_YV12 }, { "YV16", VideoInfo::CS_YV16 },
                     { "YV24", VideoInfo::CS_YV24 } };
        for (auto& f : fmts) {
            char tag[128];
            sprintf(tag, "KCleanSuper %s defaults", f.n);
            report(testCleanSuper(env, f.pt, K_NOISY, 10, 8, false), tag);
            sprintf(tag, "KCleanSuper %s thY0 thC255", f.n);
            report(testCleanSuper(env, f.pt, K_COMBED, 0, 255, false), tag);
            sprintf(tag, "KCleanSuper %s thY-1 thC256 alias", f.n);
            report(testCleanSuper(env, f.pt, K_SMOOTH, -1, 256, true), tag);
        }
    }

    // ---- KPreCycleAnalyze ----
    for (int par = 0; par <= 1; ++par) {
        char tag[128];
        sprintf(tag, "KPreCycleAnalyze defaults parity=%d", par);
        report(testPreCycle(env, K_NOISY, par, 20, 12, 24, 16, false), tag);
        sprintf(tag, "KPreCycleAnalyze sweep parity=%d alias", par);
        report(testPreCycle(env, K_COMBED, par, 0, 0, 0, 0, true), tag);
    }

    // ---- KPreCycleAnalyzeShow ----
    report(testPreCycleShow(env, false), "KPreCycleAnalyzeShow");
    report(testPreCycleShow(env, true), "KPreCycleAnalyzeShow alias");

    // ---- KContainsCombe ----
    report(testContainsCombe(env, K_NOISY, 0, false), "KContainsCombe noisy");
    report(testContainsCombe(env, K_NOISY, 1, false), "KContainsCombe all-zero");
    report(testContainsCombe(env, K_NOISY, 2, false), "KContainsCombe border-only");
    report(testContainsCombe(env, K_COMBED, 0, true), "KContainsCombe alias");

    // ---- KCombeMask ----
    {
        struct F { const char* n; int pt; };
        F fmts[] = { { "YV12", VideoInfo::CS_YV12 }, { "YV16", VideoInfo::CS_YV16 },
                     { "YV24", VideoInfo::CS_YV24 } };
        for (auto& f : fmts) {
            char tag[128];
            sprintf(tag, "KCombeMask %s noisy", f.n);
            report(testCombeMask(env, f.pt, K_NOISY, false), tag);
            sprintf(tag, "KCombeMask %s combed", f.n);
            report(testCombeMask(env, f.pt, K_COMBED, false), tag);
        }
    }

    // ---- KRemoveCombe ----
    {
        struct F { const char* n; int pt; };
        F fmts[] = { { "YV12", VideoInfo::CS_YV12 }, { "YV16", VideoInfo::CS_YV16 },
                     { "YV24", VideoInfo::CS_YV24 },
                     { "YUV420P16", VideoInfo::CS_YUV420P16 },
                     { "YUV444P16", VideoInfo::CS_YUV444P16 } };
        for (auto& f : fmts) {
            char tag[128];
            sprintf(tag, "KRemoveCombe %s defaults", f.n);
            report(testRemoveCombe(env, f.pt, K_NOISY, 6, 6, false), tag);
            sprintf(tag, "KRemoveCombe %s th0 th255", f.n);
            report(testRemoveCombe(env, f.pt, K_COMBED, 0, 255, false), tag);
        }
    }

    // ---- KCFieldDiff ----
    {
        struct F { const char* n; int pt; };
        F fmts[] = { { "YV12", VideoInfo::CS_YV12 }, { "YV16", VideoInfo::CS_YV16 },
                     { "YV24", VideoInfo::CS_YV24 },
                     { "YUV420P16", VideoInfo::CS_YUV420P16 } };
        for (auto& f : fmts) {
            char tag[128];
            sprintf(tag, "KCFieldDiff %s defaults", f.n);
            report(testFieldDiff(env, f.pt, K_NOISY, 3, true, false), tag);
            sprintf(tag, "KCFieldDiff %s nt0 chroma=false alias", f.n);
            report(testFieldDiff(env, f.pt, K_COMBED, 0, false, true), tag);
        }
    }

    // ---- KCFrameDiffDup ----
    {
        struct F { const char* n; int pt; };
        F fmts[] = { { "YV12", VideoInfo::CS_YV12 }, { "YV16", VideoInfo::CS_YV16 },
                     { "YV24", VideoInfo::CS_YV24 },
                     { "YUV420P16", VideoInfo::CS_YUV420P16 } };
        for (auto& f : fmts) {
            char tag[128];
            sprintf(tag, "KCFrameDiffDup %s bs32 defaults", f.n);
            report(testFrameDiffDup(env, f.pt, K_NOISY, true, 32, false), tag);
            sprintf(tag, "KCFrameDiffDup %s bs16 chroma=false alias", f.n);
            report(testFrameDiffDup(env, f.pt, K_COMBED, false, 16, true), tag);
            sprintf(tag, "KCFrameDiffDup %s bs8", f.n);
            report(testFrameDiffDup(env, f.pt, K_SMOOTH, true, 8, false), tag);
        }
    }

    // ---- KAnalyzeNoise ----
    report(testAnalyzeNoise(env, K_NOISY, false, false), "KAnalyzeNoise staged noisy");
    report(testAnalyzeNoise(env, K_COMBED, true, false), "KAnalyzeNoise superclip combed");
    report(testAnalyzeNoise(env, K_SMOOTH, false, true), "KAnalyzeNoise alias");

    // ---- KFMSuper ----
    {
        struct F { const char* n; int pt; };
        F fmts[] = { { "YV12", VideoInfo::CS_YV12 }, { "YV24", VideoInfo::CS_YV24 },
                     { "YUV420P16", VideoInfo::CS_YUV420P16 } };
        for (auto& f : fmts)
            for (int par = 0; par <= 1; ++par) {
                char tag[128];
                sprintf(tag, "KFMSuper %s parity=%d", f.n, par);
                report(testFMSuper(env, f.pt, K_COMBED, par != 0, par == 1), tag);
            }
    }

    // ---- validation: conditional function without current_frame ----
    {
        IScriptEnvironment* env2 = cse(AVISYNTH_INTERFACE_VERSION);
        if (env2) {
            try {
                AVSValue s = env2->Invoke("Eval",
                    AVSValue("BlankClip(width=16, height=16, pixel_type=\"YV12\", length=2)"));
                AVSValue a[3] = { s, 3.0, true };
                env2->Invoke("KCFieldDiffOCL", AVSValue(a, 3));
                printf("[FAIL] KCFieldDiff accepted outside ConditionalFilter\n");
                ++g_failures;
            } catch (const AvisynthError&) {
                printf("[PASS] KCFieldDiff throws without current_frame\n");
            } catch (const std::exception& e) {
                printf("[PASS] KCFieldDiff throws without current_frame (%s)\n", e.what());
            } catch (...) {
                printf("[PASS] KCFieldDiff throws without current_frame (unknown)\n");
            }
            ++g_cases;
            env2->DeleteScriptEnvironment();
        }
    }

    } catch (const std::exception& e) {
        printf("UNHANDLED std::exception: %s\n", e.what());
        return 2;
    } catch (const AvisynthError& e) {
        printf("UNHANDLED AvisynthError: %s\n", e.msg);
        return 2;
    }

    printf("\n=== KFM analyze: %d cases (%d checks), %d failures ===\n",
           g_cases, g_checks, g_failures);
    env->DeleteScriptEnvironment();
    return g_failures ? 1 : 0;
}
