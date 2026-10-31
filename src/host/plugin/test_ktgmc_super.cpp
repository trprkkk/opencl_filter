// test_ktgmc_super.cpp — headless verification of the KMSuperOCL AviSynth+
// plugin (eighth host plugin / first motion-stage plugin of the OpenCL port)
// against an in-exe CPU reference.
//
// REFERENCE CHOICE (stated, per the task): the reference is computed HERE, in
// this exe, transcribing the upstream CUDA KMSuper pipeline (MV.cpp
// geometry + the kernels the plugin dispatches), which is also what the repo
// mirrors pin for the shared pieces:
//   * KMFrame::FillPad   -> kl_copy_pad (MVKernel.cu, reflection mirror) %%
//   * KMPlane::Refine    -> kl_horizontal/vertical_wiener (== sim/ktgmc_cpu_ref.
//                            cpp kernel_wiener_h/v, python/run_validation.py)
//   * KMFrame::ReduceToPad -> kl_RB2B_bilinear_filtered_with_pad (== the fused
//                            algorithm of sim/ktgmc_rb2b_pad_ref.cpp,
//                            python/run_mv_rb2b_pad.py)
// The copy_pad reflection form is the CUDA path (it differs from the upstream
// CPU Copy+PadFrame *replication* border); the plugin is the CUDA-path port, so
// the reference transcribes the CUDA kernel.  (There is no in-repo CPU mirror
// for kt_copy_pad; the wiener + fused-RB2B halves ARE mirror-pinned and are
// transcribed from those.)
//
// Coverage: 8/16-bit x {4:2:0, 4:2:2, 4:4:4} x pel {1,2} x chroma {on,off}
// x levels {auto, explicit}, plus entry-point alias, defaults-only invocation,
// a real script string, and the upstream constructor validation throws.
// Every plane (Y, and U/V when chroma) is compared against the reference.
//
// Build: see build_host_super.bat.  Usage: test_ktgmc_super.exe
// <plugin_dll_path> [plugins_dir]
#define NOMINMAX
#include <windows.h>

#include "avisynth.h"
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

// ===========================================================================
// MV.cpp geometry helpers (verbatim)
// ===========================================================================
static int PlaneHeightLuma(int src_height, int level, int yRatioUV, int vpad) {
    int height = src_height;
    for (int i = 1; i <= level; i++)
        height = vpad >= yRatioUV ? ((height / yRatioUV + 1) / 2) * yRatioUV
                                  : ((height / yRatioUV) / 2) * yRatioUV;
    return height;
}
static int PlaneWidthLuma(int src_width, int level, int xRatioUV, int hpad) {
    int width = src_width;
    for (int i = 1; i <= level; i++)
        width = hpad >= xRatioUV ? ((width / xRatioUV + 1) / 2) * xRatioUV
                                 : ((width / xRatioUV) / 2) * xRatioUV;
    return width;
}
static unsigned int PlaneSuperOffset(bool chroma, int src_height, int level, int pel,
                                     int vpad, int plane_pitch, int yRatioUV) {
    int height = src_height;
    unsigned int offset;
    if (level == 0) {
        offset = 0;
    } else {
        offset = (unsigned)pel * pel * plane_pitch * (src_height + vpad * 2);
        for (int i = 1; i < level; i++) {
            height = chroma
                ? PlaneHeightLuma(src_height * yRatioUV, i, yRatioUV, vpad * yRatioUV) / yRatioUV
                : PlaneHeightLuma(src_height, i, yRatioUV, vpad);
            offset += (unsigned)plane_pitch * (height + vpad * 2);
        }
    }
    return offset;
}

// ===========================================================================
// in-exe CPU reference (CUDA KMSuper pipeline; see header)
// ===========================================================================
struct SuperGeom {
    // config
    int origW, origH, nHPad, nVPad, nPel, nLevels, xRatioUV, yRatioUV, logUVx, logUVy;
    int nSuperWidth, nSuperHeight;
    bool chroma;
};

static SuperGeom makeGeom(const VideoInfo& vi, int nHPad, int nVPad, int nPel,
                          int nLevels, bool chroma) {
    SuperGeom g;
    g.origW = vi.width; g.origH = vi.height;
    g.nHPad = nHPad; g.nVPad = nVPad; g.nPel = nPel; g.chroma = chroma;
    g.xRatioUV = 1 << vi.GetPlaneWidthSubsampling(PLANAR_U);
    g.yRatioUV = 1 << vi.GetPlaneHeightSubsampling(PLANAR_U);
    g.logUVx = vi.GetPlaneWidthSubsampling(PLANAR_U);
    g.logUVy = vi.GetPlaneHeightSubsampling(PLANAR_U);
    int nLevelsMax = 0;
    while (PlaneHeightLuma(g.origH, nLevelsMax, g.yRatioUV, nVPad) >= g.yRatioUV * 2 &&
           PlaneWidthLuma(g.origW, nLevelsMax, g.xRatioUV, nHPad) >= g.xRatioUV * 2)
        nLevelsMax++;
    if (nLevels <= 0 || nLevels > nLevelsMax) nLevels = nLevelsMax;
    if (nLevels < 1) nLevels = 1;
    g.nLevels = nLevels;
    g.nSuperWidth = g.origW + 2 * nHPad;
    g.nSuperHeight = (int)(PlaneSuperOffset(false, g.origH, nLevels, nPel, nVPad,
                                            g.nSuperWidth, g.yRatioUV) / (unsigned)g.nSuperWidth);
    if (g.yRatioUV == 2 && (g.nSuperHeight & 1)) g.nSuperHeight++;
    return g;
}

static int mirrorIdx(int c, int n) {
    if (c < 0) return -c - 1;
    if (c >= n) return n - (c - n) - 1;
    return c;
}

// build one plane's reference (pitch P = tight, planeH rows).  Returns the
// vector plus asserts the total footprint matches planeH.
template <typename PX>
static std::vector<PX> buildPlaneRef(const SuperGeom& g, bool isChroma,
                                     const std::vector<PX>& src, int srcW, int srcH,
                                     int& outPitch, int& outH) {
    const int P = isChroma ? (g.nSuperWidth >> g.logUVx) : g.nSuperWidth;
    const int planeH = isChroma ? (g.nSuperHeight >> g.logUVy) : g.nSuperHeight;
    const int hpad = isChroma ? g.nHPad / g.xRatioUV : g.nHPad;
    const int vpad = isChroma ? g.nVPad / g.yRatioUV : g.nVPad;
    const int H0 = srcH, W0 = srcW;  // level-0 plane dims (already subsampled for chroma)
    const int L = g.nLevels;
    const PX MAXV = sizeof(PX) == 1 ? (PX)255 : (PX)65535;

    std::vector<int> off(L);
    for (int l = 0; l < L; l++)
        off[l] = (int)PlaneSuperOffset(isChroma, H0, l, g.nPel, vpad, P, g.yRatioUV);
    std::vector<std::pair<int,int>> dim(L);
    dim[0] = std::make_pair(W0, H0);
    for (int l = 1; l < L; l++) {
        int lw = PlaneWidthLuma(g.origW, l, g.xRatioUV, g.nHPad);
        int lh = PlaneHeightLuma(g.origH, l, g.yRatioUV, g.nVPad);
        dim[l] = isChroma ? std::make_pair(lw / g.xRatioUV, lh / g.yRatioUV)
                          : std::make_pair(lw, lh);
    }
    // footprint check
    unsigned total = (unsigned)off[L - 1] + (unsigned)P * (dim[L - 1].second + 2 * vpad);
    if ((int)total != P * planeH) {
        printf("  [ref] footprint mismatch: off_last=%d lastH=%d total=%u P*planeH=%d\n",
               off[L - 1], dim[L - 1].second, total, P * planeH);
    }

    std::vector<PX> plane((size_t)P * planeH, (PX)0);
    auto at = [&](int base, int x, int y) -> PX& { return plane[(size_t)base + x + (size_t)y * P]; };

    // ---- 1. copy_pad (reflection) into level-0 sub-plane 0 ----
    {
        int int0 = off[0] + P * vpad + hpad;
        int extW = W0 + 2 * hpad, extH = H0 + 2 * vpad;
        for (int gy = 0; gy < extH; gy++) {
            int y = gy - vpad;
            for (int gx = 0; gx < extW; gx++) {
                int x = gx - hpad;
                if (x < W0 + hpad && y < H0 + vpad) {
                    int sx = mirrorIdx(x, W0), sy = mirrorIdx(y, H0);
                    at(int0, x, y) = src[(size_t)sx + (size_t)sy * srcW];
                }
            }
        }
    }

    // ---- 2. Refine (nPel==2): wiener ----
    if (g.nPel == 2) {
        int extW = W0 + 2 * hpad, extH = H0 + 2 * vpad;
        int ext = P * extH;
        auto wienerV = [&](int dstOff, int srcOff) {
            for (int y = 0; y < extH; y++) for (int x = 0; x < extW; x++) {
                auto S = [&](int dy) { return (int)at(srcOff, x, y + dy); };
                int v;
                if (y < 2) v = (S(0) + S(1) + 1) >> 1;
                else if (y < extH - 4) {
                    int num = S(-2) + ((-S(-1) + 4 * S(0) + 4 * S(1) - S(2)) * 5) + S(3) + 16;
                    v = (int)std::min<int>(MAXV, std::max(0, num >> 5));
                } else if (y < extH - 1) v = (S(0) + S(1) + 1) >> 1;
                else v = S(0);
                at(dstOff, x, y) = (PX)v;
            }
        };
        auto wienerH = [&](int dstOff, int srcOff) {
            for (int y = 0; y < extH; y++) for (int x = 0; x < extW; x++) {
                auto S = [&](int dx) { return (int)at(srcOff, x + dx, y); };
                int v;
                if (x < 2) v = (S(0) + S(1) + 1) >> 1;
                else if (x < extW - 4) {
                    int num = S(-2) + ((-S(-1) + 4 * S(0) + 4 * S(1) - S(2)) * 5) + S(3) + 16;
                    v = (int)std::min<int>(MAXV, std::max(0, num >> 5));
                } else if (x < extW - 1) v = (S(0) + S(1) + 1) >> 1;
                else v = S(0);
                at(dstOff, x, y) = (PX)v;
            }
        };
        wienerH(off[0] + 1 * ext, off[0]);
        wienerV(off[0] + 2 * ext, off[0]);
        wienerH(off[0] + 3 * ext, off[0] + 2 * ext);
    }

    // ---- 3. coarse levels: fused RB2B reduce + pad ----
    for (int l = 0; l + 1 < L; l++) {
        int Wd = dim[l + 1].first, Hd = dim[l + 1].second;
        int srcInt = off[l] + P * vpad + hpad;
        int dstInt = off[l + 1] + P * vpad + hpad;
        int extW = Wd + 2 * hpad, extH = Hd + 2 * vpad;
        for (int gy = 0; gy < extH; gy++) {
            int dsty = gy - vpad;
            for (int gx = 0; gx < extW; gx++) {
                int dstx = gx - hpad;
                if (!(dstx < Wd + hpad && dsty < Hd + vpad)) continue;
                int xmul0 = 0, xmul1 = 4, srcx;
                if (dstx <= 0) srcx = 0;
                else if (dstx >= Wd - 1) srcx = (Wd - 1) * 2;
                else { srcx = dstx * 2; xmul0 = 1; xmul1 = 3; }
                int ymul0 = 0, ymul1 = 4, srcy;
                if (dsty <= 0) srcy = 0;
                else if (dsty >= Hd - 1) srcy = (Hd - 1) * 2;
                else { srcy = dsty * 2; ymul0 = 1; ymul1 = 3; }
                int sum = 0;
                for (int j = -1; j <= 2; j++) {
                    int ymul = (j == 0 || j == 1) ? ymul1 : ymul0;
                    if (ymul <= 0) continue;
                    for (int i = -1; i <= 2; i++) {
                        int xmul = (i == 0 || i == 1) ? xmul1 : xmul0;
                        if (xmul <= 0) continue;
                        sum += (int)at(srcInt, srcx + i, srcy + j) * ymul * xmul;
                    }
                }
                sum = (sum + 32) / 64;
                at(dstInt, dstx, dsty) = (PX)sum;
            }
        }
    }

    outPitch = P; outH = planeH;
    return plane;
}

// ===========================================================================
// deterministic source clips
// ===========================================================================
static uint32_t mix32(uint32_t a) {
    a ^= a >> 16; a *= 0x7feb352du;
    a ^= a >> 15; a *= 0x846ca68bu;
    a ^= a >> 16;
    return a;
}

template <typename PX>
static void fillPlane(uint8_t* dstp, int pitch, int w, int h, uint32_t salt, int maxv) {
    for (int y = 0; y < h; ++y) {
        PX* row = (PX*)(dstp + (size_t)y * pitch);
        for (int x = 0; x < w; ++x) {
            int v = (int)(((int64_t)x * (maxv - 40)) / (w > 1 ? w - 1 : 1)
                          + ((int64_t)y * (maxv - 40)) / (h > 1 ? h - 1 : 1)
                          + 20);
            v += (int)(mix32((uint32_t)x * 17u + (uint32_t)y * 1013u + salt) % 9u) - 4;
            if (v < 0) v = 0;
            if (v > maxv) v = maxv;
            row[x] = (PX)v;
        }
    }
}

class RandSource : public IClip {
    VideoInfo vi_;
    uint32_t salt_;
    int px_;
public:
    RandSource(VideoInfo vi, uint32_t salt) : vi_(vi), salt_(salt), px_(vi.ComponentSize()) {}
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) {
        PVideoFrame f = env->NewVideoFrame(vi_);
        const int w = vi_.width, h = vi_.height;
        const int wUV = w >> vi_.GetPlaneWidthSubsampling(PLANAR_U);
        const int hUV = h >> vi_.GetPlaneHeightSubsampling(PLANAR_U);
        const int maxv = (px_ == 1) ? 255 : 65535;
        uint32_t s = salt_ + (uint32_t)n * 7919u;
        if (px_ == 1) {
            fillPlane<uint8_t>(f->GetWritePtr(PLANAR_Y), f->GetPitch(PLANAR_Y), w, h, s, maxv);
            fillPlane<uint8_t>(f->GetWritePtr(PLANAR_U), f->GetPitch(PLANAR_U), wUV, hUV, s + 11u, maxv);
            fillPlane<uint8_t>(f->GetWritePtr(PLANAR_V), f->GetPitch(PLANAR_V), wUV, hUV, s + 23u, maxv);
        } else {
            fillPlane<uint16_t>(f->GetWritePtr(PLANAR_Y), f->GetPitch(PLANAR_Y), w, h, s, maxv);
            fillPlane<uint16_t>(f->GetWritePtr(PLANAR_U), f->GetPitch(PLANAR_U), wUV, hUV, s + 11u, maxv);
            fillPlane<uint16_t>(f->GetWritePtr(PLANAR_V), f->GetPitch(PLANAR_V), wUV, hUV, s + 23u, maxv);
        }
        return f;
    }
    bool __stdcall GetParity(int) { return false; }
    void __stdcall GetAudio(void*, int64_t, int64_t, IScriptEnvironment*) {}
    int __stdcall SetCacheHints(int, int) { return 0; }
    const VideoInfo& __stdcall GetVideoInfo() { return vi_; }
};

// ===========================================================================
// comparison
// ===========================================================================
template <typename PX>
static void readPlane(std::vector<PX>& out, PVideoFrame f, int pl, int pw, int ph) {
    out.resize((size_t)pw * ph);
    const uint8_t* fp = f->GetReadPtr(pl);
    int pitch = f->GetPitch(pl);
    for (int y = 0; y < ph; ++y)
        memcpy(&out[(size_t)y * pw], fp + (size_t)y * pitch, (size_t)pw * sizeof(PX));
}

template <typename PX>
static bool comparePlane(const char* tag, PVideoFrame outf, int pl, const std::vector<PX>& ref,
                         int P, int planeH) {
    const uint8_t* fp = outf->GetReadPtr(pl);
    int pitch = outf->GetPitch(pl);
    for (int y = 0; y < planeH; ++y) {
        const PX* got = (const PX*)(fp + (size_t)y * pitch);
        const PX* exp = &ref[(size_t)y * P];
        for (int x = 0; x < P; ++x) {
            if (got[x] != exp[x]) {
                printf("  MISMATCH %s plane %d px(%d,%d) got=%d exp=%d\n",
                       tag, pl, x, y, (int)got[x], (int)exp[x]);
                return false;
            }
        }
    }
    return true;
}

// run one case + compare
template <typename PX>
static void runCase(IScriptEnvironment* env, const char* tag, PClip srcClip,
                    const SuperGeom& g, int n, const char* fnName) {
    AVSValue args[8] = { AVSValue(srcClip), g.nHPad, g.nVPad, g.nPel, g.nLevels,
                         AVSValue(g.chroma), 2, 2 };
    AVSValue ret;
    try {
        ret = env->Invoke(fnName, AVSValue(args, 8));
    } catch (const AvisynthError& e) {
        printf("[FAIL] %s: Invoke threw: %s\n", tag, e.msg);
        ++g_failures; ++g_cases; return;
    }
    if (!ret.IsClip()) { printf("[FAIL] %s: not a clip\n", tag); ++g_failures; ++g_cases; return; }
    PClip filtered = ret.AsClip();

    const VideoInfo& ovi = filtered->GetVideoInfo();
    if (ovi.width != g.nSuperWidth || ovi.height != g.nSuperHeight) {
        printf("[FAIL] %s: output dims %dx%d != super %dx%d\n", tag, ovi.width, ovi.height,
               g.nSuperWidth, g.nSuperHeight);
        ++g_failures; ++g_cases; return;
    }

    // source pixels for the reference
    PVideoFrame sf = srcClip->GetFrame(n, env);
    const int w = g.origW, h = g.origH;
    const int wUV = w / g.xRatioUV, hUV = h / g.yRatioUV;
    std::vector<PX> sY, sU, sV;
    readPlane<PX>(sY, sf, PLANAR_Y, w, h);
    readPlane<PX>(sU, sf, PLANAR_U, wUV, hUV);
    readPlane<PX>(sV, sf, PLANAR_V, wUV, hUV);

    int pPitch, pH;
    std::vector<PX> rY = buildPlaneRef<PX>(g, false, sY, w, h, pPitch, pH);
    std::vector<PX> rU, rV;
    if (g.chroma) {
        rU = buildPlaneRef<PX>(g, true, sU, wUV, hUV, pPitch, pH);
        rV = buildPlaneRef<PX>(g, true, sV, wUV, hUV, pPitch, pH);
    }

    PVideoFrame outf;
    try {
        outf = filtered->GetFrame(n, env);
    } catch (const AvisynthError& e) {
        printf("[FAIL] %s: GetFrame threw: %s\n", tag, e.msg);
        ++g_failures; ++g_cases; return;
    }
    ++g_frame_checks;
    bool ok = true;
    {
        int P = g.nSuperWidth, planeH = g.nSuperHeight;
        ok = comparePlane<PX>(tag, outf, PLANAR_Y, rY, P, planeH);
    }
    if (ok && g.chroma) {
        int P = g.nSuperWidth >> g.logUVx, planeH = g.nSuperHeight >> g.logUVy;
        ok = comparePlane<PX>(tag, outf, PLANAR_U, rU, P, planeH);
        if (ok) ok = comparePlane<PX>(tag, outf, PLANAR_V, rV, P, planeH);
    }
    ++g_cases;
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", tag);
    if (!ok) ++g_failures;
}

// ===========================================================================
// driver
// ===========================================================================
struct FormatSpec { const char* pixel_type; };

static const FormatSpec g_formats[] = {
    { "YV12" }, { "YV16" }, { "YV24" },
    { "YUV420P16" }, { "YUV422P16" }, { "YUV444P16" },
};

static const int W = 64, H = 48;

static VideoInfo viFor(IScriptEnvironment* env, const char* pt) {
    char script[128];
    sprintf_s(script, "BlankClip(width=%d, height=%d, pixel_type=\"%s\", length=4)", W, H, pt);
    return env->Invoke("Eval", AVSValue(script)).AsClip()->GetVideoInfo();
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    AddVectoredExceptionHandler(1, eh);
    const char* pluginDll = argc > 1 ? argv[1] : nullptr;
    std::string pluginsDir = argc > 2 ? argv[2]
        : "Q:\\Downloads\\openclfilter\\sdk\\avs\\plugins64";

    if (pluginDll) {
        std::string dst = pluginsDir + "\\ktgmc_super_avs.dll";
        if (!CopyFileA(pluginDll, dst.c_str(), FALSE)) {
            printf("cannot copy plugin '%s' -> '%s' (%lu)\n", pluginDll, dst.c_str(), GetLastError());
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

    if (!env->FunctionExists("KMSuperOCL")) {
        printf("[FAIL] autoload: KMSuperOCL not registered\n"); return 1;
    }
    if (!env->FunctionExists("KMSuper")) {
        printf("[FAIL] autoload: KMSuper alias not registered\n"); return 1;
    }
    printf("autoload OK (KMSuperOCL + KMSuper alias registered)\n");

    // ---- main sweep ----
    const int pels[] = { 1, 2 };
    const bool chromas[] = { true, false };
    const int levelsArr[] = { 0 };  // auto
    for (const FormatSpec& fs : g_formats) {
        VideoInfo vi = viFor(env, fs.pixel_type);
        const bool is16 = vi.ComponentSize() == 2;
        for (int ip = 0; ip < 2; ++ip) {
            for (int ic = 0; ic < 2; ++ic) {
                for (int il = 0; il < 1; ++il) {
                    SuperGeom g = makeGeom(vi, 8, 8, pels[ip], levelsArr[il], chromas[ic]);
                    PClip srcClip(new RandSource(vi, 0x5EEDu ^ ((uint32_t)ip << 8) ^ ((uint32_t)ic << 16)));
                    char tag[160];
                    sprintf_s(tag, "%s pel=%d chroma=%d levels=auto (super %dx%d nLevels=%d)",
                              fs.pixel_type, pels[ip], chromas[ic], g.nSuperWidth, g.nSuperHeight, g.nLevels);
                    if (is16) runCase<uint16_t>(env, tag, srcClip, g, 1, "KMSuperOCL");
                    else      runCase<uint8_t>(env, tag, srcClip, g, 1, "KMSuperOCL");
                }
            }
        }
    }

    // ---- explicit levels (2) on YV12 ----
    {
        VideoInfo vi = viFor(env, "YV12");
        SuperGeom g = makeGeom(vi, 8, 8, 2, 2, true);
        PClip srcClip(new RandSource(vi, 0xABCDu));
        char tag[160];
        sprintf_s(tag, "YV12 pel=2 chroma=1 levels=2 (super %dx%d nLevels=%d)",
                  g.nSuperWidth, g.nSuperHeight, g.nLevels);
        runCase<uint8_t>(env, tag, srcClip, g, 0, "KMSuperOCL");
    }

    // ---- defaults-only invocation (pel 2, chroma on, levels auto) ----
    {
        VideoInfo vi = viFor(env, "YV12");
        SuperGeom g = makeGeom(vi, 8, 8, 2, 0, true);
        PClip srcClip(new RandSource(vi, 0xD00Du));
        AVSValue args[1] = { AVSValue(srcClip) };
        try {
            AVSValue ret = env->Invoke("KMSuperOCL", AVSValue(args, 1));
            if (!ret.IsClip()) throw AvisynthError("no clip");
            PClip filtered = ret.AsClip();
            // reuse the runCase comparison machinery by hand
            PVideoFrame sf = srcClip->GetFrame(2, env);
            std::vector<uint8_t> sY, sU, sV;
            readPlane<uint8_t>(sY, sf, PLANAR_Y, W, H);
            readPlane<uint8_t>(sU, sf, PLANAR_U, W / 2, H / 2);
            readPlane<uint8_t>(sV, sf, PLANAR_V, W / 2, H / 2);
            int pp, ph;
            std::vector<uint8_t> rY = buildPlaneRef<uint8_t>(g, false, sY, W, H, pp, ph);
            std::vector<uint8_t> rU = buildPlaneRef<uint8_t>(g, true, sU, W / 2, H / 2, pp, ph);
            std::vector<uint8_t> rV = buildPlaneRef<uint8_t>(g, true, sV, W / 2, H / 2, pp, ph);
            PVideoFrame outf = filtered->GetFrame(2, env);
            ++g_frame_checks;
            bool ok = comparePlane<uint8_t>("defaults", outf, PLANAR_Y, rY, g.nSuperWidth, g.nSuperHeight);
            if (ok) ok = comparePlane<uint8_t>("defaults", outf, PLANAR_U, rU, g.nSuperWidth >> 1, g.nSuperHeight >> 1);
            if (ok) ok = comparePlane<uint8_t>("defaults", outf, PLANAR_V, rV, g.nSuperWidth >> 1, g.nSuperHeight >> 1);
            printf("[%s] YV12 defaults-only invocation\n", ok ? "PASS" : "FAIL");
            if (!ok) ++g_failures;
        } catch (const AvisynthError& e) {
            printf("[FAIL] defaults-only: %s\n", e.msg); ++g_failures;
        }
        ++g_cases;
    }

    // ---- upstream alias invocation (KMSuper) ----
    {
        VideoInfo vi = viFor(env, "YV12");
        SuperGeom g = makeGeom(vi, 8, 8, 1, 0, true);
        PClip srcClip(new RandSource(vi, 0xA1A5u));
        char tag[160];
        sprintf_s(tag, "YV12 KMSuper alias pel=1 chroma=1 (super %dx%d)", g.nSuperWidth, g.nSuperHeight);
        runCase<uint8_t>(env, tag, srcClip, g, 1, "KMSuper");
    }

    // ---- real script string over ColorBars ----
    try {
        const char* script =
            "KMSuperOCL(ColorBars(pixel_type=\"YV12\", width=64, height=48).Trim(0,3), "
            "pel=2, chroma=true)";
        AVSValue ret = env->Invoke("Eval", AVSValue(script));
        if (!ret.IsClip()) throw AvisynthError("Eval did not return a clip");
        AVSValue srcV = env->Invoke("Eval",
            AVSValue("ColorBars(pixel_type=\"YV12\", width=64, height=48).Trim(0,3)"));
        PClip srcClip = srcV.AsClip();
        VideoInfo vi = srcClip->GetVideoInfo();
        SuperGeom g = makeGeom(vi, 8, 8, 2, 0, true);
        PClip filtered = ret.AsClip();
        bool ok = true;
        for (int n = 0; n < 4 && ok; ++n) {
            PVideoFrame sf = srcClip->GetFrame(n, env);
            std::vector<uint8_t> sY, sU, sV;
            readPlane<uint8_t>(sY, sf, PLANAR_Y, W, H);
            readPlane<uint8_t>(sU, sf, PLANAR_U, W / 2, H / 2);
            readPlane<uint8_t>(sV, sf, PLANAR_V, W / 2, H / 2);
            int pp, ph;
            std::vector<uint8_t> rY = buildPlaneRef<uint8_t>(g, false, sY, W, H, pp, ph);
            std::vector<uint8_t> rU = buildPlaneRef<uint8_t>(g, true, sU, W / 2, H / 2, pp, ph);
            std::vector<uint8_t> rV = buildPlaneRef<uint8_t>(g, true, sV, W / 2, H / 2, pp, ph);
            PVideoFrame outf = filtered->GetFrame(n, env);
            ++g_frame_checks;
            ok = comparePlane<uint8_t>("script", outf, PLANAR_Y, rY, g.nSuperWidth, g.nSuperHeight);
            if (ok) ok = comparePlane<uint8_t>("script", outf, PLANAR_U, rU, g.nSuperWidth >> 1, g.nSuperHeight >> 1);
            if (ok) ok = comparePlane<uint8_t>("script", outf, PLANAR_V, rV, g.nSuperWidth >> 1, g.nSuperHeight >> 1);
        }
        ++g_cases;
        printf("[%s] script-string ColorBars YV12 (4 frames)\n", ok ? "PASS" : "FAIL");
        if (!ok) ++g_failures;
    } catch (const AvisynthError& e) {
        printf("[FAIL] script-string case: %s\n", e.msg); ++g_failures;
    }

    // ---- validation throws (upstream constructor) ----
    auto expectThrow = [&](const char* what, int hpad, int vpad, int pel, int levels,
                           bool chroma, int sharp, int filter) {
        ++g_cases;
        try {
            VideoInfo vi = viFor(env, "YV12");
            PClip srcClip(new RandSource(vi, 0x1234u));
            AVSValue args[8] = { AVSValue(srcClip), hpad, vpad, pel, levels, AVSValue(chroma), sharp, filter };
            try {
                env->Invoke("KMSuperOCL", AVSValue(args, 8));
                printf("[FAIL] %s accepted (expected throw)\n", what);
                ++g_failures;
            } catch (const AvisynthError&) {
                printf("[PASS] %s rejected\n", what);
            }
        } catch (const AvisynthError& e) {
            printf("[FAIL] %s setup: %s\n", what, e.msg); ++g_failures;
        }
    };
    expectThrow("hpad!=8", 16, 8, 2, 0, true, 2, 2);
    expectThrow("vpad!=8", 8, 16, 2, 0, true, 2, 2);
    expectThrow("pel=4", 8, 8, 4, 0, true, 2, 2);
    expectThrow("sharp!=2", 8, 8, 2, 0, true, 4, 2);
    expectThrow("filter!=2", 8, 8, 2, 0, true, 2, 1);

    printf("\n=== KMSuperOCL: %d cases (%d frame checks), %d failures ===\n",
           g_cases, g_frame_checks, g_failures);

    env->DeleteScriptEnvironment();
    // deliberately not FreeLibrary (AviSynth+ shutdown threads).
    return g_failures ? 1 : 0;
}
