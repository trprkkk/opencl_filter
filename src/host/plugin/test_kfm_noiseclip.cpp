// test_kfm_noiseclip.cpp — headless verification of the KNoiseClipOCL
// AviSynth+ plugin (fifth host plugin of the OpenCL port) against a CPU
// reference computed in this exe.
//
// REFERENCE CHOICE (stated, per the task): the reference is computed HERE,
// in this exe, from the actual input frame pixels, transcribing the repo's
// CPU-mirror/golden semantics (sim/kfm_noiseclip_ref.cpp +
// python/run_kfm_noiseclip.py — the golden-pinned authority, == the upstream
// cpu_noise_clip twin in DecombeUCF.cu): per pixel,
//   s   = (src - noise + 256) >> 1            // 128 == equal
//   out = dev_limitter(s, nmin, range)        // {0,56,128,199,255}
// with Y using nmin_y/range_y and U/V each using nmin_uv/range_uv.  All
// arithmetic is integer.
//
// NOTE: KNoiseClip is 8-BIT ONLY upstream (ComponentSize==1 throws), so this
// verifier covers only 8-bit formats (YV12/YV16/YV24) and checks that 16-bit
// input is rejected.
//
// Coverage:
//   - 3 formats (YV12/YV16/YV24 — the three subsampling geometries at 8 bit),
//   - parameter sweep incl. upstream defaults (nmin_y 1, range_y 128,
//     nmin_uv 1, range_uv 128) and boundaries: nmin 0 / 1 / 128 / 200,
//     range 0 / 1 / 128 / 255, independent luma/chroma params,
//   - source pairs: NOISY-vs-NOISY, CLEAN-vs-CLEAN, EXTREME (all-0 vs
//     all-255 and all-255 vs all-0), MIXED (noisy vs clean) — every pixel of
//     every plane compared against the in-exe reference,
//   - defaults-only + upstream-alias (KNoiseClip) invocations, a real script
//     string over ColorBars, and validation: width%4, height%4, resolution
//     mismatch and 16-bit input must throw.
//
// Load path: the exe copies the plugin DLL into the AviSynth+ autoload
// directory (registry HKLM\Software\AviSynth plugins64), creates the script
// environment, verifies autoload via FunctionExists, then invokes
// KNoiseClipOCL (and the upstream KNoiseClip alias) and compares EVERY pixel
// of every plane.
//
// Build: see build_host_noiseclip.bat.  Usage: test_kfm_noiseclip.exe
// <plugin_dll_path> [plugins_dir]
#include <windows.h>

#include "avisynth.h"

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
// deterministic source clips (scripted pixel content, 8-bit only)
// ---------------------------------------------------------------------------
static uint32_t mix32(uint32_t a) {
    a ^= a >> 16; a *= 0x7feb352du;
    a ^= a >> 15; a *= 0x846ca68bu;
    a ^= a >> 16;
    return a;
}

enum SourceKind {
    SRC_SMOOTH,  // smooth 2D gradient (clean content)
    SRC_NOISY,   // full-range per-pixel noise
    SRC_ZERO,    // every pixel 0 (extreme)
    SRC_MAX,     // every pixel 255 (extreme)
    SRC_MIXED,   // checkerboard: noisy blocks on a smooth base
};

static void fillPattern(uint8_t* dstp, int pitch, int w, int h, uint32_t salt,
                        SourceKind kind) {
    for (int y = 0; y < h; ++y) {
        uint8_t* row = dstp + (size_t)y * pitch;
        for (int x = 0; x < w; ++x) {
            int v;
            switch (kind) {
            case SRC_SMOOTH:
                v = (int)(((int64_t)x * 127) / (w > 1 ? w - 1 : 1)
                          + ((int64_t)y * 127) / (h > 1 ? h - 1 : 1));
                break;
            case SRC_NOISY:
                v = (int)(mix32((uint32_t)x * 17u + (uint32_t)y * 1013u + salt) & 255u);
                break;
            case SRC_ZERO:
                v = 0;
                break;
            case SRC_MAX:
                v = 255;
                break;
            default:  // SRC_MIXED: 16x16 blocks alternating noisy / flat
                if (((x / 16) + (y / 16)) & 1)
                    v = (int)(mix32((uint32_t)x * 31u + (uint32_t)y * 7u + salt) & 255u);
                else
                    v = 128 + ((int)(mix32(salt ^ ((uint32_t)x * 3u + (uint32_t)y * 5u)) % 11u) - 5);
                break;
            }
            row[x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
    }
}

class RandSource : public IClip {
    VideoInfo vi_;
    uint32_t salt_;
    SourceKind kind_;
public:
    RandSource(VideoInfo vi, uint32_t salt, SourceKind kind) : vi_(vi), salt_(salt), kind_(kind) {}
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) {
        PVideoFrame f = env->NewVideoFrame(vi_);
        const int w = vi_.width, h = vi_.height;
        const int wUV = w >> vi_.GetPlaneWidthSubsampling(PLANAR_U);
        const int hUV = h >> vi_.GetPlaneHeightSubsampling(PLANAR_U);
        uint32_t s = salt_ + (uint32_t)n * 7919u;  // content varies per frame
        fillPattern(f->GetWritePtr(PLANAR_Y), f->GetPitch(PLANAR_Y), w, h, s, kind_);
        fillPattern(f->GetWritePtr(PLANAR_U), f->GetPitch(PLANAR_U), wUV, hUV, s + 11u, kind_);
        fillPattern(f->GetWritePtr(PLANAR_V), f->GetPitch(PLANAR_V), wUV, hUV, s + 23u, kind_);
        return f;
    }
    bool __stdcall GetParity(int) { return false; }
    void __stdcall GetAudio(void*, int64_t, int64_t, IScriptEnvironment*) {}
    int __stdcall SetCacheHints(int, int) { return 0; }
    const VideoInfo& __stdcall GetVideoInfo() { return vi_; }
};

// ---------------------------------------------------------------------------
// CPU reference (golden semantics; see file header)
// ---------------------------------------------------------------------------
template <typename PX>
static void readPlane(std::vector<PX>& out, PVideoFrame f, int pl, int pw, int ph) {
    out.resize((size_t)pw * ph);
    const uint8_t* fp = f->GetReadPtr(pl);
    int pitch = f->GetPitch(pl);
    for (int y = 0; y < ph; ++y)
        memcpy(&out[(size_t)y * pw], fp + (size_t)y * pitch, (size_t)pw * sizeof(PX));
}

// upstream dev_limitter (verbatim)
static int dev_limitter(int x, int nmin, int range) {
    return (x == 128)
        ? 128
        : ((x < 128)
            ? ((((127 - range) < x) && (x < (128 - nmin))) ? 0 : 56)
            : ((((128 + nmin) < x) && (x < (129 + range))) ? 255 : 199));
}

// upstream cpu_noise_clip twin (scalar; pitch == width packed planes)
static void ref_noise_clip(std::vector<uint8_t>& dst, const std::vector<uint8_t>& src,
                           const std::vector<uint8_t>& noise,
                           int width, int height, int nmin, int range) {
    const int pitch = width;
    dst.resize((size_t)pitch * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int offset = y * pitch + x;
            int s = ((int)src[offset] - (int)noise[offset] + 256) >> 1;
            dst[offset] = (uint8_t)dev_limitter(s, nmin, range);
        }
    }
}

// The GetFrame into the source clip needs an IScriptEnvironment; pass it
// through a file-scope pointer set by the driver (single-threaded test exe).
static IScriptEnvironment* g_env = nullptr;
static IScriptEnvironment* env_of(const VideoInfo&) { return g_env; }

// ---------------------------------------------------------------------------
// case pipeline reference (upstream GetFrameT semantics: one pass per plane)
// + full-plane comparison
// ---------------------------------------------------------------------------
struct Params {
    int nmin_y, range_y, nmin_uv, range_uv;
};

static bool refAndCompare(const char* tag, const VideoInfo& vi, PClip srcClip,
                          PClip noiseClip, PVideoFrame outf, const Params& p, int n) {
    const int w = vi.width, h = vi.height;
    const int lx = vi.GetPlaneWidthSubsampling(PLANAR_U);
    const int ly = vi.GetPlaneHeightSubsampling(PLANAR_U);
    const int wUV = w >> lx, hUV = h >> ly;

    PVideoFrame sf = srcClip->GetFrame(n, env_of(vi));
    PVideoFrame nf = noiseClip->GetFrame(n, env_of(vi));
    std::vector<uint8_t> sY, sU, sV, nY, nU, nV;
    readPlane(sY, sf, PLANAR_Y, w, h);
    readPlane(sU, sf, PLANAR_U, wUV, hUV);
    readPlane(sV, sf, PLANAR_V, wUV, hUV);
    readPlane(nY, nf, PLANAR_Y, w, h);
    readPlane(nU, nf, PLANAR_U, wUV, hUV);
    readPlane(nV, nf, PLANAR_V, wUV, hUV);

    std::vector<uint8_t> rY, rU, rV;
    ref_noise_clip(rY, sY, nY, w, h, p.nmin_y, p.range_y);
    ref_noise_clip(rU, sU, nU, wUV, hUV, p.nmin_uv, p.range_uv);
    ref_noise_clip(rV, sV, nV, wUV, hUV, p.nmin_uv, p.range_uv);

    // ---- compare every pixel of every plane ----
    bool ok = true;
    auto cmp = [&](const char* plane, const std::vector<uint8_t>& exp, int pl, int pw, int ph) {
        const uint8_t* fp = outf->GetReadPtr(pl);
        int pitch = outf->GetPitch(pl);
        for (int y = 0; y < ph && ok; ++y) {
            for (int x = 0; x < pw; ++x) {
                int got = fp[(size_t)y * pitch + x];
                if (got != exp[(size_t)y * pw + x]) {
                    printf("  MISMATCH %s frame %d %s: px(%d,%d) got=%d exp=%d\n",
                           tag, n, plane, x, y, got, exp[(size_t)y * pw + x]);
                    ok = false;
                    break;
                }
            }
        }
    };
    cmp("Y", rY, PLANAR_Y, w, h);
    cmp("U", rU, PLANAR_U, wUV, hUV);
    cmp("V", rV, PLANAR_V, wUV, hUV);
    return ok;
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------
struct FormatSpec {
    const char* pixel_type;
};

static const FormatSpec g_formats[] = {
    { "YV12" },
    { "YV16" },
    { "YV24" },
};

static const Params g_params[] = {
    { 1,   128, 1,   128 },  // upstream defaults
    { 0,   128, 0,   128 },  // nmin 0 boundary
    { 1,   0,   1,   0   },  // range 0 boundary
    { 128, 128, 128, 128 },  // nmin = range = 128 boundary
    { 200, 255, 3,   60  },  // nmin beyond the band + independent chroma params
};

static void runCase(IScriptEnvironment* env, const char* tag, PClip srcClip,
                    PClip noiseClip, const Params& p, const VideoInfo& vi) {
    AVSValue args[6] = { AVSValue(srcClip), AVSValue(noiseClip),
                         p.nmin_y, p.range_y, p.nmin_uv, p.range_uv };
    AVSValue ret;
    try {
        ret = env->Invoke("KNoiseClipOCL", AVSValue(args, 6));
    } catch (const AvisynthError& e) {
        printf("[FAIL] %s: Invoke threw: %s\n", tag, e.msg);
        ++g_failures;
        return;
    }
    if (!ret.IsClip()) {
        printf("[FAIL] %s: Invoke did not return a clip\n", tag);
        ++g_failures;
        return;
    }
    PClip filtered = ret.AsClip();

    bool ok = true;
    for (int n = 0; n < 2 && ok; ++n) {
        PVideoFrame outf;
        try {
            outf = filtered->GetFrame(n, env);
        } catch (const AvisynthError& e) {
            printf("  GetFrame threw: %s\n", e.msg);
            ++g_failures;
            return;
        }
        ++g_frame_checks;
        ok = refAndCompare(tag, vi, srcClip, noiseClip, outf, p, n);
    }
    ++g_cases;
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", tag);
    if (!ok) ++g_failures;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    AddVectoredExceptionHandler(1, eh);
    const char* pluginDll = argc > 1 ? argv[1] : nullptr;
    std::string pluginsDir = argc > 2 ? argv[2]
        : "Q:\\Downloads\\openclfilter\\sdk\\avs\\plugins64";

    // ---- install the plugin into the autoload directory ----
    if (pluginDll) {
        std::string dst = pluginsDir + "\\kfm_noiseclip_avs.dll";
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
    if (!env->FunctionExists("KNoiseClipOCL")) {
        printf("[FAIL] autoload: KNoiseClipOCL not registered after env creation\n");
        return 1;
    }
    if (!env->FunctionExists("KNoiseClip")) {
        printf("[FAIL] autoload: KNoiseClip alias not registered\n");
        return 1;
    }
    printf("autoload OK (KNoiseClipOCL + KNoiseClip alias registered)\n");

    // ---- deterministic-source cases ----
    const int W = 64, H = 48;
    const SourceKind g_kinds[] = { SRC_SMOOTH, SRC_NOISY, SRC_MIXED, SRC_ZERO, SRC_MAX };
    const char* kindNames[] = { "smooth", "noisy", "mixed", "zero", "max" };

    for (const FormatSpec& fs : g_formats) {
        char srcScript[160];
        sprintf_s(srcScript,
                  "BlankClip(width=%d, height=%d, pixel_type=\"%s\", length=4, color=$4080C0)",
                  W, H, fs.pixel_type);
        AVSValue srcClipV;
        try {
            srcClipV = env->Invoke("Eval", AVSValue(srcScript));
        } catch (const AvisynthError& e) {
            printf("[FAIL] source eval %s: %s\n", fs.pixel_type, e.msg);
            ++g_failures;
            continue;
        }
        const VideoInfo& cvi = srcClipV.AsClip()->GetVideoInfo();

        // --- parameter sweep over every src/noise source pairing ---
        for (int ks = 0; ks < 5; ++ks) {
            for (int kn = 0; kn < 5; ++kn) {
                // keep the matrix bounded: full diagonal + noisy/clean cross
                // pairs for the sweep, every pair at the defaults
                bool full = (ks == kn) || (ks == (int)SRC_NOISY && kn == (int)SRC_SMOOTH)
                         || (ks == (int)SRC_SMOOTH && kn == (int)SRC_NOISY);
                for (int ip = 0; ip < 5; ++ip) {
                    if (!full && ip != 0) continue;
                    PClip srcClip(new RandSource(cvi, 0xC0FFEEu ^ ((uint32_t)ks << 20), g_kinds[ks]));
                    PClip noiseClip(new RandSource(cvi, 0xFACE0FFu ^ ((uint32_t)kn << 20), g_kinds[kn]));
                    char tag[128];
                    sprintf_s(tag, "%s %s/%s nmin_y=%d range_y=%d nmin_uv=%d range_uv=%d",
                              fs.pixel_type, kindNames[ks], kindNames[kn],
                              g_params[ip].nmin_y, g_params[ip].range_y,
                              g_params[ip].nmin_uv, g_params[ip].range_uv);
                    runCase(env, tag, srcClip, noiseClip, g_params[ip], cvi);
                }
            }
        }

        // defaults-only invocation (named-param defaults path)
        {
            PClip srcClip(new RandSource(cvi, 0xD1CEu, SRC_MIXED));
            PClip noiseClip(new RandSource(cvi, 0xBEEFu, SRC_NOISY));
            AVSValue args[2] = { AVSValue(srcClip), AVSValue(noiseClip) };
            try {
                AVSValue ret = env->Invoke("KNoiseClipOCL", AVSValue(args, 2));
                if (!ret.IsClip()) throw AvisynthError("no clip");
                // check frame 1 pixel-exact too (defaults = 1/128/1/128)
                PVideoFrame outf = ret.AsClip()->GetFrame(1, env);
                bool ok = refAndCompare("defaults", cvi, srcClip, noiseClip, outf,
                                        Params{ 1, 128, 1, 128 }, 1);
                printf("[%s] %s defaults-only invocation\n", ok ? "PASS" : "FAIL", fs.pixel_type);
                if (!ok) ++g_failures;
            } catch (const AvisynthError& e) {
                printf("[FAIL] %s defaults-only: %s\n", fs.pixel_type, e.msg);
                ++g_failures;
            }
            ++g_cases;
        }

        // upstream-alias invocation (KNoiseClip by its upstream name)
        {
            PClip srcClip(new RandSource(cvi, 0xA1A5u, SRC_NOISY));
            PClip noiseClip(new RandSource(cvi, 0x5EEDu, SRC_SMOOTH));
            AVSValue args[6] = { AVSValue(srcClip), AVSValue(noiseClip), 5, 100, 0, 64 };
            try {
                AVSValue ret = env->Invoke("KNoiseClip", AVSValue(args, 6));
                if (!ret.IsClip()) throw AvisynthError("no clip");
                PVideoFrame outf = ret.AsClip()->GetFrame(1, env);
                bool ok = refAndCompare("alias", cvi, srcClip, noiseClip, outf,
                                        Params{ 5, 100, 0, 64 }, 1);
                printf("[%s] %s KNoiseClip alias invocation\n", ok ? "PASS" : "FAIL", fs.pixel_type);
                if (!ok) ++g_failures;
            } catch (const AvisynthError& e) {
                printf("[FAIL] %s alias: %s\n", fs.pixel_type, e.msg);
                ++g_failures;
            }
            ++g_cases;
        }
    }

    // ---- real script string with a real Source clip (ColorBars) ----
    try {
        const char* script =
            "KNoiseClipOCL(ColorBars(pixel_type=\"YV12\", width=64, height=48).Trim(0, 3), "
            "ColorBars(pixel_type=\"YV12\", width=64, height=48).Trim(0, 3).Invert(), "
            "nmin_y=2, range_y=90, nmin_uv=1, range_uv=100)";
        AVSValue ret = env->Invoke("Eval", AVSValue(script));
        if (!ret.IsClip()) throw AvisynthError("Eval did not return a clip");
        AVSValue srcV = env->Invoke("Eval",
            AVSValue("ColorBars(pixel_type=\"YV12\", width=64, height=48).Trim(0, 3)"));
        AVSValue noiseV = env->Invoke("Eval",
            AVSValue("ColorBars(pixel_type=\"YV12\", width=64, height=48).Trim(0, 3).Invert()"));
        PClip srcClip = srcV.AsClip();
        PClip noiseClip = noiseV.AsClip();
        PClip filtered = ret.AsClip();
        const VideoInfo& vi = filtered->GetVideoInfo();
        bool ok = true;
        for (int n = 0; n < 4 && ok; ++n) {
            PVideoFrame outf = filtered->GetFrame(n, env);
            ++g_frame_checks;
            ok = refAndCompare("script-string", vi, srcClip, noiseClip, outf,
                               Params{ 2, 90, 1, 100 }, n);
        }
        ++g_cases;
        printf("[%s] script-string ColorBars YV12 (nmin_y=2 range_y=90 nmin_uv=1 range_uv=100, 4 frames)\n",
               ok ? "PASS" : "FAIL");
        if (!ok) ++g_failures;
    } catch (const AvisynthError& e) {
        printf("[FAIL] script-string case: %s\n", e.msg);
        ++g_failures;
    }

    // ---- width%4 must throw (upstream constructor validation) ----
    {
        try {
            AVSValue srcV = env->Invoke("Eval",
                AVSValue("BlankClip(width=62, height=48, pixel_type=\"YV12\", length=2)"));
            AVSValue noiseV = env->Invoke("Eval",
                AVSValue("BlankClip(width=62, height=48, pixel_type=\"YV12\", length=2)"));
            AVSValue args[2] = { srcV, noiseV };
            try {
                env->Invoke("KNoiseClipOCL", AVSValue(args, 2));
                printf("[FAIL] width%%4!=0 accepted (expected throw)\n");
                ++g_failures;
            } catch (const AvisynthError&) {
                printf("[PASS] width=62 rejected (width must be multiple of 4)\n");
            }
        } catch (const AvisynthError& e) {
            printf("[FAIL] width-validation setup: %s\n", e.msg);
            ++g_failures;
        }
        ++g_cases;
    }

    // ---- height%4 must throw (upstream constructor validation) ----
    {
        try {
            AVSValue srcV = env->Invoke("Eval",
                AVSValue("BlankClip(width=64, height=46, pixel_type=\"YV12\", length=2)"));
            AVSValue noiseV = env->Invoke("Eval",
                AVSValue("BlankClip(width=64, height=46, pixel_type=\"YV12\", length=2)"));
            AVSValue args[2] = { srcV, noiseV };
            try {
                env->Invoke("KNoiseClipOCL", AVSValue(args, 2));
                printf("[FAIL] height%%4!=0 accepted (expected throw)\n");
                ++g_failures;
            } catch (const AvisynthError&) {
                printf("[PASS] height=46 rejected (height must be multiple of 4)\n");
            }
        } catch (const AvisynthError& e) {
            printf("[FAIL] height-validation setup: %s\n", e.msg);
            ++g_failures;
        }
        ++g_cases;
    }

    // ---- resolution mismatch must throw (upstream constructor validation) ----
    {
        try {
            AVSValue srcV = env->Invoke("Eval",
                AVSValue("BlankClip(width=64, height=48, pixel_type=\"YV12\", length=2)"));
            AVSValue noiseV = env->Invoke("Eval",
                AVSValue("BlankClip(width=32, height=48, pixel_type=\"YV12\", length=2)"));
            AVSValue args[2] = { srcV, noiseV };
            try {
                env->Invoke("KNoiseClipOCL", AVSValue(args, 2));
                printf("[FAIL] resolution mismatch accepted (expected throw)\n");
                ++g_failures;
            } catch (const AvisynthError&) {
                printf("[PASS] resolution mismatch rejected (src and noiseclip must be same resoluction)\n");
            }
        } catch (const AvisynthError& e) {
            printf("[FAIL] mismatch-validation setup: %s\n", e.msg);
            ++g_failures;
        }
        ++g_cases;
    }

    // ---- 16-bit input must throw (upstream: ComponentSize 1 only) ----
    {
        try {
            AVSValue srcV = env->Invoke("Eval",
                AVSValue("BlankClip(width=64, height=48, pixel_type=\"YUV420P16\", length=2)"));
            AVSValue noiseV = env->Invoke("Eval",
                AVSValue("BlankClip(width=64, height=48, pixel_type=\"YUV420P16\", length=2)"));
            AVSValue args[2] = { srcV, noiseV };
            try {
                env->Invoke("KNoiseClipOCL", AVSValue(args, 2));
                printf("[FAIL] 16-bit input accepted (expected throw)\n");
                ++g_failures;
            } catch (const AvisynthError&) {
                printf("[PASS] YUV420P16 rejected (Unsupported pixel format)\n");
            }
        } catch (const AvisynthError& e) {
            printf("[FAIL] 16-bit-validation setup: %s\n", e.msg);
            ++g_failures;
        }
        ++g_cases;
    }

    printf("\n=== KNoiseClipOCL: %d cases (%d frame checks), %d failures ===\n",
           g_cases, g_frame_checks, g_failures);

    env->DeleteScriptEnvironment();
    // NOTE: deliberately not FreeLibrary — AviSynth+ keeps internal threads
    // referencing the DLL at shutdown.
    return g_failures ? 1 : 0;
}
