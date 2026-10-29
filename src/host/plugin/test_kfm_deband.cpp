// test_kfm_deband.cpp — headless verification of the KDebandOCL AviSynth+
// plugin (third host plugin of the OpenCL port) against a CPU reference
// computed in this exe.
//
// REFERENCE CHOICE (stated, per the task): the reference is computed HERE,
// in this exe, from the actual input frame pixels, transcribing the repo's
// CPU-mirror/golden semantics (sim/kfm_deband_ref.cpp +
// python/run_kfm_deband.py — the golden-pinned authority, == the upstream
// cpu_reduce_banding twin in KDeband.cu) plus the upstream KDeband host
// pipeline: ONE reduce_banding pass per plane (Y at luma dims, U/V at chroma
// dims), thresh host-scaled by scaleParam
// ((int)(thresh*(1<<(bits-8))+0.5f)), and ONE shared XorShift rand stream
// (seed 0, width*height*2 bytes at LUMA dims, uint32 little-endian words)
// indexed by every plane with its own rand_step = width*height.  All
// arithmetic is integer; the only float op is the thresh scale (IEEE float32
// per-op, MSVC x64 /fp:precise, no FMA contraction).
//
// Coverage:
//   - 6 formats (YV12/YV16/YV24 + 16-bit 420/422/444) — plane variants via
//     the three subsampling geometries,
//   - parameter sweep incl. upstream defaults (range 25, thresh 1,
//     sample_mode 1, blur_first false) and boundaries: range 0 / 127,
//     thresh 0 / large, sample_mode 0/1/2, blur_first true/false,
//   - three synthetic source kinds: SMOOTH (smooth gradient), NOISY
//     (full-range noise), BANDED (flat bands — the debander's target),
//   - every pixel of every plane compared against the in-exe reference,
//   - defaults-only + upstream-alias (KDeband) invocations, a real script
//     string over ColorBars, and validation: sample_mode=3 and range=128
//     must throw.
//
// Load path: the exe copies the plugin DLL into the AviSynth+ autoload
// directory (registry HKLM\Software\AviSynth plugins64), creates the script
// environment, verifies autoload via FunctionExists, then invokes
// KDebandOCL (and the upstream KDeband alias) and compares EVERY pixel of
// every plane.
//
// Build: see build_host_deband.bat.  Usage: test_kfm_deband.exe
// <plugin_dll_path> [plugins_dir]
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

// ---------------------------------------------------------------------------
// deterministic source clips (scripted pixel content)
// ---------------------------------------------------------------------------
static uint32_t mix32(uint32_t a) {
    a ^= a >> 16; a *= 0x7feb352du;
    a ^= a >> 15; a *= 0x846ca68bu;
    a ^= a >> 16;
    return a;
}

enum SourceKind {
    SRC_SMOOTH,  // smooth horizontal/vertical gradient (clean content)
    SRC_NOISY,   // full-range per-pixel noise (diff gate mostly rejects)
    SRC_BANDED,  // flat bands + slight dither (the debander's target)
};

template <typename PX>
static void fillPattern(uint8_t* dstp, int pitch, int w, int h, int maxv,
                        uint32_t salt, SourceKind kind) {
    for (int y = 0; y < h; ++y) {
        PX* row = (PX*)(dstp + (size_t)y * pitch);
        for (int x = 0; x < w; ++x) {
            int v;
            switch (kind) {
            case SRC_SMOOTH:
                // smooth 2D gradient over the full range
                v = (int)(((int64_t)x * (maxv / 2)) / (w > 1 ? w - 1 : 1)
                          + ((int64_t)y * (maxv / 2)) / (h > 1 ? h - 1 : 1));
                break;
            case SRC_NOISY:
                v = (int)(mix32((uint32_t)x * 17u + (uint32_t)y * 1013u + salt)
                          & (uint32_t)maxv);
                break;
            default:  // SRC_BANDED: flat bands (steps of maxv/8) + tiny dither
                v = (int)(((x / 7 + y / 5) % 8) * (maxv / 8));
                v += (int)(mix32(salt ^ ((uint32_t)x * 31u + (uint32_t)y * 7u)) % 5u) - 2;
                break;
            }
            row[x] = (PX)(v < 0 ? 0 : v > maxv ? maxv : v);
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
        const int maxv = (1 << vi_.BitsPerComponent()) - 1;
        const int w = vi_.width, h = vi_.height;
        const int wUV = w >> vi_.GetPlaneWidthSubsampling(PLANAR_U);
        const int hUV = h >> vi_.GetPlaneHeightSubsampling(PLANAR_U);
        const int px = vi_.ComponentSize();
        uint32_t s = salt_ + (uint32_t)n * 7919u;  // content varies per frame
        if (px == 1) {
            fillPattern<uint8_t>(f->GetWritePtr(PLANAR_Y), f->GetPitch(PLANAR_Y), w, h, maxv, s, kind_);
            fillPattern<uint8_t>(f->GetWritePtr(PLANAR_U), f->GetPitch(PLANAR_U), wUV, hUV, maxv, s + 11u, kind_);
            fillPattern<uint8_t>(f->GetWritePtr(PLANAR_V), f->GetPitch(PLANAR_V), wUV, hUV, maxv, s + 23u, kind_);
        } else {
            fillPattern<uint16_t>(f->GetWritePtr(PLANAR_Y), f->GetPitch(PLANAR_Y), w, h, maxv, s, kind_);
            fillPattern<uint16_t>(f->GetWritePtr(PLANAR_U), f->GetPitch(PLANAR_U), wUV, hUV, maxv, s + 11u, kind_);
            fillPattern<uint16_t>(f->GetWritePtr(PLANAR_V), f->GetPitch(PLANAR_V), wUV, hUV, maxv, s + 23u, kind_);
        }
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

// upstream KDeband.cu XorShift (verbatim)
struct XorShift {
    uint32_t x, y, z, w;
    explicit XorShift(int seed)
        : x((uint32_t)(123456789 - seed)), y(362436069u), z(521288629u), w(88675123u) {}
    uint32_t next() {
        uint32_t t;
        t = x ^ (x << 11);
        x = y; y = z; z = w;
        w ^= t ^ (t >> 8) ^ (w >> 19);
        return w;
    }
};

// upstream CreateDebandRandom (verbatim): seed 0, luma dims, 2 bytes/pixel
static std::vector<uint8_t> createDebandRandom(int width, int height, int seed) {
    const int length = width * height * 2;
    std::vector<uint8_t> rand_buf((size_t)length);
    XorShift rng(seed);
    size_t i = 0;
    for (; i + 4 <= (size_t)length; i += 4) {
        uint32_t r = rng.next();
        memcpy(&rand_buf[i], &r, 4);
    }
    if (i < (size_t)length) {
        uint32_t r = rng.next();
        memcpy(&rand_buf[i], &r, (size_t)length - i);
    }
    return rand_buf;
}

// upstream random_range (verbatim; range <= 127 is validated by the plugin)
static int random_range(int random, int range) {
    return ((((range << 1) + 1) * random) >> 8) - range;
}

// upstream cpu_reduce_banding (verbatim twin; pitch == width packed planes)
template <typename PX>
static void ref_reduce_banding(std::vector<PX>& dst, const std::vector<PX>& src,
                               const std::vector<uint8_t>& rand,
                               int width, int height, int range, int thresh,
                               int sample_mode, bool blur_first) {
    const int pitch = width;
    dst.assign((size_t)pitch * height, (PX)0);
    int rand_step = width * height;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int offset = y * pitch + x;

            int range_limited = (std::min)((std::min)(range, y),
                (std::min)(height - y - 1, (std::min)(x, width - x - 1)));
            int refA = random_range(rand[offset + rand_step * 0], range_limited);
            int refB = random_range(rand[offset + rand_step * 1], range_limited);

            int src_val = (int)src[offset];
            int avg, diff;

            if (sample_mode == 0) {
                int ref = refA * pitch + refB;

                avg = (int)src[offset + ref];
                diff = abs(src_val - avg);

            } else if (sample_mode == 1) {
                int ref = refA * pitch + refB;

                int ref_p = (int)src[offset + ref];
                int ref_m = (int)src[offset - ref];

                avg = (ref_p + ref_m) >> 1;
                diff = blur_first
                    ? abs(src_val - avg)
                    : (std::max)(abs(src_val - ref_p), abs(src_val - ref_m));
            } else {
                int ref_0 = refA * pitch + refB;
                int ref_1 = refA - refB * pitch;

                int ref_0p = (int)src[offset + ref_0];
                int ref_0m = (int)src[offset - ref_0];
                int ref_1p = (int)src[offset + ref_1];
                int ref_1m = (int)src[offset - ref_1];

                avg = (ref_0p + ref_0m + ref_1p + ref_1m) >> 2;
                diff = blur_first
                    ? abs(src_val - avg)
                    : (std::max)(
                        (std::max)(abs(src_val - ref_0p), abs(src_val - ref_0m)),
                        (std::max)(abs(src_val - ref_1p), abs(src_val - ref_1m)));
            }

            dst[offset] = (PX)((diff <= thresh) ? avg : src_val);
        }
    }
}

// The rand stream depends only on the luma dims; build it once (upstream
// builds it in the constructor and shares it across frames too).
static std::vector<uint8_t> g_rand;

// The GetFrame into the source clip needs an IScriptEnvironment; pass it
// through a file-scope pointer set by the driver (single-threaded test exe).
static IScriptEnvironment* g_env = nullptr;
static IScriptEnvironment* env_of(const VideoInfo&) { return g_env; }

// ---------------------------------------------------------------------------
// case pipeline reference (upstream GetFrameT semantics: one pass per plane,
// shared luma-dims rand) + full-plane comparison
// ---------------------------------------------------------------------------
struct Params {
    int range; float thresh; int sample_mode; bool blur_first;
};

template <typename PX>
static bool refAndCompare(const char* tag, const VideoInfo& vi, PClip srcClip,
                          PVideoFrame outf, const Params& p, int n) {
    const int w = vi.width, h = vi.height;
    const int lx = vi.GetPlaneWidthSubsampling(PLANAR_U);
    const int ly = vi.GetPlaneHeightSubsampling(PLANAR_U);
    const int wUV = w >> lx, hUV = h >> ly;
    const int bits = vi.BitsPerComponent();
    const int thresh = (int)((float)p.thresh * (float)(1 << (bits - 8)) + 0.5f);

    PVideoFrame f = srcClip->GetFrame(n, env_of(vi));
    std::vector<PX> sY, sU, sV;
    readPlane(sY, f, PLANAR_Y, w, h);
    readPlane(sU, f, PLANAR_U, wUV, hUV);
    readPlane(sV, f, PLANAR_V, wUV, hUV);

    std::vector<PX> rY, rU, rV;
    // ONE shared luma-dims rand stream for all three planes (upstream verbatim)
    ref_reduce_banding(rY, sY, g_rand, w, h, p.range, thresh, p.sample_mode, p.blur_first);
    ref_reduce_banding(rU, sU, g_rand, wUV, hUV, p.range, thresh, p.sample_mode, p.blur_first);
    ref_reduce_banding(rV, sV, g_rand, wUV, hUV, p.range, thresh, p.sample_mode, p.blur_first);

    // ---- compare every pixel of every plane ----
    bool ok = true;
    auto cmp = [&](const char* plane, const std::vector<PX>& exp, int pl, int pw, int ph) {
        const uint8_t* fp = outf->GetReadPtr(pl);
        int pitch = outf->GetPitch(pl);
        for (int y = 0; y < ph && ok; ++y) {
            for (int x = 0; x < pw; ++x) {
                PX got;
                memcpy(&got, fp + (size_t)y * pitch + (size_t)x * sizeof(PX), sizeof(PX));
                if ((int)got != (int)exp[(size_t)y * pw + x]) {
                    printf("  MISMATCH %s frame %d %s: px(%d,%d) got=%d exp=%d\n",
                           tag, n, plane, x, y, (int)got, (int)exp[(size_t)y * pw + x]);
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
    int bits;
};

static const FormatSpec g_formats[] = {
    { "YV12",      8 },
    { "YV16",      8 },
    { "YV24",      8 },
    { "YUV420P16", 16 },
    { "YUV422P16", 16 },
    { "YUV444P16", 16 },
};

static const Params g_params[] = {
    { 25, 1.0f, 1, false },  // upstream defaults
    { 0,  1.0f, 1, false },  // range 0: only the pixel itself is sampled -> output == src
    { 127, 2.0f, 0, false }, // range 127 boundary, sample_mode 0
    { 127, 2.0f, 1, true },  // range 127, sample_mode 1, blur_first
    { 10, 0.0f, 2, false },  // thresh 0: only exact-equal refs pass the gate
    { 40, 64.0f, 2, true },  // large thresh: gate always passes -> smooth avg
    { 3,  1.0f, 0, true },   // small sweep corner
};

static void runCase(IScriptEnvironment* env, const char* tag, PClip clip,
                    const Params& p, const VideoInfo& vi, int bits) {
    AVSValue args[5] = { AVSValue(clip), p.range, p.thresh, p.sample_mode, p.blur_first };
    AVSValue ret;
    try {
        ret = env->Invoke("KDebandOCL", AVSValue(args, 5));
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
        if (bits == 8)
            ok = refAndCompare<uint8_t>(tag, vi, clip, outf, p, n);
        else
            ok = refAndCompare<uint16_t>(tag, vi, clip, outf, p, n);
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
        std::string dst = pluginsDir + "\\kfm_deband_avs.dll";
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
    if (!env->FunctionExists("KDebandOCL")) {
        printf("[FAIL] autoload: KDebandOCL not registered after env creation\n");
        return 1;
    }
    if (!env->FunctionExists("KDeband")) {
        printf("[FAIL] autoload: KDeband alias not registered\n");
        return 1;
    }
    printf("autoload OK (KDebandOCL + KDeband alias registered)\n");

    // ---- deterministic-source cases ----
    const int W = 64, H = 48;
    const SourceKind g_kinds[] = { SRC_SMOOTH, SRC_NOISY, SRC_BANDED };
    const char* kindNames[] = { "smooth", "noisy", "banded" };

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
        g_rand = createDebandRandom(cvi.width, cvi.height, 0);

        for (int k = 0; k < 3; ++k) {
            PClip clip(new RandSource(cvi, 0xC0FFEEu ^ (uint32_t)fs.bits ^ ((uint32_t)k << 20), g_kinds[k]));
            for (const Params& p : g_params) {
                char tag[128];
                sprintf_s(tag, "%s %s range=%d thresh=%.1f mode=%d blur=%d",
                          fs.pixel_type, kindNames[k], p.range, p.thresh,
                          p.sample_mode, (int)p.blur_first);
                runCase(env, tag, clip, p, cvi, fs.bits);
            }
        }

        // defaults-only invocation (named-param defaults path)
        {
            PClip clip(new RandSource(cvi, 0xD1CEu ^ (uint32_t)fs.bits, SRC_BANDED));
            AVSValue args[1] = { AVSValue(clip) };
            try {
                AVSValue ret = env->Invoke("KDebandOCL", AVSValue(args, 1));
                if (!ret.IsClip()) throw AvisynthError("no clip");
                // check frame 1 pixel-exact too (defaults = range 25, thresh 1, mode 1)
                PVideoFrame outf = ret.AsClip()->GetFrame(1, env);
                bool ok = (fs.bits == 8)
                    ? refAndCompare<uint8_t>("defaults", cvi, clip, outf, Params{ 25, 1.0f, 1, false }, 1)
                    : refAndCompare<uint16_t>("defaults", cvi, clip, outf, Params{ 25, 1.0f, 1, false }, 1);
                printf("[%s] %s defaults-only invocation\n", ok ? "PASS" : "FAIL", fs.pixel_type);
                if (!ok) ++g_failures;
            } catch (const AvisynthError& e) {
                printf("[FAIL] %s defaults-only: %s\n", fs.pixel_type, e.msg);
                ++g_failures;
            }
            ++g_cases;
        }

        // upstream-alias invocation (KDeband by its upstream name)
        {
            PClip clip(new RandSource(cvi, 0xA1A5u ^ (uint32_t)fs.bits, SRC_SMOOTH));
            AVSValue args[5] = { AVSValue(clip), 12, 4.0, 2, true };
            try {
                AVSValue ret = env->Invoke("KDeband", AVSValue(args, 5));
                if (!ret.IsClip()) throw AvisynthError("no clip");
                PVideoFrame outf = ret.AsClip()->GetFrame(1, env);
                bool ok = (fs.bits == 8)
                    ? refAndCompare<uint8_t>("alias", cvi, clip, outf, Params{ 12, 4.0f, 2, true }, 1)
                    : refAndCompare<uint16_t>("alias", cvi, clip, outf, Params{ 12, 4.0f, 2, true }, 1);
                printf("[%s] %s KDeband alias invocation\n", ok ? "PASS" : "FAIL", fs.pixel_type);
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
            "KDebandOCL(ColorBars(pixel_type=\"YV12\", width=64, height=48).Trim(0, 3), "
            "range=16, thresh=3.0, sample_mode=2, blur_first=true)";
        AVSValue ret = env->Invoke("Eval", AVSValue(script));
        if (!ret.IsClip()) throw AvisynthError("Eval did not return a clip");
        AVSValue srcV = env->Invoke("Eval",
            AVSValue("ColorBars(pixel_type=\"YV12\", width=64, height=48).Trim(0, 3)"));
        PClip srcClip = srcV.AsClip();
        PClip filtered = ret.AsClip();
        const VideoInfo& vi = filtered->GetVideoInfo();
        g_rand = createDebandRandom(vi.width, vi.height, 0);
        bool ok = true;
        for (int n = 0; n < 4 && ok; ++n) {
            PVideoFrame outf = filtered->GetFrame(n, env);
            ++g_frame_checks;
            ok = refAndCompare<uint8_t>("script-string", vi, srcClip, outf,
                                        Params{ 16, 3.0f, 2, true }, n);
        }
        ++g_cases;
        printf("[%s] script-string ColorBars YV12 (range=16 thresh=3.0 mode=2 blur, 4 frames)\n",
               ok ? "PASS" : "FAIL");
        if (!ok) ++g_failures;
    } catch (const AvisynthError& e) {
        printf("[FAIL] script-string case: %s\n", e.msg);
        ++g_failures;
    }

    // ---- sample_mode=3 must throw (upstream validation) ----
    {
        AVSValue srcClipV;
        try {
            srcClipV = env->Invoke("Eval", AVSValue("BlankClip(width=64, height=48, pixel_type=\"YV12\", length=2)"));
            g_rand = createDebandRandom(64, 48, 0);
            AVSValue args[5] = { srcClipV, 25, 1.0, 3, false };
            try {
                env->Invoke("KDebandOCL", AVSValue(args, 5));
                printf("[FAIL] sample_mode=3 accepted (expected throw)\n");
                ++g_failures;
            } catch (const AvisynthError&) {
                printf("[PASS] sample_mode=3 rejected (sample_mode must be 0,1,2)\n");
            }
        } catch (const AvisynthError& e) {
            printf("[FAIL] sample_mode-validation setup: %s\n", e.msg);
            ++g_failures;
        }
        ++g_cases;
    }

    // ---- range=128 must throw (documented deviation 1: upstream char truncation) ----
    {
        AVSValue srcClipV;
        try {
            srcClipV = env->Invoke("Eval", AVSValue("BlankClip(width=64, height=48, pixel_type=\"YV12\", length=2)"));
            AVSValue args[5] = { srcClipV, 128, 1.0, 1, false };
            try {
                env->Invoke("KDebandOCL", AVSValue(args, 5));
                printf("[FAIL] range=128 accepted (expected throw)\n");
                ++g_failures;
            } catch (const AvisynthError&) {
                printf("[PASS] range=128 rejected (range must be 0..127)\n");
            }
        } catch (const AvisynthError& e) {
            printf("[FAIL] range-validation setup: %s\n", e.msg);
            ++g_failures;
        }
        ++g_cases;
    }

    printf("\n=== KDebandOCL: %d cases (%d frame checks), %d failures ===\n",
           g_cases, g_frame_checks, g_failures);

    env->DeleteScriptEnvironment();
    // NOTE: deliberately not FreeLibrary — AviSynth+ keeps internal threads
    // referencing the DLL at shutdown.
    return g_failures ? 1 : 0;
}
