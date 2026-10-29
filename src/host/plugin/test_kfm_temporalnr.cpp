// test_kfm_temporalnr.cpp — headless verification of the KTemporalNROCL
// AviSynth+ plugin (second host plugin of the OpenCL port) against a CPU
// reference computed in this exe.
//
// REFERENCE CHOICE (stated, per the task): the reference is computed HERE,
// in this exe, from the actual input frame pixels, transcribing the repo's
// CPU-mirror/golden semantics (sim/kfm_temporalnr_ref.cpp +
// python/run_kfm_temporalnr.py — the golden-pinned authority, == the
// upstream cpu_temporal_nr twin in KDeband.cu) for kf_temporal_nr, plus the
// upstream KTemporalNR::run host pipeline: window frames at
// clamp(n-dist+i, 0, num_frames-1), mid = dist, thresh host-scaled by
// scaleParam ((int)(thresh*(1<<(bits-8))+0.5f)), one pass per plane at that
// plane's own dims.  All float ops are IEEE float32 per-op (MSVC x64 SSE2,
// /fp:precise default; no FMA contraction), matching the mirror's
// -ffp-contract=off and the device build (-cl-fp32-correctly-rounded-divide-
// sqrt; see kfm_temporalnr.cl's DEVICE RECORD).
//
// The plugin holds NO cross-frame state (see the plugin header's state model):
// each output frame is a pure function of the child window + parameters, so
// the test fetches the whole window from the source clip and compares it
// directly.  The cases nevertheless exercise the temporal dimension:
//   - multi-frame param sweep over 6 formats x 6 param sets, frames 0..2
//     (first-frame clamping at n=0, interior at n>=dist, tail clamp via
//     dist=16 > num_frames),
//   - a CONSTANT clip (identical every frame): with thresh >= 0 every window
//     pixel equals the centre, so the average must converge EXACTLY to the
//     frame content for every dist (accumulation-convergence check),
//   - a SCENE-CHANGE clip (abrupt content jump at frame 2): the thresh gate
//     must drop the changed window members per pixel.
//
// Load path: the exe copies the plugin DLL into the AviSynth+ autoload
// directory (registry HKLM\Software\AviSynth plugins64), creates the script
// environment, verifies autoload via FunctionExists, then invokes
// KTemporalNROCL (and the upstream KTemporalNR alias, plus a real script
// string via Eval) and compares EVERY pixel of every plane.
//
// Build: see build_host_temporalnr.bat.  Usage: test_kfm_temporalnr.exe
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

template <typename PX>
static void fillPattern(uint8_t* dstp, int pitch, int w, int h, int maxv, uint32_t salt) {
    for (int y = 0; y < h; ++y) {
        PX* row = (PX*)(dstp + (size_t)y * pitch);
        for (int x = 0; x < w; ++x) {
            uint32_t r = mix32((uint32_t)x * 17u + (uint32_t)y * 1013u + salt);
            int v = (int)(r & (uint32_t)maxv);
            // low-frequency structure + noise so the thresh gate selects a
            // varying per-pixel subset of the window
            if ((x / 4) & 1) v += maxv / 8;
            if ((y / 8) & 1) v += maxv / 6;
            v += (int)(mix32(salt ^ ((uint32_t)x * 31u + (uint32_t)y * 7u)) % 16u) - 8;
            row[x] = (PX)(v < 0 ? 0 : v > maxv ? maxv : v);
        }
    }
}

enum SourceKind {
    SRC_RANDOM,       // independent content every frame (covers scene-change-
                      // like abrupt changes across the window)
    SRC_CONSTANT,     // identical content every frame (accumulation converges)
    SRC_SCENECHANGE,  // identical within a scene, abrupt jump at frame 2
};

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
        uint32_t s;
        switch (kind_) {
        case SRC_CONSTANT:    s = salt_; break;                    // frame-independent
        case SRC_SCENECHANGE: s = salt_ + (n >= 2 ? 911u : 0u); break;
        default:              s = salt_ + (uint32_t)n * 7919u; break;
        }
        if (px == 1) {
            fillPattern<uint8_t>(f->GetWritePtr(PLANAR_Y), f->GetPitch(PLANAR_Y), w, h, maxv, s);
            fillPattern<uint8_t>(f->GetWritePtr(PLANAR_U), f->GetPitch(PLANAR_U), wUV, hUV, maxv, s + 11u);
            fillPattern<uint8_t>(f->GetWritePtr(PLANAR_V), f->GetPitch(PLANAR_V), wUV, hUV, maxv, s + 23u);
        } else {
            fillPattern<uint16_t>(f->GetWritePtr(PLANAR_Y), f->GetPitch(PLANAR_Y), w, h, maxv, s);
            fillPattern<uint16_t>(f->GetWritePtr(PLANAR_U), f->GetPitch(PLANAR_U), wUV, hUV, maxv, s + 11u);
            fillPattern<uint16_t>(f->GetWritePtr(PLANAR_V), f->GetPitch(PLANAR_V), wUV, hUV, maxv, s + 23u);
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

template <typename PX>
static void ref_temporal_nr(std::vector<PX>& dst, const std::vector<std::vector<PX>>& frames,
                            int width, int height, int nframes, int mid, int thresh) {
    const int pitch = width;  // packed planes, pitch == width (kernel layout)
    dst.assign((size_t)pitch * height, (PX)0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t idx = (size_t)y * pitch + x;
            const int center = (int)frames[mid][idx];
            int count = 0, sum = 0;
            for (int i = 0; i < nframes; ++i) {
                const int ref = (int)frames[i][idx];
                int diff = ref - center;
                if (diff < 0) diff = -diff;
                if (diff <= thresh) { ++count; sum += ref; }
            }
            // float32 per-op: (float)sum/count + 0.5f, truncated (count >= 1)
            float avg = (float)sum / (float)count + 0.5f;
            dst[idx] = (PX)(int)avg;
        }
    }
}

// The window fetch needs an IScriptEnvironment; pass it through a file-scope
// pointer set by the driver (single-threaded test exe).
static IScriptEnvironment* g_env = nullptr;
static IScriptEnvironment* env_of(const VideoInfo&) { return g_env; }

// ---------------------------------------------------------------------------
// case pipeline reference (upstream run() semantics: clamped window + per-plane
// kernel) + full-plane comparison
// ---------------------------------------------------------------------------
struct Params {
    int dist; float thresh;
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
    const int nframes = p.dist * 2 + 1;
    const int mid = p.dist;

    // fetch the clamped window from the source (upstream verbatim)
    std::vector<std::vector<PX>> wY(nframes), wU(nframes), wV(nframes);
    for (int i = 0; i < nframes; ++i) {
        const int fn = (std::max)(0, (std::min)(n - p.dist + i, vi.num_frames - 1));
        PVideoFrame f = srcClip->GetFrame(fn, env_of(vi));
        readPlane(wY[i], f, PLANAR_Y, w, h);
        readPlane(wU[i], f, PLANAR_U, wUV, hUV);
        readPlane(wV[i], f, PLANAR_V, wUV, hUV);
    }

    std::vector<PX> rY, rU, rV;
    ref_temporal_nr(rY, wY, w, h, nframes, mid, thresh);
    ref_temporal_nr(rU, wU, wUV, hUV, nframes, mid, thresh);
    ref_temporal_nr(rV, wV, wUV, hUV, nframes, mid, thresh);

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
    { 3, 1.0f },     // upstream defaults
    { 0, 1.0f },     // radius 0: output == centre frame (window of 1)
    { 1, 3.0f },
    { 5, 8.0f },
    { 2, 0.0f },     // thresh 0: only exact-equal window members average
    { 16, 4.0f },    // dist beyond the clip length: heavy boundary clamping
};

static void runCase(IScriptEnvironment* env, const char* tag, PClip clip,
                    const Params& p, const VideoInfo& vi, int bits) {
    AVSValue args[3] = { AVSValue(clip), p.dist, p.thresh };
    AVSValue ret;
    try {
        ret = env->Invoke("KTemporalNROCL", AVSValue(args, 3));
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
    for (int n = 0; n < 3 && ok; ++n) {
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
        std::string dst = pluginsDir + "\\kfm_temporalnr_avs.dll";
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
    if (!env->FunctionExists("KTemporalNROCL")) {
        printf("[FAIL] autoload: KTemporalNROCL not registered after env creation\n");
        return 1;
    }
    if (!env->FunctionExists("KTemporalNR")) {
        printf("[FAIL] autoload: KTemporalNR alias not registered\n");
        return 1;
    }
    printf("autoload OK (KTemporalNROCL + KTemporalNR alias registered)\n");

    // ---- deterministic-source cases ----
    const int W = 64, H = 48;
    for (const FormatSpec& fs : g_formats) {
        // Build the source through the script environment so pixel_type
        // handling matches real scripts (BlankClip only pins the format;
        // length=4 gives a real multi-frame clip for the temporal window).
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

        // random per-frame content: covers abrupt inter-frame changes + all
        // three clamp regimes (first frames, interior, tail)
        {
            PClip clip(new RandSource(cvi, 0xC0FFEEu ^ (uint32_t)fs.bits, SRC_RANDOM));
            for (const Params& p : g_params) {
                char tag[128];
                sprintf_s(tag, "%s random dist=%d thresh=%.1f", fs.pixel_type, p.dist, p.thresh);
                runCase(env, tag, clip, p, cvi, fs.bits);
            }
        }
        // constant clip: accumulation converges to the frame content for
        // every dist (sum of nframes identical values / nframes, +0.5, trunc)
        {
            PClip clip(new RandSource(cvi, 0x51DEu ^ (uint32_t)fs.bits, SRC_CONSTANT));
            for (const Params& p : g_params) {
                if (p.dist < 1) continue;  // convergence is the point of dist>0
                char tag[128];
                sprintf_s(tag, "%s constant dist=%d thresh=%.1f (converged)", fs.pixel_type, p.dist, p.thresh);
                runCase(env, tag, clip, p, cvi, fs.bits);
            }
        }
        // scene-change clip: identical content per scene, abrupt jump at
        // frame 2 (windows spanning the cut exercise the thresh gate)
        {
            PClip clip(new RandSource(cvi, 0xACEu ^ (uint32_t)fs.bits, SRC_SCENECHANGE));
            for (const Params& p : g_params) {
                char tag[128];
                sprintf_s(tag, "%s scene-change dist=%d thresh=%.1f", fs.pixel_type, p.dist, p.thresh);
                runCase(env, tag, clip, p, cvi, fs.bits);
            }
        }

        // defaults-only invocation (named-param defaults path)
        {
            PClip clip(new RandSource(cvi, 0xD1CEu ^ (uint32_t)fs.bits, SRC_RANDOM));
            AVSValue args[1] = { AVSValue(clip) };
            try {
                AVSValue ret = env->Invoke("KTemporalNROCL", AVSValue(args, 1));
                if (!ret.IsClip()) throw AvisynthError("no clip");
                // check frame 1 pixel-exact too (defaults = dist 3, thresh 1)
                PVideoFrame outf = ret.AsClip()->GetFrame(1, env);
                bool ok = (fs.bits == 8)
                    ? refAndCompare<uint8_t>("defaults", cvi, clip, outf, Params{ 3, 1.0f }, 1)
                    : refAndCompare<uint16_t>("defaults", cvi, clip, outf, Params{ 3, 1.0f }, 1);
                printf("[%s] %s defaults-only invocation\n", ok ? "PASS" : "FAIL", fs.pixel_type);
                if (!ok) ++g_failures;
            } catch (const AvisynthError& e) {
                printf("[FAIL] %s defaults-only: %s\n", fs.pixel_type, e.msg);
                ++g_failures;
            }
            ++g_cases;
        }

        // upstream-alias invocation (KTemporalNR by its upstream name)
        {
            PClip clip(new RandSource(cvi, 0xA1A5u ^ (uint32_t)fs.bits, SRC_RANDOM));
            AVSValue args[3] = { AVSValue(clip), 2, 5.0 };
            try {
                AVSValue ret = env->Invoke("KTemporalNR", AVSValue(args, 3));
                if (!ret.IsClip()) throw AvisynthError("no clip");
                PVideoFrame outf = ret.AsClip()->GetFrame(2, env);
                bool ok = (fs.bits == 8)
                    ? refAndCompare<uint8_t>("alias", cvi, clip, outf, Params{ 2, 5.0f }, 2)
                    : refAndCompare<uint16_t>("alias", cvi, clip, outf, Params{ 2, 5.0f }, 2);
                printf("[%s] %s KTemporalNR alias invocation\n", ok ? "PASS" : "FAIL", fs.pixel_type);
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
            "KTemporalNROCL(ColorBars(pixel_type=\"YV12\", width=64, height=48).Trim(0, 7), "
            "dist=2, thresh=6.0)";
        AVSValue ret = env->Invoke("Eval", AVSValue(script));
        if (!ret.IsClip()) throw AvisynthError("Eval did not return a clip");
        AVSValue srcV = env->Invoke("Eval",
            AVSValue("ColorBars(pixel_type=\"YV12\", width=64, height=48).Trim(0, 7)"));
        PClip srcClip = srcV.AsClip();
        PClip filtered = ret.AsClip();
        const VideoInfo& vi = filtered->GetVideoInfo();
        bool ok = true;
        for (int n = 0; n < 4 && ok; ++n) {
            PVideoFrame outf = filtered->GetFrame(n, env);
            ++g_frame_checks;
            ok = refAndCompare<uint8_t>("script-string", vi, srcClip, outf, Params{ 2, 6.0f }, n);
        }
        ++g_cases;
        printf("[%s] script-string ColorBars YV12 (dist=2 thresh=6.0, 4 frames)\n",
               ok ? "PASS" : "FAIL");
        if (!ok) ++g_failures;
    } catch (const AvisynthError& e) {
        printf("[FAIL] script-string case: %s\n", e.msg);
        ++g_failures;
    }

    // ---- dist > MAX_TEMPORAL_DIST must throw (upstream validation) ----
    {
        AVSValue srcClipV;
        try {
            srcClipV = env->Invoke("Eval", AVSValue("BlankClip(width=64, height=48, pixel_type=\"YV12\", length=4)"));
            AVSValue args[3] = { srcClipV, 17, 1.0 };
            try {
                env->Invoke("KTemporalNROCL", AVSValue(args, 3));
                printf("[FAIL] dist=17 accepted (expected throw)\n");
                ++g_failures;
            } catch (const AvisynthError&) {
                printf("[PASS] dist=17 rejected (maximum dist is 16)\n");
            }
        } catch (const AvisynthError& e) {
            printf("[FAIL] dist-validation setup: %s\n", e.msg);
            ++g_failures;
        }
        ++g_cases;
    }

    printf("\n=== KTemporalNROCL: %d cases (%d frame checks), %d failures ===\n",
           g_cases, g_frame_checks, g_failures);

    env->DeleteScriptEnvironment();
    // NOTE: deliberately not FreeLibrary — AviSynth+ keeps internal threads
    // referencing the DLL at shutdown.
    return g_failures ? 1 : 0;
}
