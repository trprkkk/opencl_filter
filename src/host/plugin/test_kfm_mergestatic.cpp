// test_kfm_mergestatic.cpp — headless verification of the KTemporalDiffOCL +
// KMergeStaticOCL AviSynth+ plugin (fourth host plugin of the OpenCL port)
// against CPU references computed in this exe.
//
// REFERENCE CHOICE (stated, per the task): the references are computed HERE,
// in this exe, from the actual input frame pixels, transcribing the repo's
// CPU-mirror/golden semantics (sim/kfm_mergestatic_ref.cpp +
// python/run_kfm_mergestatic.py — the golden-pinned authority, == the
// upstream cpu_compare_frames / cpu_merge_static twins in MergeStatic.cu)
// plus the upstream host pipelines:
//   KTemporalDiff: 5-frame window at clamp(n-2+i, 0, num_frames-1) (the
//     plugin's documented deviation-1 upper bound; upstream clamps to
//     num_frames which only works through AviSynth+ cache clamping), then
//     per plane dst = max(f0..f4) - min(f0..f4) — integer, exact twin.
//   KMergeStatic: flag = sttclip(n>>1), frame60 = child(n),
//     frame30 = clip30(n>>1), per plane
//     dst = (coef*v30 + (128-coef)*v60 + 64) >> 7 — integer, exact twin
//     (arithmetic shift on non-negative values; the flag sources here stay
//     in the documented [0,128] coefficient range).
// Both are pure integer ops — no float-parity caveats at all.
//
// The plugins hold NO cross-frame state (see the plugin header's state
// model): each output frame is a pure function of the fetched input frames,
// so the test fetches exactly those frames from the source clips and
// compares every pixel of every plane.  The cases exercise the temporal /
// clip-mapping dimensions:
//   - KTemporalDiff over 6 formats x random / constant / scene-change
//     sources, frames 0..3 (first-frame clamping at n<2, interior, tail
//     clamp on an 8-frame clip), constant-clip convergence
//     (max-min == 0 everywhere), plus a short 2-frame clip (every window
//     member clamps to one of the two frames).
//   - KMergeStatic over 6 formats x random / constant / flag-corner sources,
//     output frames 0..7 with 4-frame 30fps/flag clips (n>>1 mapping,
//     including the odd->even floor), coef corners 0 and 128 (output must
//     equal frame60 / frame30 exactly), plus the upstream
//     "Num frames don't match" rejection.
//
// Load path: the exe copies the plugin DLL into the AviSynth+ autoload
// directory (registry HKLM\Software\AviSynth plugins64), creates the script
// environment, verifies autoload via FunctionExists, then invokes
// KTemporalDiffOCL / KMergeStaticOCL (and the upstream KTemporalDiff /
// KMergeStatic aliases, plus real script strings via Eval) and compares
// EVERY pixel of every plane.
//
// Build: see build_host_mergestatic.bat.  Usage: test_kfm_mergestatic.exe
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
            // low-frequency structure + motion so temporal windows see spread
            if ((x / 4) & 1) v += maxv / 8;
            if ((y / 8) & 1) v += maxv / 6;
            v += (int)(mix32(salt ^ ((uint32_t)x * 31u + (uint32_t)y * 7u)) % 16u) - 8;
            row[x] = (PX)(v < 0 ? 0 : v > maxv ? maxv : v);
        }
    }
}

template <typename PX>
static void fillFlag(uint8_t* dstp, int pitch, int w, int h, int maxv, uint32_t salt) {
    // KMergeStatic coefficient plane: values in [0,128] (the kernel's
    // documented domain), with 0/128 corners exercised via the salt
    for (int y = 0; y < h; ++y) {
        PX* row = (PX*)(dstp + (size_t)y * pitch);
        for (int x = 0; x < w; ++x) {
            uint32_t r = mix32((uint32_t)x * 13u + (uint32_t)y * 509u + salt);
            int v = (int)(r % 129u);
            if (salt == 0xC0C0u && ((x / 8) & 1)) v = 0;     // keep-60fps corner
            if (salt == 0xC0C0u && ((y / 8) & 1)) v = 128;   // take-30fps corner
            row[x] = (PX)v;
        }
    }
}

enum SourceKind {
    SRC_RANDOM,       // independent content every frame
    SRC_CONSTANT,     // identical content every frame
    SRC_SCENECHANGE,  // identical within a scene, abrupt jump at frame 2
    SRC_FLAG,         // KMergeStatic coefficient plane in [0,128]
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
        case SRC_FLAG:        s = salt_ + (uint32_t)(n >> 1) * 7919u; break; // 30fps cadence
        default:              s = salt_ + (uint32_t)n * 7919u; break;
        }
        if (px == 1) {
            if (kind_ == SRC_FLAG) {
                fillFlag<uint8_t>(f->GetWritePtr(PLANAR_Y), f->GetPitch(PLANAR_Y), w, h, maxv, s);
                fillFlag<uint8_t>(f->GetWritePtr(PLANAR_U), f->GetPitch(PLANAR_U), wUV, hUV, maxv, s + 11u);
                fillFlag<uint8_t>(f->GetWritePtr(PLANAR_V), f->GetPitch(PLANAR_V), wUV, hUV, maxv, s + 23u);
            } else {
                fillPattern<uint8_t>(f->GetWritePtr(PLANAR_Y), f->GetPitch(PLANAR_Y), w, h, maxv, s);
                fillPattern<uint8_t>(f->GetWritePtr(PLANAR_U), f->GetPitch(PLANAR_U), wUV, hUV, maxv, s + 11u);
                fillPattern<uint8_t>(f->GetWritePtr(PLANAR_V), f->GetPitch(PLANAR_V), wUV, hUV, maxv, s + 23u);
            }
        } else {
            if (kind_ == SRC_FLAG) {
                fillFlag<uint16_t>(f->GetWritePtr(PLANAR_Y), f->GetPitch(PLANAR_Y), w, h, maxv, s);
                fillFlag<uint16_t>(f->GetWritePtr(PLANAR_U), f->GetPitch(PLANAR_U), wUV, hUV, maxv, s + 11u);
                fillFlag<uint16_t>(f->GetWritePtr(PLANAR_V), f->GetPitch(PLANAR_V), wUV, hUV, maxv, s + 23u);
            } else {
                fillPattern<uint16_t>(f->GetWritePtr(PLANAR_Y), f->GetPitch(PLANAR_Y), w, h, maxv, s);
                fillPattern<uint16_t>(f->GetWritePtr(PLANAR_U), f->GetPitch(PLANAR_U), wUV, hUV, maxv, s + 11u);
                fillPattern<uint16_t>(f->GetWritePtr(PLANAR_V), f->GetPitch(PLANAR_V), wUV, hUV, maxv, s + 23u);
            }
        }
        return f;
    }
    bool __stdcall GetParity(int) { return false; }
    void __stdcall GetAudio(void*, int64_t, int64_t, IScriptEnvironment*) {}
    int __stdcall SetCacheHints(int, int) { return 0; }
    const VideoInfo& __stdcall GetVideoInfo() { return vi_; }
};

// ---------------------------------------------------------------------------
// CPU references (golden semantics; see file header)
// ---------------------------------------------------------------------------
template <typename PX>
static void readPlane(std::vector<PX>& out, PVideoFrame f, int pl, int pw, int ph) {
    out.resize((size_t)pw * ph);
    const uint8_t* fp = f->GetReadPtr(pl);
    int pitch = f->GetPitch(pl);
    for (int y = 0; y < ph; ++y)
        memcpy(&out[(size_t)y * pw], fp + (size_t)y * pitch, (size_t)pw * sizeof(PX));
}

// kf_compare_frames twin over the 5 fetched frames (packed planes, pitch == w)
template <typename PX>
static void ref_compare_frames(std::vector<PX>& dst,
                               const std::vector<std::vector<PX>>& frames,
                               int width, int height) {
    const int pitch = width;
    dst.assign((size_t)pitch * height, (PX)0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t idx = (size_t)y * pitch + x;
            int mn = 1 << 30, mx = -(1 << 30);
            for (int i = 0; i < 5; ++i) {
                int v = (int)frames[i][idx];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            dst[idx] = (PX)(mx - mn);
        }
    }
}

// kf_merge_static twin: dst = (coef*v30 + (128-coef)*v60 + 64) >> 7
template <typename PX>
static void ref_merge_static(std::vector<PX>& dst, const std::vector<PX>& s60,
                             const std::vector<PX>& s30, const std::vector<PX>& flag,
                             int width, int height) {
    const int pitch = width;
    dst.assign((size_t)pitch * height, (PX)0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t idx = (size_t)y * pitch + x;
            int coef = (int)flag[idx];
            int v30 = (int)s30[idx];
            int v60 = (int)s60[idx];
            int tmp = (coef * v30 + (128 - coef) * v60 + 64) >> 7;
            dst[idx] = (PX)tmp;
        }
    }
}

// The window fetch needs an IScriptEnvironment; pass it through a file-scope
// pointer set by the driver (single-threaded test exe).
static IScriptEnvironment* g_env = nullptr;
static IScriptEnvironment* env_of(const VideoInfo&) { return g_env; }

// ---------------------------------------------------------------------------
// KTemporalDiff reference pipeline + full-plane comparison
// ---------------------------------------------------------------------------
template <typename PX>
static bool refAndCompareDiff(const char* tag, const VideoInfo& vi, PClip srcClip,
                              PVideoFrame outf, int n) {
    const int w = vi.width, h = vi.height;
    const int lx = vi.GetPlaneWidthSubsampling(PLANAR_U);
    const int ly = vi.GetPlaneHeightSubsampling(PLANAR_U);
    const int wUV = w >> lx, hUV = h >> ly;

    // fetch the clamped window from the source (plugin semantics)
    std::vector<std::vector<PX>> wY(5), wU(5), wV(5);
    for (int i = 0; i < 5; ++i) {
        const int fn = (std::max)(0, (std::min)(n - 2 + i, vi.num_frames - 1));
        PVideoFrame f = srcClip->GetFrame(fn, env_of(vi));
        readPlane(wY[i], f, PLANAR_Y, w, h);
        readPlane(wU[i], f, PLANAR_U, wUV, hUV);
        readPlane(wV[i], f, PLANAR_V, wUV, hUV);
    }

    std::vector<PX> rY, rU, rV;
    ref_compare_frames(rY, wY, w, h);
    ref_compare_frames(rU, wU, wUV, hUV);
    ref_compare_frames(rV, wV, wUV, hUV);

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
// KMergeStatic reference pipeline + full-plane comparison
// ---------------------------------------------------------------------------
template <typename PX>
static bool refAndCompareMerge(const char* tag, const VideoInfo& vi,
                               PClip clip60, PClip clip30, PClip sttclip,
                               PVideoFrame outf, int n) {
    const int w = vi.width, h = vi.height;
    const int lx = vi.GetPlaneWidthSubsampling(PLANAR_U);
    const int ly = vi.GetPlaneHeightSubsampling(PLANAR_U);
    const int wUV = w >> lx, hUV = h >> ly;
    const int n30 = n >> 1;

    PVideoFrame f60 = clip60->GetFrame(n, env_of(vi));
    PVideoFrame f30 = clip30->GetFrame(n30, env_of(vi));
    PVideoFrame ffl = sttclip->GetFrame(n30, env_of(vi));

    std::vector<PX> s60Y, s60U, s60V, s30Y, s30U, s30V, flY, flU, flV;
    readPlane(s60Y, f60, PLANAR_Y, w, h);
    readPlane(s60U, f60, PLANAR_U, wUV, hUV);
    readPlane(s60V, f60, PLANAR_V, wUV, hUV);
    readPlane(s30Y, f30, PLANAR_Y, w, h);
    readPlane(s30U, f30, PLANAR_U, wUV, hUV);
    readPlane(s30V, f30, PLANAR_V, wUV, hUV);
    readPlane(flY, ffl, PLANAR_Y, w, h);
    readPlane(flU, ffl, PLANAR_U, wUV, hUV);
    readPlane(flV, ffl, PLANAR_V, wUV, hUV);

    std::vector<PX> rY, rU, rV;
    ref_merge_static(rY, s60Y, s30Y, flY, w, h);
    ref_merge_static(rU, s60U, s30U, flU, wUV, hUV);
    ref_merge_static(rV, s60V, s30V, flV, wUV, hUV);

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

static PClip makeSource(const VideoInfo& cvi, uint32_t salt, SourceKind kind) {
    return PClip(new RandSource(cvi, salt, kind));
}

// --- KTemporalDiff cases ---
static void runDiffCase(IScriptEnvironment* env, const char* tag, PClip clip,
                        const VideoInfo& vi, int bits, int nframes) {
    AVSValue args[1] = { AVSValue(clip) };
    AVSValue ret;
    try {
        ret = env->Invoke("KTemporalDiffOCL", AVSValue(args, 1));
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
    for (int n = 0; n < nframes && ok; ++n) {
        PVideoFrame outf;
        try {
            outf = filtered->GetFrame(n, env);
        } catch (const AvisynthError& e) {
            printf("  GetFrame threw: %s\n", e.msg);
            ++g_failures;
            return;
        }
        ++g_frame_checks;
        ok = (bits == 8)
            ? refAndCompareDiff<uint8_t>(tag, vi, clip, outf, n)
            : refAndCompareDiff<uint16_t>(tag, vi, clip, outf, n);
    }
    ++g_cases;
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", tag);
    if (!ok) ++g_failures;
}

// --- KMergeStatic cases ---
static void runMergeCase(IScriptEnvironment* env, const char* tag,
                         PClip c60, PClip c30, PClip flag, const char* func,
                         const VideoInfo& vi, int bits, int nframes) {
    AVSValue args[3] = { AVSValue(c60), AVSValue(c30), AVSValue(flag) };
    AVSValue ret;
    try {
        ret = env->Invoke(func, AVSValue(args, 3));
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
    for (int n = 0; n < nframes && ok; ++n) {
        PVideoFrame outf;
        try {
            outf = filtered->GetFrame(n, env);
        } catch (const AvisynthError& e) {
            printf("  GetFrame threw: %s\n", e.msg);
            ++g_failures;
            return;
        }
        ++g_frame_checks;
        ok = (bits == 8)
            ? refAndCompareMerge<uint8_t>(tag, vi, c60, c30, flag, outf, n)
            : refAndCompareMerge<uint16_t>(tag, vi, c60, c30, flag, outf, n);
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
        std::string dst = pluginsDir + "\\kfm_mergestatic_avs.dll";
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
    if (!env->FunctionExists("KTemporalDiffOCL")) {
        printf("[FAIL] autoload: KTemporalDiffOCL not registered after env creation\n");
        return 1;
    }
    if (!env->FunctionExists("KTemporalDiff")) {
        printf("[FAIL] autoload: KTemporalDiff alias not registered\n");
        return 1;
    }
    if (!env->FunctionExists("KMergeStaticOCL")) {
        printf("[FAIL] autoload: KMergeStaticOCL not registered after env creation\n");
        return 1;
    }
    if (!env->FunctionExists("KMergeStatic")) {
        printf("[FAIL] autoload: KMergeStatic alias not registered\n");
        return 1;
    }
    printf("autoload OK (KTemporalDiffOCL + KTemporalDiff + KMergeStaticOCL + KMergeStatic registered)\n");

    // ---- deterministic-source cases ----
    const int W = 64, H = 48;
    for (const FormatSpec& fs : g_formats) {
        char srcScript[160];
        sprintf_s(srcScript,
                  "BlankClip(width=%d, height=%d, pixel_type=\"%s\", length=8, color=$4080C0)",
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
        // 30fps-cadence clip (half the frames) for the KMergeStatic cases
        VideoInfo cvi30 = cvi;
        cvi30.num_frames = cvi.num_frames / 2;

        // ==== KTemporalDiff ====
        // random per-frame content: covers abrupt inter-frame changes + all
        // clamp regimes (first frames n<2, interior, tail)
        {
            PClip clip = makeSource(cvi, 0xD17F00u ^ (uint32_t)fs.bits, SRC_RANDOM);
            runDiffCase(env, "diff random", clip, cvi, fs.bits, 4);
        }
        // constant clip: every 5-frame window holds one value -> max-min == 0
        // everywhere (temporal-spread convergence check)
        {
            PClip clip = makeSource(cvi, 0x51DEu ^ (uint32_t)fs.bits, SRC_CONSTANT);
            runDiffCase(env, "diff constant (converged to 0)", clip, cvi, fs.bits, 4);
        }
        // scene-change clip: windows spanning the cut at frame 2
        {
            PClip clip = makeSource(cvi, 0xACEu ^ (uint32_t)fs.bits, SRC_SCENECHANGE);
            runDiffCase(env, "diff scene-change", clip, cvi, fs.bits, 4);
        }
        // short 2-frame clip: every window member clamps to frame 0 or 1
        {
            AVSValue shortV;
            try {
                char s2[160];
                sprintf_s(s2, "BlankClip(width=%d, height=%d, pixel_type=\"%s\", length=2, color=$4080C0)",
                          W, H, fs.pixel_type);
                shortV = env->Invoke("Eval", AVSValue(s2));
            } catch (const AvisynthError& e) {
                printf("[FAIL] short eval %s: %s\n", fs.pixel_type, e.msg);
                ++g_failures;
                continue;
            }
            PClip clip = makeSource(shortV.AsClip()->GetVideoInfo(),
                                    0x5EEDu ^ (uint32_t)fs.bits, SRC_RANDOM);
            runDiffCase(env, "diff 2-frame clip (heavy clamp)", clip,
                        shortV.AsClip()->GetVideoInfo(), fs.bits, 2);
        }
        // upstream alias (KTemporalDiff by its upstream name)
        {
            PClip clip = makeSource(cvi, 0xA1A5u ^ (uint32_t)fs.bits, SRC_RANDOM);
            AVSValue args[1] = { AVSValue(clip) };
            try {
                AVSValue ret = env->Invoke("KTemporalDiff", AVSValue(args, 1));
                if (!ret.IsClip()) throw AvisynthError("no clip");
                PVideoFrame outf = ret.AsClip()->GetFrame(2, env);
                ++g_frame_checks;
                bool ok = (fs.bits == 8)
                    ? refAndCompareDiff<uint8_t>("diff alias", cvi, clip, outf, 2)
                    : refAndCompareDiff<uint16_t>("diff alias", cvi, clip, outf, 2);
                printf("[%s] %s KTemporalDiff alias invocation\n", ok ? "PASS" : "FAIL", fs.pixel_type);
                if (!ok) ++g_failures;
            } catch (const AvisynthError& e) {
                printf("[FAIL] %s diff alias: %s\n", fs.pixel_type, e.msg);
                ++g_failures;
            }
            ++g_cases;
        }

        // ==== KMergeStatic ====
        // 8-frame 60fps clip + 4-frame 30fps + 4-frame flag clip (30fps
        // cadence): exercises the n>>1 mapping incl. the odd->even floor
        {
            PClip c60 = makeSource(cvi, 0xBEEF00u ^ (uint32_t)fs.bits, SRC_RANDOM);
            PClip c30 = makeSource(cvi30, 0xCAFE00u ^ (uint32_t)fs.bits, SRC_RANDOM);
            PClip fl  = makeSource(cvi30, 0xF1A700u ^ (uint32_t)fs.bits, SRC_FLAG);
            runMergeCase(env, "merge random", c60, c30, fl, "KMergeStaticOCL",
                         cvi, fs.bits, 8);
        }
        // constant 60/30fps + flag: output = the exact blend of constants
        {
            PClip c60 = makeSource(cvi, 0x600D00u ^ (uint32_t)fs.bits, SRC_CONSTANT);
            PClip c30 = makeSource(cvi30, 0x30CD00u ^ (uint32_t)fs.bits, SRC_CONSTANT);
            PClip fl  = makeSource(cvi30, 0xF1A700u ^ (uint32_t)fs.bits, SRC_FLAG);
            runMergeCase(env, "merge constant", c60, c30, fl, "KMergeStaticOCL",
                         cvi, fs.bits, 4);
        }
        // flag corners: coef 0 => output == frame60 exactly; coef 128 =>
        // output == frame30 exactly (per 8x8 blocks of both)
        {
            PClip c60 = makeSource(cvi, 0xD0D000u ^ (uint32_t)fs.bits, SRC_RANDOM);
            PClip c30 = makeSource(cvi30, 0xD1D100u ^ (uint32_t)fs.bits, SRC_RANDOM);
            PClip fl  = makeSource(cvi30, 0xC0C0u ^ (uint32_t)fs.bits, SRC_FLAG);
            runMergeCase(env, "merge flag corners 0/128", c60, c30, fl, "KMergeStaticOCL",
                         cvi, fs.bits, 4);
        }
        // scene-change 60fps clip
        {
            PClip c60 = makeSource(cvi, 0x5C5C00u ^ (uint32_t)fs.bits, SRC_SCENECHANGE);
            PClip c30 = makeSource(cvi30, 0x3E3E00u ^ (uint32_t)fs.bits, SRC_RANDOM);
            PClip fl  = makeSource(cvi30, 0xF1A700u ^ (uint32_t)fs.bits, SRC_FLAG);
            runMergeCase(env, "merge scene-change", c60, c30, fl, "KMergeStaticOCL",
                         cvi, fs.bits, 4);
        }
        // upstream alias (KMergeStatic by its upstream name)
        {
            PClip c60 = makeSource(cvi, 0xA1B200u ^ (uint32_t)fs.bits, SRC_RANDOM);
            PClip c30 = makeSource(cvi30, 0xA2B300u ^ (uint32_t)fs.bits, SRC_RANDOM);
            PClip fl  = makeSource(cvi30, 0xA3B400u ^ (uint32_t)fs.bits, SRC_FLAG);
            runMergeCase(env, "merge alias", c60, c30, fl, "KMergeStatic",
                         cvi, fs.bits, 4);
        }
    }

    // ---- real script strings (real Source-derived clips) ----
    try {
        // KTemporalDiff via Eval (constant ColorBars -> diff 0 everywhere,
        // checked against the pixel-level reference anyway)
        const char* script =
            "KTemporalDiffOCL(ColorBars(pixel_type=\"YV12\", width=64, height=48).Trim(0, 7))";
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
            ok = refAndCompareDiff<uint8_t>("script-string", vi, srcClip, outf, n);
        }
        ++g_cases;
        printf("[%s] script-string KTemporalDiffOCL ColorBars YV12 (4 frames)\n",
               ok ? "PASS" : "FAIL");
        if (!ok) ++g_failures;
    } catch (const AvisynthError& e) {
        printf("[FAIL] script-string diff case: %s\n", e.msg);
        ++g_failures;
    }

    try {
        // KMergeStatic via Eval: constant BlankClips (black 60fps / white
        // 30fps / mid-gray flag) — the flag planes are the BlankClip's own
        // constant values; the pixel-level reference handles whatever they are.
        // Newline-separated statements (Eval rejects ';' as a separator).
        const char* script =
            "a = BlankClip(pixel_type=\"YV12\", width=64, height=48, length=8, color=$000000)\n"
            "b = BlankClip(pixel_type=\"YV12\", width=64, height=48, length=4, color=$FFFFFF)\n"
            "c = BlankClip(pixel_type=\"YV12\", width=64, height=48, length=8, color=$404040)\n"
            "KMergeStatic(a, b, c)";
        AVSValue ret = env->Invoke("Eval", AVSValue(script));
        if (!ret.IsClip()) throw AvisynthError("Eval did not return a clip");
        AVSValue aV = env->Invoke("Eval",
            AVSValue("BlankClip(pixel_type=\"YV12\", width=64, height=48, length=8, color=$000000)"));
        AVSValue bV = env->Invoke("Eval",
            AVSValue("BlankClip(pixel_type=\"YV12\", width=64, height=48, length=4, color=$FFFFFF)"));
        AVSValue cV = env->Invoke("Eval",
            AVSValue("BlankClip(pixel_type=\"YV12\", width=64, height=48, length=8, color=$404040)"));
        PClip c60 = aV.AsClip();
        PClip c30 = bV.AsClip();
        PClip fl = cV.AsClip();
        PClip filtered = ret.AsClip();
        const VideoInfo& vi = filtered->GetVideoInfo();
        bool ok = true;
        for (int n = 0; n < 4 && ok; ++n) {
            PVideoFrame outf = filtered->GetFrame(n, env);
            ++g_frame_checks;
            ok = refAndCompareMerge<uint8_t>("script-string-merge", vi, c60, c30, fl, outf, n);
        }
        ++g_cases;
        printf("[%s] script-string KMergeStatic BlankClips YV12 (4 frames)\n",
               ok ? "PASS" : "FAIL");
        if (!ok) ++g_failures;
    } catch (const AvisynthError& e) {
        printf("[FAIL] script-string merge case: %s\n", e.msg);
        ++g_failures;
    }

    // ---- KMergeStatic: upstream "Num frames don't match" validation ----
    {
        try {
            AVSValue aV = env->Invoke("Eval",
                AVSValue("BlankClip(pixel_type=\"YV12\", width=64, height=48, length=8)"));
            AVSValue bV = env->Invoke("Eval",
                AVSValue("BlankClip(pixel_type=\"YV12\", width=64, height=48, length=3)"));
            AVSValue cV = env->Invoke("Eval",
                AVSValue("BlankClip(pixel_type=\"YV12\", width=64, height=48, length=8)"));
            AVSValue args[3] = { aV, bV, cV };
            try {
                env->Invoke("KMergeStaticOCL", AVSValue(args, 3));
                printf("[FAIL] mismatched frame counts accepted (expected throw)\n");
                ++g_failures;
            } catch (const AvisynthError&) {
                printf("[PASS] 8 vs 3*2 frames rejected (Num frames don't match)\n");
            }
        } catch (const AvisynthError& e) {
            printf("[FAIL] frame-count-validation setup: %s\n", e.msg);
            ++g_failures;
        }
        ++g_cases;
    }

    printf("\n=== KTemporalDiffOCL + KMergeStaticOCL: %d cases (%d frame checks), %d failures ===\n",
           g_cases, g_frame_checks, g_failures);

    env->DeleteScriptEnvironment();
    // NOTE: deliberately not FreeLibrary — AviSynth+ keeps internal threads
    // referencing the DLL at shutdown.
    return g_failures ? 1 : 0;
}
