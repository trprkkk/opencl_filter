// test_kfm_edgelevel.cpp — headless verification of the KEdgeLevelOCL
// AviSynth+ plugin (first host plugin of the OpenCL port) against a CPU
// reference computed in this exe.
//
// REFERENCE CHOICE (stated, per the task): the reference is computed HERE,
// in this exe, from the actual input frame pixels, transcribing the repo's
// CPU-mirror/golden semantics (sim/kfm_edgelevel_ref.cpp +
// python/run_kfm_edgelevel.py — the golden-pinned authority) for all four
// kernels, plus the upstream KEdgeLevel::GetFrameT host pipeline
// (KDeband.cu).  The repair kernel is dispatched over the INTERIOR only and
// its destination planes' border rings are host-defined (ring = pass-0 source
// ring; the output Y ring passes through from the source frame) — see the
// plugin header's deviation 2; the repair INTERIOR is pinned by the repo
// goldens either way.  All float ops are IEEE float32 per-op (MSVC x64 SSE2,
// /fp:precise default; no FMA contraction), matching the mirror's
// -ffp-contract=off and the device build (pragma + correctly-rounded divide).
//
// Load path: the exe copies the plugin DLL into the AviSynth+ autoload
// directory (registry HKLM\Software\AviSynth plugins64), creates the script
// environment, verifies autoload via FunctionExists, then invokes
// KEdgeLevelOCL (and the upstream KEdgeLevel alias, plus a real script
// string via Eval) over deterministic sources and compares EVERY pixel of
// every plane.
//
// Build: see build_host_edgelevel.bat.  Usage: test_kfm_edgelevel.exe
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
// deterministic source clip (scripted pixel content)
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
            // structured edges so the window/threshold paths are exercised
            if ((x / 4) & 1) v += maxv / 8;
            if ((y / 8) & 1) v += maxv / 6;
            if (x < 2 || y < 2 || x >= w - 2 || y >= h - 2) v = maxv - ((x + y) % 7);
            row[x] = (PX)(v > maxv ? maxv : v);
        }
    }
}

class RandSource : public IClip {
    VideoInfo vi_;
    uint32_t salt_;
public:
    RandSource(VideoInfo vi, uint32_t salt) : vi_(vi), salt_(salt) {}
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) {
        PVideoFrame f = env->NewVideoFrame(vi_);
        const int maxv = (1 << vi_.BitsPerComponent()) - 1;
        const int w = vi_.width, h = vi_.height;
        const int wUV = w >> vi_.GetPlaneWidthSubsampling(PLANAR_U);
        const int hUV = h >> vi_.GetPlaneHeightSubsampling(PLANAR_U);
        const int px = vi_.ComponentSize();
        const uint32_t s = salt_ + (uint32_t)n * 7919u;
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
static inline int el_scale(int c, int maxv) {
    return (int)((((float)c) / 255.0f) * (float)maxv);
}

template <typename PX>
static void ref_edgelevel(int width, int height, int pitch, int maxv, int str, int strUV,
                          int thrs, int check, int selective, int uv,
                          const std::vector<PX>& src, const std::vector<PX>& u,
                          const std::vector<PX>& v, std::vector<PX>& dY, std::vector<PX>& dU,
                          std::vector<PX>& dV) {
    const int S = selective ? 1 : 0;
    dY.assign((size_t)pitch * height, (PX)0);
    if (uv) {
        dU.assign((size_t)pitch * height, (PX)0);
        dV.assign((size_t)pitch * height, (PX)0);
    }
    for (int yy = 0; yy < height; ++yy) {
        for (int xx = 0; xx < width; ++xx) {
            const int off = yy * pitch + xx;
            if (yy <= (1 + S) || yy >= height - (2 + S) ||
                xx <= (1 + S) || xx >= width - (2 + S)) {
                dY[off] = check ? (PX)el_scale(16, maxv) : src[off];
                if (uv) { dU[off] = u[off]; dV[off] = v[off]; }
                continue;
            }
            const PX* sy = &src[off];
            int hmax, hmin, vmax, vmin;
            int hdiffmax = 0, vdiffmax = 0;
            int hprev = hmax = hmin = (int)sy[-(2 + S)];
            int vprev = vmax = vmin = (int)sy[-(2 + S) * pitch];
            for (int i = -(1 + S); i < (3 + S); ++i) {
                int hcur = (int)sy[i];
                int vcur = (int)sy[i * pitch];
                if (hcur > hmax) hmax = hcur;
                if (hcur < hmin) hmin = hcur;
                if (vcur > vmax) vmax = vcur;
                if (vcur < vmin) vmin = vcur;
                if (selective) {
                    int dh = hcur - hprev; if (dh < 0) dh = -dh;
                    int dv = vcur - vprev; if (dv < 0) dv = -dv;
                    if (dh > hdiffmax) hdiffmax = dh;
                    if (dv > vdiffmax) vdiffmax = dv;
                }
                hprev = hcur;
                vprev = vcur;
            }
            if (hmax - hmin < vmax - vmin) {
                hmax = vmax;
                hmin = vmin;
            }
            if (vdiffmax > hdiffmax) hdiffmax = vdiffmax;

            float factor = 1.0f;
            if (selective) {
                float rdiff = (float)hdiffmax / (float)(hmax - hmin);
                float a = (0.55f - rdiff) * 10.0f;  if (a < 0.0f) a = 0.0f; else if (a > 1.0f) a = 1.0f;
                float b = (0.35f - rdiff) * 10.0f;  if (b < 0.0f) b = 0.0f; else if (b > 1.0f) b = 1.0f;
                factor = a - b;
            }

            int srcvY = (int)sy[0];
            int dstvY = srcvY, dstvU = 0, dstvV = 0;

            if (check) {
                if (hmax - hmin > thrs && factor > 0.0f) {
                    int avgY = (hmax + hmin) >> 1;
                    if (srcvY > avgY)
                        dstvY = (factor == 1.0f) ? el_scale(50, maxv) : el_scale(120, maxv);
                    else
                        dstvY = (factor == 1.0f) ? el_scale(240, maxv) : el_scale(180, maxv);
                } else {
                    dstvY = el_scale(16, maxv);
                }
                dstvU = (int)u[off];
                dstvV = (int)v[off];
            } else {
                if (hmax - hmin > thrs && factor > 0.0f) {
                    float factorY = ((float)str * factor) * 0.0625f;
                    int avgY = (hmax + hmin) >> 1;
                    int val = srcvY + (int)(((float)(srcvY - avgY)) * factorY);
                    if (val < hmin) val = hmin; else if (val > hmax) val = hmax;
                    if (val < 0) val = 0; else if (val > maxv) val = maxv;
                    dstvY = val;

                    if (uv) {
                        float factorUV = (float)strUV * 0.0625f;
                        // U window
                        int Uhmax, Uhmin, Uvmax, Uvmin;
                        Uhmax = Uhmin = (int)u[off - (2 + S)];
                        Uvmax = Uvmin = (int)u[off - (2 + S) * pitch];
                        for (int i = -(1 + S); i < (3 + S); ++i) {
                            int hc = (int)u[off + i];
                            int vc = (int)u[off + i * pitch];
                            if (hc > Uhmax) Uhmax = hc;
                            if (hc < Uhmin) Uhmin = hc;
                            if (vc > Uvmax) Uvmax = vc;
                            if (vc < Uvmin) Uvmin = vc;
                        }
                        if (Uhmax - Uhmin < Uvmax - Uvmin) { Uhmax = Uvmax; Uhmin = Uvmin; }
                        int avgU = (Uhmax + Uhmin) >> 1;
                        int svU = (int)u[off];
                        int uout = svU + (int)(((float)(svU - avgU)) * factorUV);
                        if (uout < Uhmin) uout = Uhmin; else if (uout > Uhmax) uout = Uhmax;
                        if (uout < 0) uout = 0; else if (uout > maxv) uout = maxv;
                        dstvU = uout;
                        // V window
                        int Vhmax, Vhmin, Vvmax, Vvmin;
                        Vhmax = Vhmin = (int)v[off - (2 + S)];
                        Vvmax = Vvmin = (int)v[off - (2 + S) * pitch];
                        for (int i = -(1 + S); i < (3 + S); ++i) {
                            int hc = (int)v[off + i];
                            int vc = (int)v[off + i * pitch];
                            if (hc > Vhmax) Vhmax = hc;
                            if (hc < Vhmin) Vhmin = hc;
                            if (vc > Vvmax) Vvmax = vc;
                            if (vc < Vvmin) Vvmin = vc;
                        }
                        if (Vhmax - Vhmin < Vvmax - Vvmin) { Vhmax = Vvmax; Vhmin = Vvmin; }
                        int avgV = (Vhmax + Vhmin) >> 1;
                        int svV = (int)v[off];
                        int vout = svV + (int)(((float)(svV - avgV)) * factorUV);
                        if (vout < Vhmin) vout = Vhmin; else if (vout > Vhmax) vout = Vhmax;
                        if (vout < 0) vout = 0; else if (vout > maxv) vout = maxv;
                        dstvV = vout;
                    }
                } else {
                    dstvY = srcvY;
                    dstvU = (int)u[off];
                    dstvV = (int)v[off];
                }
            }

            dY[off] = (PX)dstvY;
            if (uv) {
                dU[off] = (PX)dstvU;
                dV[off] = (PX)dstvV;
            }
        }
    }
}

template <typename PX>
static void ref_repair(std::vector<PX>& dst, const std::vector<PX>& el,
                       const std::vector<PX>& src, int width, int height, int pitch, int N) {
    // Interior only (plugin deviation 2): the kernel is dispatched over
    // (1,1)..(w-2,h-2); every 3x3 neighbour read stays in bounds of the plain
    // plane image.  dst's border ring is untouched (host-patched to the
    // pass-0 source ring before the loop).
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            const int off = y * pitch + x;
            int srcv = (int)src[off];
            int elv = (int)el[off];
            int dstv = srcv;
            if (elv != srcv) {
                int a[8];
                a[0] = (int)src[(x - 1) + (y - 1) * pitch];
                a[1] = (int)src[(x    ) + (y - 1) * pitch];
                a[2] = (int)src[(x + 1) + (y - 1) * pitch];
                a[3] = (int)src[(x - 1) + (y    ) * pitch];
                a[4] = (int)src[(x + 1) + (y    ) * pitch];
                a[5] = (int)src[(x - 1) + (y + 1) * pitch];
                a[6] = (int)src[(x    ) + (y + 1) * pitch];
                a[7] = (int)src[(x + 1) + (y + 1) * pitch];
                std::sort(a, a + 8);
                int lo = srcv < a[N - 1] ? srcv : a[N - 1];
                int hi = srcv > a[8 - N] ? srcv : a[8 - N];
                if (elv < lo) dstv = lo; else if (elv > hi) dstv = hi; else dstv = elv;
            }
            dst[off] = (PX)dstv;
        }
    }
}

// border ring of a full-plane image (pitch == w)
template <typename PX>
static void ref_patch_ring(std::vector<PX>& dst, const std::vector<PX>& src, int w, int h) {
    for (int x = 0; x < w; ++x) {
        dst[x] = src[x];
        dst[x + (size_t)(h - 1) * w] = src[x + (size_t)(h - 1) * w];
    }
    for (int y = 1; y < h - 1; ++y) {
        dst[(size_t)y * w] = src[(size_t)y * w];
        dst[(size_t)y * w + w - 1] = src[(size_t)y * w + w - 1];
    }
}

template <typename PX>
static void ref_to444(const std::vector<PX>& src, int width, int height, int sp, int lx, int ly,
                      std::vector<PX>& dst) {
    const int BW = 1 << lx, BH = 1 << ly;
    const int dp = BW * width;
    dst.assign((size_t)dp * (BH * height), (PX)0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int v00 = (int)src[x + y * sp];
            dst[BW * x + 0 + (BH * y + 0) * dp] = (PX)v00;
            if (lx) {
                int v10 = (x + 1 < width) ? (int)src[(x + 1) + y * sp] : v00;
                dst[BW * x + 1 + (BH * y + 0) * dp] = (PX)((v00 + v10 + 1) >> 1);
                if (ly) {
                    int v01, v11;
                    if (y + 1 < height) {
                        v01 = (int)src[x + (y + 1) * sp];
                        v11 = (x + 1 < width) ? (int)src[(x + 1) + (y + 1) * sp] : v01;
                    } else {
                        v01 = v00;
                        v11 = (x + 1 < width) ? v10 : v00;
                    }
                    dst[BW * x + 0 + (BH * y + 1) * dp] = (PX)((v00 + v01 + 1) >> 1);
                    dst[BW * x + 1 + (BH * y + 1) * dp] = (PX)((v00 + v10 + v01 + v11 + 2) >> 2);
                }
            }
        }
    }
}

template <typename PX>
static void ref_from444(const std::vector<PX>& src, int width, int height, int sp, int lx, int ly,
                        std::vector<PX>& dst, int dp) {
    const int BW = 1 << lx, BH = 1 << ly;
    dst.assign((size_t)dp * height, (PX)0);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            dst[x + y * dp] = src[BW * x + BH * y * sp];
}

// ---------------------------------------------------------------------------
// case pipeline reference (upstream GetFrameT semantics)
// ---------------------------------------------------------------------------
struct Params {
    int str; float thrs; int repair; int struv; int show;
};

template <typename PX>
static bool refAndCompare(const char* tag, const VideoInfo& vi, PVideoFrame srcf,
                          PVideoFrame outf, const Params& p) {
    const int w = vi.width, h = vi.height;
    const int lx = vi.GetPlaneWidthSubsampling(PLANAR_U);
    const int ly = vi.GetPlaneHeightSubsampling(PLANAR_U);
    const int wUV = w >> lx, hUV = h >> ly;
    const int bits = vi.BitsPerComponent();
    const int maxv = (1 << bits) - 1;
    const bool uv = p.struv > 0;

    // read actual input planes
    std::vector<PX> sY((size_t)w * h), sU((size_t)wUV * hUV), sV((size_t)wUV * hUV);
    {
        const uint8_t* fp = srcf->GetReadPtr(PLANAR_Y);
        int pitch = srcf->GetPitch(PLANAR_Y);
        for (int y = 0; y < h; ++y) memcpy(&sY[(size_t)y * w], fp + (size_t)y * pitch, (size_t)w * sizeof(PX));
        fp = srcf->GetReadPtr(PLANAR_U);
        pitch = srcf->GetPitch(PLANAR_U);
        for (int y = 0; y < hUV; ++y) memcpy(&sU[(size_t)y * wUV], fp + (size_t)y * pitch, (size_t)wUV * sizeof(PX));
        fp = srcf->GetReadPtr(PLANAR_V);
        pitch = srcf->GetPitch(PLANAR_V);
        for (int y = 0; y < hUV; ++y) memcpy(&sV[(size_t)y * wUV], fp + (size_t)y * pitch, (size_t)wUV * sizeof(PX));
    }

    const int thr = (int)((float)p.thrs * (float)(1 << (bits - 8)) + 0.5f);

    // uv sources at 444 dims (upstream GetUVFrame; chroma-dims deviation)
    std::vector<PX> sU4, sV4;
    if (uv) {
        if (lx > 0) {
            ref_to444(sU, wUV, hUV, wUV, lx, ly, sU4);
            ref_to444(sV, wUV, hUV, wUV, lx, ly, sV4);
        } else {
            sU4 = sU;
            sV4 = sV;
        }
    } else {
        sU4.assign((size_t)w * h, (PX)0);
        sV4.assign((size_t)w * h, (PX)0);
    }

    // strengthening passes
    const int numel = (std::max)((p.repair + 1) / 2, 1);
    std::vector<PX> dY((size_t)w * h), dU((size_t)w * h), dV((size_t)w * h);
    std::vector<PX> tY, tU, tV;
    std::vector<PX> eY, eU, eV;  // el
    const std::vector<PX> *elsY = &sY, *elsU = &sU4, *elsV = &sV4;
    for (int i = 0; i < numel; ++i) {
        const int idx = ((p.show && (i + 1 == numel)) ? 4 : 0) + (p.repair ? 2 : 0) + (uv ? 1 : 0);
        if (i > 0) {
            std::swap(dY, tY); std::swap(dU, tU); std::swap(dV, tV);
            elsY = &tY; elsU = &tU; elsV = &tV;
        }
        ref_edgelevel(w, h, w, maxv, p.str, p.struv, thr, (idx >> 2) & 1, (idx >> 1) & 1, idx & 1,
                      *elsY, *elsU, *elsV, dY, dU, dV);
    }

    std::vector<PX> outU, outV;
    if (p.show) {
        outU.assign((size_t)wUV * hUV, (PX)(1 << (bits - 1)));
        outV = outU;
    } else {
        if (p.repair > 0) {
            // dst <-> el
            eY = std::move(dY); eU = std::move(dU); eV = std::move(dV);
            dY.assign((size_t)w * h, (PX)0); dU.assign((size_t)w * h, (PX)0); dV.assign((size_t)w * h, (PX)0);
            // border policy (deviation 2): every repair dst plane carries the
            // pass-0 repair source's ring before the loop (t* arrays only
            // exist as full planes when a second strengthening pass ran)
            if (tY.size() != (size_t)w * h) tY.assign((size_t)w * h, (PX)0);
            ref_patch_ring(dY, sY, w, h);
            ref_patch_ring(tY, sY, w, h);
            if (uv) {
                if (tU.size() != (size_t)w * h) tU.assign((size_t)w * h, (PX)0);
                if (tV.size() != (size_t)w * h) tV.assign((size_t)w * h, (PX)0);
                ref_patch_ring(dU, sU4, w, h);
                ref_patch_ring(tU, sU4, w, h);
                ref_patch_ring(dV, sV4, w, h);
                ref_patch_ring(tV, sV4, w, h);
            }
            for (int i = 0; i < p.repair; ++i) {
                const std::vector<PX> *rsY = &sY, *rsU = &sU4, *rsV = &sV4;
                if (i > 0) {
                    std::swap(dY, tY); std::swap(dU, tU); std::swap(dV, tV);
                    rsY = &tY; rsU = &tU; rsV = &tV;
                }
                ref_repair(dY, eY, *rsY, w, h, w, 3);
                if (uv) {
                    ref_repair(dU, eU, *rsU, w, h, w, 3);
                    ref_repair(dV, eV, *rsV, w, h, w, 3);
                }
            }
        }
        if (uv) {
            if (lx > 0) {
                ref_from444(dU, wUV, hUV, w, lx, ly, outU, wUV);
                ref_from444(dV, wUV, hUV, w, lx, ly, outV, wUV);
            } else {
                outU = dU;
                outV = dV;
            }
        } else {
            outU = sU;
            outV = sV;
        }
    }

    // ---- compare every pixel of every plane ----
    bool ok = true;
    auto cmp = [&](const char* plane, const std::vector<PX>& exp, PVideoFrame f, int pl,
                   int pw, int ph) {
        const uint8_t* fp = f->GetReadPtr(pl);
        int pitch = f->GetPitch(pl);
        for (int y = 0; y < ph && ok; ++y) {
            for (int x = 0; x < pw; ++x) {
                PX got;
                memcpy(&got, fp + (size_t)y * pitch + (size_t)x * sizeof(PX), sizeof(PX));
                if ((int)got != (int)exp[(size_t)y * pw + x]) {
                    printf("  MISMATCH %s frame %s: px(%d,%d) got=%d exp=%d\n",
                           tag, plane, x, y, (int)got, (int)exp[(size_t)y * pw + x]);
                    ok = false;
                    break;
                }
            }
        }
    };
    cmp("Y", dY, outf, PLANAR_Y, w, h);
    cmp("U", outU, outf, PLANAR_U, wUV, hUV);
    cmp("V", outV, outf, PLANAR_V, wUV, hUV);
    return ok;
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------
struct FormatSpec {
    const char* pixel_type;  // for script sources
    int bits;
    int logUVx, logUVy;
};

static const FormatSpec g_formats[] = {
    { "YV12",      8, 1, 1 },
    { "YV16",      8, 1, 0 },
    { "YV24",      8, 0, 0 },
    { "YUV420P16", 16, 1, 1 },
    { "YUV422P16", 16, 1, 0 },
    { "YUV444P16", 16, 0, 0 },
};

static const Params g_params[] = {
    { 10, 25.0f, 0, 0, 0 },   // upstream defaults
    { 30, 10.0f, 0, 0, 0 },
    { 16, 8.0f, 1, 0, 0 },
    { 16, 8.0f, 1, 16, 0 },
    { 16, 8.0f, 2, 32, 0 },
    { 55, 20.0f, 3, 20, 0 },
    { 5, 2.5f, 0, 8, 0 },
    { 60, 200.0f, 0, 8, 0 },
    { 0, 25.0f, 0, 0, 0 },
    { 10, 25.0f, 0, 0, 1 },   // show
    { 10, 25.0f, 1, 0, 1 },   // show + selective/check
    { 10, 25.0f, 1, 24, 1 },  // show + uv
};

static void runCase(IScriptEnvironment* env, const char* tag, PClip clip, const Params& p,
                    const VideoInfo& vi, int bits) {
    AVSValue args[6] = { AVSValue(clip), p.str, p.thrs, p.repair, p.struv, (bool)(p.show != 0) };
    AVSValue ret;
    try {
        ret = env->Invoke("KEdgeLevelOCL", AVSValue(args, 6));
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
        PVideoFrame srcf, outf;
        try {
            srcf = clip->GetFrame(n, env);
            outf = filtered->GetFrame(n, env);
        } catch (const AvisynthError& e) {
            printf("  GetFrame threw: %s\n", e.msg);
            ++g_failures;
            return;
        }
        ++g_frame_checks;
        if (bits == 8)
            ok = refAndCompare<uint8_t>(tag, vi, srcf, outf, p);
        else
            ok = refAndCompare<uint16_t>(tag, vi, srcf, outf, p);
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
        std::string dst = pluginsDir + "\\kfm_edgelevel_avs.dll";
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

    // ---- autoload validation ----
    if (!env->FunctionExists("KEdgeLevelOCL")) {
        printf("[FAIL] autoload: KEdgeLevelOCL not registered after env creation\n");
        return 1;
    }
    if (!env->FunctionExists("KEdgeLevel")) {
        printf("[FAIL] autoload: KEdgeLevel alias not registered\n");
        return 1;
    }
    printf("autoload OK (KEdgeLevelOCL + KEdgeLevel alias registered)\n");

    // ---- deterministic-source cases ----
    const int W = 64, H = 48;
    for (const FormatSpec& fs : g_formats) {
        // Build the source through the script environment so pixel_type
        // handling matches real scripts:
        char srcScript[128];
        sprintf_s(srcScript, "BlankClip(width=%d, height=%d, pixel_type=\"%s\", length=4, color=$4080C0)",
                  W, H, fs.pixel_type);
        AVSValue srcClipV;
        try {
            srcClipV = env->Invoke("Eval", AVSValue(srcScript));
        } catch (const AvisynthError& e) {
            printf("[FAIL] source eval %s: %s\n", fs.pixel_type, e.msg);
            ++g_failures;
            continue;
        }
        // Wrap in our deterministic pattern source (BlankClip only pins the
        // format); the reference reads the actual frame pixels either way.
        PClip clip(new RandSource(srcClipV.AsClip()->GetVideoInfo(), 0xC0FFEEu ^ fs.bits));
        const VideoInfo& cvi = clip->GetVideoInfo();

        for (const Params& p : g_params) {
            char tag[160];
            sprintf_s(tag, "%s str=%d thrs=%.1f repair=%d struv=%d show=%d",
                      fs.pixel_type, p.str, p.thrs, p.repair, p.struv, p.show);
            runCase(env, tag, clip, p, cvi, fs.bits);
        }

        // defaults-only invocation (named-param defaults path)
        {
            AVSValue args[1] = { AVSValue(clip) };
            try {
                AVSValue ret = env->Invoke("KEdgeLevelOCL", AVSValue(args, 1));
                if (!ret.IsClip()) throw AvisynthError("no clip");
                printf("[PASS] %s defaults-only invocation\n", fs.pixel_type);
            } catch (const AvisynthError& e) {
                printf("[FAIL] %s defaults-only: %s\n", fs.pixel_type, e.msg);
                ++g_failures;
            }
            ++g_cases;
        }
    }

    // ---- real script string with a real Source clip (ColorBars) ----
    try {
        const char* script =
            "KEdgeLevelOCL(ColorBars(pixel_type=\"YV12\", width=64, height=48), "
            "str=16, thrs=8.0, repair=1, struv=16)";
        AVSValue ret = env->Invoke("Eval", AVSValue(script));
        if (!ret.IsClip()) throw AvisynthError("Eval did not return a clip");
        AVSValue srcV = env->Invoke("Eval",
            AVSValue("ColorBars(pixel_type=\"YV12\", width=64, height=48)"));
        PClip srcClip = srcV.AsClip();
        PClip filtered = ret.AsClip();
        const VideoInfo& vi = filtered->GetVideoInfo();
        PVideoFrame srcf = srcClip->GetFrame(0, env);
        PVideoFrame outf = filtered->GetFrame(0, env);
        Params p{ 16, 8.0f, 1, 16, 0 };
        bool ok = refAndCompare<uint8_t>("script-string", vi, srcf, outf, p);
        ++g_cases;
        printf("[%s] script-string ColorBars YV12 (str=16 thrs=8 repair=1 struv=16)\n",
               ok ? "PASS" : "FAIL");
        if (!ok) ++g_failures;
    } catch (const AvisynthError& e) {
        printf("[FAIL] script-string case: %s\n", e.msg);
        ++g_failures;
    }

    printf("\n=== KEdgeLevelOCL: %d cases (%d frame checks), %d failures ===\n",
           g_cases, g_frame_checks, g_failures);

    env->DeleteScriptEnvironment();
    // NOTE: deliberately not FreeLibrary — AviSynth+ keeps internal threads
    // referencing the DLL at shutdown.
    return g_failures ? 1 : 0;
}
