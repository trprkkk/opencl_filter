// kfm_edgelevel_avs.cpp — AviSynth+ host plugin for the OpenCL port of KFM's
// KEdgeLevel (upstream: rigaya/AviSynthCUDAFilters KFM/KDeband.cu, MIT).
//
// This is the FIRST real AviSynth+ host plugin of the port: it loads the
// ALG-VERIFIED kernels of src/opencl/kfm/kernels/kfm_edgelevel.cl
// (kf_edgelevel / kf_edgelevel_repair / kf_el_to444 / kf_el_from444, embedded
// into the DLL at build time) through the small host runtime in
// src/host/ocl_host.* and replays the upstream filter's host pipeline:
//
//   GetFrameT (upstream KEdgeLevel::GetFrameT):
//     numel = max((repair+1)/2, 1) strengthening passes with per-iteration
//     table_idx = ((show && last)?4:0) + (repair?2:0) + (uv?1:0) over the
//     check x selective x uv template tuples (selective == repair>0, exactly
//     as the upstream dispatch table encodes it), then repair passes of
//     edgelevel_repair<N=3> (ping-pong through tmp), then either ClearUV
//     (show), the from444 + Y-copy GetOutFrame (uv), or a plain CopyUV.
//
// Documented deviations from the upstream host code (all forced by memory
// safety / OpenCL pointer arithmetic; the per-kernel arithmetic is untouched):
//   1. GetUVFrame's to444 call: upstream passes the LUMA (vi.width/height)
//      dims as the kernel's source width/height, which drives both the source
//      reads and the BW*x/BH*y destination writes out of bounds for any
//      subsampled clip (an upstream call-site bug; the kernel itself is a
//      source-dims transliteration and the CPU mirror/goldens pin the
//      source-dims reading).  This plugin passes the CHROMA plane dims
//      (vi.width>>logUVx, vi.height>>logUVy) and dst pitch = luma pitch,
//      which is the only self-consistent reading (and matches GetOutFrame,
//      where upstream passes the chroma dims correctly).
//   2. kf_edgelevel_repair reads its 3x3 window with no border guard
//      (upstream relies on undefined out-of-plane reads of the AviSynth
//      allocation).  This plugin launches the repair kernel over the INTERIOR
//      only (global work offset (1,1), size (w-2,h-2) — the rig's own
//      technique, dev_kfm1.py), so every neighbour read stays in bounds of a
//      plain zero-initialised plane buffer.  Border policy (deterministic,
//      replacing upstream's undefined reads): before the repair loop the host
//      sets the border ring of every repair DESTINATION plane (el/tmp scratch)
//      to the pass-0 repair source's ring (device ring copy), so later passes'
//      neighbour reads of previous outputs are deterministic; the Y border
//      ring of the output frame passes through from the SOURCE frame (host
//      copy at download).  The repair INTERIOR is pinned by the repo goldens
//      either way.
//   3. show+uv: upstream returns the 444-sized work frame while the filter
//      still declares the subsampled vi (its ClearUV grey fill is read back at
//      the chroma dims by any consumer).  This plugin returns a vi-typed frame
//      with the same grey chroma at the chroma dims — identical pixel content.
//   4. !uv CopyUV is a plain plane copy; done on the host (memcpy-equivalent
//      of upstream's kl_copy_2plane/cpu_copy).
//
// The plugin is a plain DLL exporting AvisynthPluginInit3 (no import lib);
// it validates autoload + registration against the installed AviSynth+ 3.7.3
// in test_kfm_edgelevel.cpp.
#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "avisynth.h"
#include "../ocl_host.h"
#include "kfm_edgelevel_cl.h"  // GENERATED from src/opencl/kfm/kernels/kfm_edgelevel.cl

namespace {

using oclhost::Buffer;
using oclhost::Context;
using oclhost::Program;

// upstream KDeband.cu / KFMFilterBase.cu helpers (verbatim semantics)
int scaleParam(float thresh, int pixelBits) {
    return (int)(thresh * (1 << (pixelBits - 8)) + 0.5f);
}

int Get444Type(const VideoInfo& vi) {
    switch (vi.BitsPerComponent()) {
    case 8: return VideoInfo::CS_YV24;
    case 10: return VideoInfo::CS_YUV444P10;
    case 12: return VideoInfo::CS_YUV444P12;
    case 14: return VideoInfo::CS_YUV444P14;
    case 16: return VideoInfo::CS_YUV444P16;
    case 32: return VideoInfo::CS_YUV444PS;
    }
    return VideoInfo::CS_YV24;
}

// One plain plane: a zero-initialised w*h device image, pitch == width, no
// guard ring.  Kernels with unguarded neighbour reads (repair) are dispatched
// over the interior only (see dispatchRepair); the border ring is owned by the
// host (see patchRing / the download composition in GetFrameT).
struct Plane {
    Buffer buf;
    int w = 0, h = 0, pitch = 0;
    int px = 1;  // bytes per sample

    void alloc(Context& ctx, int w_, int h_, int px_) {
        w = w_; h = h_; pitch = w_; px = px_;
        buf = Buffer(ctx, (size_t)w * h * px, /*zero=*/true);
    }
    size_t interiorOffset() const { return 0; }
    size_t interiorBytes() const { return (size_t)w * h * px; }
    cl_mem mem() const { return buf.mem(); }
};

// Copies the border ring (row 0, row h-1, col 0, col w-1) between two
// full-plane host images (pitch == w, px bytes per sample).
void copyRing(uint8_t* dst, const uint8_t* src, int w, int h, int px) {
    memcpy(dst, src, (size_t)w * px);
    memcpy(dst + (size_t)(h - 1) * w * px, src + (size_t)(h - 1) * w * px, (size_t)w * px);
    for (int y = 1; y < h - 1; ++y) {
        memcpy(dst + (size_t)y * w * px, src + (size_t)y * w * px, (size_t)px);
        memcpy(dst + (size_t)y * w * px + (size_t)(w - 1) * px,
               src + (size_t)y * w * px + (size_t)(w - 1) * px, (size_t)px);
    }
}

// Copies a source frame's border ring into an output frame plane (frame
// pitches may differ from w).
void copyRingToFrame(uint8_t* dstp, int dstPitch, const uint8_t* srcp, int srcPitch,
                     int w, int h, int px) {
    memcpy(dstp, srcp, (size_t)w * px);
    memcpy(dstp + (size_t)(h - 1) * dstPitch, srcp + (size_t)(h - 1) * srcPitch, (size_t)w * px);
    for (int y = 1; y < h - 1; ++y) {
        memcpy(dstp + (size_t)y * dstPitch, srcp + (size_t)y * srcPitch, (size_t)px);
        memcpy(dstp + (size_t)y * dstPitch + (size_t)(w - 1) * px,
               srcp + (size_t)y * srcPitch + (size_t)(w - 1) * px, (size_t)px);
    }
}

void copyFromFrame(const uint8_t* srcp, int srcPitch, int rowsize, int height, uint8_t* dst) {
    for (int y = 0; y < height; ++y)
        memcpy(dst + (size_t)y * rowsize, srcp + (size_t)y * srcPitch, rowsize);
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

class KEdgeLevelOCL : public GenericVideoFilter {
    int str_, strUV_, thrs_, repair_;
    bool uv_, show_;
    int logUVx_, logUVy_;
    int bits_, maxv_;
    int px_;  // ComponentSize

    // device planes (guarded layout); named like the upstream frames
    Plane srcY_, dstY_, tmpY_, elY_;
    Plane chU_, chV_;                // upload buffers at chroma dims (subsampled)
    Plane srcU_, srcV_;              // "GetUVFrame" result (444 dims when uv_)
    Plane dstU_, dstV_, tmpU_, elU_, tmpV_, elV_;
    Plane outU_, outV_;              // from444 outputs at chroma dims

    std::vector<uint8_t> stage_;     // host staging (max plane)
    std::vector<uint8_t> stage2_;    // second staging (ring patching)
    std::mutex* mtx_;                // serialises the shared in-order queue

    Program& program() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "kfm_edgelevel", kfm_edgelevel_cl_src, px_);
    }

    void upload(const Plane& p, const uint8_t* srcp, int srcPitch, int rowsize) {
        copyFromFrame(srcp, srcPitch, rowsize, p.h, stage_.data());
        p.buf.write(stage_.data(), p.interiorOffset(), p.interiorBytes());
    }

    void download(const Plane& p, uint8_t* dstp, int dstPitch, int rowsize) {
        p.buf.read(stage_.data(), p.interiorOffset(), p.interiorBytes());
        copyToFrame(stage_.data(), (int)rowsize, rowsize, p.h, dstp, dstPitch);
    }

    // Border policy for the repair destination planes (deviation 2): before
    // the repair loop, dst's ring := src's ring (the pass-0 repair source),
    // so every later pass reads a deterministic border.  Blocking host round
    // trip through staging (in-order queue; planes are small).
    void patchRing(Plane& dstp, const Plane& srcp) {
        if (dstp.w < 3 || dstp.h < 3) return;
        srcp.buf.read(stage2_.data(), 0, srcp.interiorBytes());
        dstp.buf.read(stage_.data(), 0, dstp.interiorBytes());
        copyRing(stage_.data(), stage2_.data(), dstp.w, dstp.h, dstp.px);
        dstp.buf.write(stage_.data(), 0, dstp.interiorBytes());
    }

    void dispatchEdgelevel(Plane* dstY, Plane* dstU, Plane* dstV,
                           Plane* srcY, Plane* srcU, Plane* srcV,
                           int check, int selective, int uv) {
        cl_kernel k = program().kernel("kf_edgelevel");
        oclhost::set_arg(k, 0, dstY->mem());
        oclhost::set_arg(k, 1, dstU->mem());
        oclhost::set_arg(k, 2, dstV->mem());
        oclhost::set_arg(k, 3, srcY->mem());
        oclhost::set_arg(k, 4, srcU->mem());
        oclhost::set_arg(k, 5, srcV->mem());
        oclhost::set_arg(k, 6, dstY->w);
        oclhost::set_arg(k, 7, dstY->h);
        oclhost::set_arg(k, 8, dstY->pitch);
        oclhost::set_arg(k, 9, maxv_);
        oclhost::set_arg(k, 10, str_);
        oclhost::set_arg(k, 11, strUV_);
        oclhost::set_arg(k, 12, thrs_);
        oclhost::set_arg(k, 13, check);
        oclhost::set_arg(k, 14, selective);
        oclhost::set_arg(k, 15, uv);
        oclhost::dispatch2D(sharedContext(), k, dstY->w, dstY->h);
    }

    void dispatchRepair(Plane* dst, Plane* el, Plane* src) {
        cl_kernel k = program().kernel("kf_edgelevel_repair");
        oclhost::set_arg(k, 0, dst->mem());
        oclhost::set_arg(k, 1, el->mem());
        oclhost::set_arg(k, 2, src->mem());
        oclhost::set_arg(k, 3, dst->w);
        oclhost::set_arg(k, 4, dst->h);
        oclhost::set_arg(k, 5, dst->pitch);
        oclhost::set_arg(k, 6, 3);  // upstream N=3
        // Interior only (deviation 2): the kernel's 3x3 window is unguarded,
        // so (1,1)..(w-2,h-2) keeps every neighbour read in bounds of the
        // plain plane buffer.  The border ring is host-owned (patchRing /
        // download composition).
        oclhost::dispatch2DOffset(sharedContext(), k, 1, 1, dst->w - 2, dst->h - 2);
    }

    void dispatchTo444(Plane* dst444, Plane* srcCh) {
        cl_kernel k = program().kernel("kf_el_to444");
        oclhost::set_arg(k, 0, dst444->mem());
        oclhost::set_arg(k, 1, srcCh->mem());
        oclhost::set_arg(k, 2, srcCh->w);
        oclhost::set_arg(k, 3, srcCh->h);
        oclhost::set_arg(k, 4, dst444->pitch);
        oclhost::set_arg(k, 5, srcCh->pitch);
        oclhost::set_arg(k, 6, logUVx_);
        oclhost::set_arg(k, 7, logUVy_);
        oclhost::dispatch2D(sharedContext(), k, srcCh->w, srcCh->h);
    }

    void dispatchFrom444(Plane* dstCh, Plane* src444) {
        cl_kernel k = program().kernel("kf_el_from444");
        oclhost::set_arg(k, 0, dstCh->mem());
        oclhost::set_arg(k, 1, src444->mem());
        oclhost::set_arg(k, 2, dstCh->w);
        oclhost::set_arg(k, 3, dstCh->h);
        oclhost::set_arg(k, 4, dstCh->pitch);
        oclhost::set_arg(k, 5, src444->pitch);
        oclhost::set_arg(k, 6, logUVx_);
        oclhost::set_arg(k, 7, logUVy_);
        oclhost::dispatch2D(sharedContext(), k, dstCh->w, dstCh->h);
    }

public:
    KEdgeLevelOCL(PClip clip, int str, float thrs, int repair, int strUV, bool show,
                  IScriptEnvironment* env_)
        : GenericVideoFilter(clip)
        , str_(str), strUV_(strUV)
        , thrs_(scaleParam(thrs, vi.BitsPerComponent()))
        , repair_(repair), uv_(strUV > 0), show_(show)
        , logUVx_(vi.GetPlaneWidthSubsampling(PLANAR_U))
        , logUVy_(vi.GetPlaneHeightSubsampling(PLANAR_U)) {
        if (!vi.IsPlanar() || !vi.IsYUV() || vi.IsYUVA())
            env_->ThrowError("[KEdgeLevelOCL] planar YUV input required");
        bits_ = vi.BitsPerComponent();
        maxv_ = (1 << bits_) - 1;
        px_ = vi.ComponentSize();
        if (px_ != 1 && px_ != 2)
            env_->ThrowError("[KEdgeLevelOCL] Unsupported pixel format");

        static std::mutex mtx;  // one shared queue per process
        mtx_ = &mtx;

        try {
            Context& ctx = sharedContext();
            const int w = vi.width, h = vi.height;
            const int wUV = w >> logUVx_, hUV = h >> logUVy_;

            srcY_.alloc(ctx, w, h, px_);
            dstY_.alloc(ctx, w, h, px_);
            if (repair_ > 0) {
                tmpY_.alloc(ctx, w, h, px_);
                elY_.alloc(ctx, w, h, px_);
            }
            if (uv_) {
                if (logUVx_ > 0) {
                    chU_.alloc(ctx, wUV, hUV, px_);
                    chV_.alloc(ctx, wUV, hUV, px_);
                }
                srcU_.alloc(ctx, w, h, px_);
                srcV_.alloc(ctx, w, h, px_);
                dstU_.alloc(ctx, w, h, px_);
                dstV_.alloc(ctx, w, h, px_);
                if (repair_ > 0) {
                    tmpU_.alloc(ctx, w, h, px_);
                    elU_.alloc(ctx, w, h, px_);
                    tmpV_.alloc(ctx, w, h, px_);
                    elV_.alloc(ctx, w, h, px_);
                }
                if (logUVx_ > 0) {
                    outU_.alloc(ctx, wUV, hUV, px_);
                    outV_.alloc(ctx, wUV, hUV, px_);
                }
            }
            stage_.resize((size_t)w * h * px_);
        stage2_.resize(stage_.size());
            program();  // build both PX instantiations up front (HOST_CONTRACT §1)
        } catch (const std::exception& e) {
            env_->ThrowError("[KEdgeLevelOCL] OpenCL init failed: %s", e.what());
        }
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            if (px_ == 1) return GetFrameT<uint8_t>(n, env_);
            return GetFrameT<uint16_t>(n, env_);
        } catch (const std::exception& e) {
            env_->ThrowError("[KEdgeLevelOCL] %s", e.what());
        }
        return PVideoFrame();
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) {
        // mirror upstream KDebandBase (MT_NICE_FILTER); device work is
        // serialised through the shared in-order queue + plugin mutex.
        if (cachehints == CACHE_GET_MTMODE) return MT_NICE_FILTER;
        return 0;
    }

private:
    template <typename pixel_t>
    PVideoFrame GetFrameT(int n, IScriptEnvironment* env_) {
        const int w = vi.width, h = vi.height;
        const int wUV = w >> logUVx_, hUV = h >> logUVy_;
        const size_t bytesY = (size_t)w * px_;
        const size_t rowsizeUV = (size_t)wUV * px_;

        std::lock_guard<std::mutex> lock(*mtx_);

        PVideoFrame srcFrame = child->GetFrame(n, env_);
        PVideoFrame dstFrame = env_->NewVideoFrame(vi);

        // ---- upload Y (and chroma sources) ----
        upload(srcY_, srcFrame->GetReadPtr(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y), (int)bytesY);
        if (uv_) {
            if (logUVx_ > 0) {
                upload(chU_, srcFrame->GetReadPtr(PLANAR_U), srcFrame->GetPitch(PLANAR_U), (int)rowsizeUV);
                upload(chV_, srcFrame->GetReadPtr(PLANAR_V), srcFrame->GetPitch(PLANAR_V), (int)rowsizeUV);
                // upstream GetUVFrame (deviation 1: chroma source dims)
                dispatchTo444(&srcU_, &chU_);
                dispatchTo444(&srcV_, &chV_);
            } else {
                upload(srcU_, srcFrame->GetReadPtr(PLANAR_U), srcFrame->GetPitch(PLANAR_U), (int)bytesY);
                upload(srcV_, srcFrame->GetReadPtr(PLANAR_V), srcFrame->GetPitch(PLANAR_V), (int)bytesY);
            }
        }

        // ---- strengthening passes (upstream GetFrameT loop) ----
        const int numel = std::max((repair_ + 1) / 2, 1);
        Plane *dstY = &dstY_, *dstU = &dstU_, *dstV = &dstV_;
        Plane *tmpY = repair_ > 0 ? &tmpY_ : nullptr;
        Plane *tmpU = repair_ > 0 ? &tmpU_ : nullptr;
        Plane *tmpV = repair_ > 0 ? &tmpV_ : nullptr;
        Plane *elY = repair_ > 0 ? &elY_ : nullptr;
        Plane *elU = repair_ > 0 ? &elU_ : nullptr;
        Plane *elV = repair_ > 0 ? &elV_ : nullptr;
        Plane* srcUin = uv_ ? &srcU_ : &srcY_;
        Plane* srcVin = uv_ ? &srcV_ : &srcY_;

        for (int i = 0; i < numel; ++i) {
            const int table_idx = ((show_ && (i + 1 == numel)) ? 4 : 0) + (repair_ ? 2 : 0) + (uv_ ? 1 : 0);
            Plane* elsrcY = &srcY_;
            Plane* elsrcU = srcUin;
            Plane* elsrcV = srcVin;
            if (i > 0) {
                std::swap(dstY, tmpY);
                std::swap(dstU, tmpU);
                std::swap(dstV, tmpV);
                elsrcY = tmpY;
                elsrcU = tmpU;
                elsrcV = tmpV;
            }
            dispatchEdgelevel(dstY, uv_ ? dstU : &dstY_, uv_ ? dstV : &dstY_,
                              elsrcY, uv_ ? elsrcU : elsrcY, uv_ ? elsrcV : elsrcY,
                              (table_idx >> 2) & 1, (table_idx >> 1) & 1, table_idx & 1);
        }

        // ---- post passes ----
        if (show_) {
            // ClearUV: chroma becomes mid-grey (deviation 3: written straight
            // into the output frame below, at the chroma dims).
        } else {
            if (repair_ > 0) {
                // dst <-> el, then repair passes ping-ponging through tmp.
                // Deterministic border policy (deviation 2): every repair dst
                // plane carries the pass-0 repair source's ring.
                std::swap(dstY, elY);
                std::swap(dstU, elU);
                std::swap(dstV, elV);
                patchRing(*dstY, srcY_);
                patchRing(*tmpY, srcY_);
                if (uv_) {
                    patchRing(*dstU, srcU_);
                    patchRing(*tmpU, srcU_);
                    patchRing(*dstV, srcV_);
                    patchRing(*tmpV, srcV_);
                }
                for (int i = 0; i < repair_; ++i) {
                    Plane* repsrcY = &srcY_;
                    Plane* repsrcU = srcUin;
                    Plane* repsrcV = srcVin;
                    if (i > 0) {
                        std::swap(dstY, tmpY);
                        std::swap(dstU, tmpU);
                        std::swap(dstV, tmpV);
                        repsrcY = tmpY;
                        repsrcU = tmpU;
                        repsrcV = tmpV;
                    }
                    dispatchRepair(dstY, elY, repsrcY);
                    if (uv_) {
                        dispatchRepair(dstU, elU, repsrcU);
                        dispatchRepair(dstV, elV, repsrcV);
                    }
                }
            }
            if (uv_ && logUVx_ > 0) {
                // upstream GetOutFrame: from444 back to the chroma dims
                dispatchFrom444(&outU_, dstU);
                dispatchFrom444(&outV_, dstV);
            }
        }

        sharedContext().finish();

        // ---- download ----
        download(*dstY, dstFrame->GetWritePtr(PLANAR_Y), dstFrame->GetPitch(PLANAR_Y), (int)bytesY);
        if (repair_ > 0 && !show_) {
            // deviation 2: the repair dst border ring is host-owned — the
            // output Y ring passes through from the SOURCE frame.
            copyRingToFrame(dstFrame->GetWritePtr(PLANAR_Y), dstFrame->GetPitch(PLANAR_Y),
                            srcFrame->GetReadPtr(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y),
                            w, h, px_);
        }

        const int hUVout = h >> logUVy_;
        uint8_t* outUp = dstFrame->GetWritePtr(PLANAR_U);
        uint8_t* outVp = dstFrame->GetWritePtr(PLANAR_V);
        const int dstPitchUV = dstFrame->GetPitch(PLANAR_U);

        if (show_) {
            // ClearUV: mid-grey fill (1 << (bits-1)) at the chroma dims
            for (int y = 0; y < hUVout; ++y) {
                for (int x = 0; x < wUV; ++x) {
                    pixel_t v = (pixel_t)(1 << (bits_ - 1));
                    memcpy(outUp + (size_t)y * dstPitchUV + (size_t)x * px_, &v, px_);
                    memcpy(outVp + (size_t)y * dstPitchUV + (size_t)x * px_, &v, px_);
                }
            }
        } else if (uv_) {
            if (logUVx_ > 0) {
                download(outU_, outUp, dstPitchUV, (int)rowsizeUV);
                download(outV_, outVp, dstFrame->GetPitch(PLANAR_V), (int)rowsizeUV);
            } else {
                // 4:4:4 input: chroma planes are already the processed 444
                // planes (the local pointers track the repair ping-pong)
                download(*dstU, outUp, dstPitchUV, (int)bytesY);
                download(*dstV, outVp, dstFrame->GetPitch(PLANAR_V), (int)bytesY);
            }
        } else {
            // !uv CopyUV: plain host copy of the untouched chroma
            copyToFrame(srcFrame->GetReadPtr(PLANAR_U), srcFrame->GetPitch(PLANAR_U),
                        (int)rowsizeUV, hUVout, outUp, dstPitchUV);
            copyToFrame(srcFrame->GetReadPtr(PLANAR_V), srcFrame->GetPitch(PLANAR_V),
                        (int)rowsizeUV, hUVout, outVp, dstFrame->GetPitch(PLANAR_V));
        }

        return dstFrame;
    }
};

AVSValue __cdecl CreateKEdgeLevelOCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        return new KEdgeLevelOCL(
            args[0].AsClip(),            // clip
            args[1].AsInt(10),           // str
            (float)args[2].AsFloat(25),  // thrs
            args[3].AsInt(0),            // repair
            args[4].AsInt(0),            // struv
            args[5].AsBool(false),       // show
            env);
    } catch (const std::exception& e) {
        env->ThrowError("[KEdgeLevelOCL] %s", e.what());
    }
    return AVSValue();
}

}  // namespace

// AviSynth+ linkage: the core passes the function-table pointer as the 2nd
// argument of AvisynthPluginInit3 (see the header's extern declaration above;
// no avisynth.lib import needed).
const AVS_Linkage* AVS_linkage{ nullptr };

extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
    AVS_linkage = vectors;
    try {
        env->AddFunction("KEdgeLevelOCL", "c[str]i[thrs]f[repair]i[struv]i[show]b",
                         CreateKEdgeLevelOCL, 0);
        // upstream function name as an alias (upstream signature identical)
        env->AddFunction("KEdgeLevel", "c[str]i[thrs]f[repair]i[struv]i[show]b",
                         CreateKEdgeLevelOCL, 0);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "KEdgeLevel (OpenCL port of KFM/KDeband.cu, MIT; first host plugin of the OpenCL port)";
}
