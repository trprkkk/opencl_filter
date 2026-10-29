// kfm_mergestatic_avs.cpp — AviSynth+ host plugin for the OpenCL port of KFM's
// MergeStatic.cu (upstream: rigaya/AviSynthCUDAFilters KFM/MergeStatic.cu, MIT).
//
// The FOURTH real AviSynth+ host plugin of the port (after KEdgeLevelOCL,
// KTemporalNROCL, KDebandOCL): it loads the ALG-VERIFIED kernels of
// src/opencl/kfm/kernels/kfm_mergestatic.cl (kf_compare_frames and
// kf_merge_static, embedded into the DLL at build time) through the small host
// runtime in src/host/ocl_host.* and replays the upstream host pipeline of TWO
// of the three filters that MergeStatic.cu registers:
//
//   KTemporalDiff ("c") — GetFrame (upstream KTemporalDiff::GetFrameT):
//     DIST = 2, N_REFS = 5; fetch child frames at ref = clamp(n-2+i) for
//     i in [0,5); then ONE kf_compare_frames launch per plane (Y at luma dims,
//     U/V at chroma dims), exactly upstream's three kl_compare_frames /
//     cpu_compare_frames calls: dst = max(f0..f4) - min(f0..f4), the temporal
//     spread ("diff flag") frame.
//
//   KMergeStatic ("ccc": clip60, clip30, sttclip) — GetFrame (upstream
//     KMergeStatic::GetFrameT): n30 = n>>1; flag = sttclip->GetFrame(n30);
//     frame60 = child->GetFrame(n); frame30 = clip30->GetFrame(n30); then ONE
//     kf_merge_static launch per plane:
//       dst = (coef*v30 + (128-coef)*v60 + 64) >> 7
//     with coef = flag value (0 => keep the 60fps frame, 128 => take the
//     30fps frame).  The constructor mirrors upstream's
//     `vi.num_frames != srcvi.num_frames * 2` check.
//
// STATE MODEL (both filters, stated like the KTemporalNROCL precedent):
// neither upstream class holds ANY persistent state between GetFrame calls —
// no accumulation, no scene-change reset, no previous-frame memory.  Every
// output frame is a pure function of the fetched input frames + the class's
// fixed configuration:
//   - KTemporalDiff: output n is a function of the child's frames
//     [n-2 .. n+2] (clamped to the clip) alone.
//   - KMergeStatic: output n is a function of child(n), clip30(n>>1) and
//     sttclip(n>>1) alone.  The CROSS-FRAME aspect is the clip mapping
//     (60fps frame n blends against 30fps frame n/2 with a per-frame
//     coefficient clip), not state: KMergeStatic never looks at any other
//     output or input frame.  The "temporal" knowledge lives entirely in the
//     sttclip producer (upstream: KAnalyzeStatic over a KTemporalDiff chain —
//     a separate filter, assembled in SCRIPTS, not inside this class).
// Therefore the "scene-change handling" is per-pixel inside one frame's data
// (the coefficient plane) and there is no host-side state to reset.
//
// UPSTREAM REGISTRATION INVENTORY (AddFuncMergeStatic, MergeStatic.cu verbatim):
//   env->AddFunction("KTemporalDiff", "c", ...)
//   env->AddFunction("KAnalyzeStatic", "c[thcombe]f[thdiff]f[pad]c", ...)
//   env->AddFunction("KMergeStatic", "ccc", ...)
// KAnalyzeStatic is deliberately NOT implemented here: (a) the KMergeStatic
// AVS function never instantiates or invokes it (its Create only takes three
// clips; scripts feed it e.g. KAnalyzeStatic(KTemporalDiff(...)) output), and
// (b) KAnalyzeStatic's pipeline needs CompareFields / MergeUVCoefs /
// ExtendCoefs / ApplyUVCoefs from the KFMFilterBase/CombingAnalyze machinery
// (a later milestone).  Its two kernels (kf_min_frames, kf_and_coefs) are
// already ported and ALG-VERIFIED in the same .cl and are simply unused by
// this plugin.  The two classes implemented here launch exactly two of the
// file's four kernels (kf_compare_frames, kf_merge_static) — 100% of their
// device work.
//
// Kernel layout seam: the CUDA host launches per plane passing 5 (resp. 3+dst)
// separate plane pointers with vector pitch; the OpenCL kernels take the same
// SHAPE — separate plain planes with pitch == width in ELEMENTS (px-sized
// elements; with PX=uchar elements == bytes, with PX=ushort the ELEMENT
// convention is the golden-pinned one, see the KTemporalNROCL deviation 4
// precedent).  This plugin allocates one plain w*h device plane per input
// plane per frame and passes one pointer each, exactly mirroring the upstream
// src0..src4 / src60/src30/flag call signatures.
//
// Documented deviations from the upstream host code (none change any pixel):
//   1. Upstream's GetRefFrame clamps to `clamp(ref, 0, vi.num_frames)` — an
//      OFF-BY-ONE upper bound (GetFrame(num_frames) is past the last frame;
//      it only ever worked because AviSynth+ caches clamp internally).  This
//      host clamps to vi.num_frames - 1 (the KTemporalNROCL convention).
//      Identical pixels for any cache-backed clip; avoids UB on raw clips.
//   2. Upstream's KMergeStatic::GetFrameT first CopyFrame's frame60 into dst
//      and then launches kl_merge_static over every column of the plane — the
//      copy is fully overwritten by the kernel on the scalar path and is
//      therefore skipped here (the upstream CUDA path needs it only for the
//      width%4 trailing columns its vector kernels never write; the OpenCL
//      kernel processes every column).
//   3. Upstream's CUDA path processes only full 4-wide vectors
//      (width4 = width>>2), leaving width%4 trailing columns unwritten (its
//      scalar CPU path processes every column).  The scalar OpenCL kernel
//      processes every column always — the goldens pin this.
//   4. Device planes use pitch == width plain layouts (host staging copies
//      from the frame's real pitch); kernels are launched over the whole
//      plane with every access guarded, so no border policy is needed.
//   5. Upstream's KMergeStatic constructor also checks
//      `GetDeviceTypes(clip60) & GetDeviceTypes(clip30)` (CUDA device
//      matching) — meaningless on the OpenCL host (everything goes through
//      this plugin's one device queue), so it is not reproduced.
//
// The plugin is a plain DLL exporting AvisynthPluginInit3 (no import lib); it
// validates autoload + registration against the installed AviSynth+ 3.7.3 in
// test_kfm_mergestatic.cpp.
#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "avisynth.h"
#include "../ocl_host.h"
#include "kfm_mergestatic_cl.h"  // GENERATED from src/opencl/kfm/kernels/kfm_mergestatic.cl

namespace {

using oclhost::Buffer;
using oclhost::Context;
using oclhost::Program;

// One output/input plane: plain w*h device image, pitch == width.
struct Plane {
    Buffer buf;
    int w = 0, h = 0, px = 1;
    size_t bytes() const { return (size_t)w * h * px; }
    void alloc(Context& ctx, int w_, int h_, int px_) {
        w = w_; h = h_; px = px_;
        buf = Buffer(ctx, bytes(), /*zero=*/false);
    }
    cl_mem mem() const { return buf.mem(); }
};

// Copies a frame plane (arbitrary source pitch) into a plain device plane
// (pitch == w) through host staging.
void uploadPlane(Plane& p, const uint8_t* srcp, int srcPitch,
                 int rowsize, int height, std::vector<uint8_t>& stage) {
    for (int y = 0; y < height; ++y)
        memcpy(stage.data() + (size_t)y * rowsize, srcp + (size_t)y * srcPitch, rowsize);
    p.buf.write(stage.data(), 0, p.bytes());
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

// ---------------------------------------------------------------------------
// KTemporalDiff — upstream DIST=2, N_REFS=5, per plane one kf_compare_frames.
// ---------------------------------------------------------------------------
class KTemporalDiffOCL : public GenericVideoFilter {
    static constexpr int DIST = 2;
    static constexpr int N_REFS = DIST * 2 + 1;

    int logUVx_, logUVy_;
    int px_;  // ComponentSize

    Plane dstY_, dstU_, dstV_;
    Plane srcY_[N_REFS], srcU_[N_REFS], srcV_[N_REFS];

    std::vector<uint8_t> stageY_;
    std::vector<uint8_t> stageUV_;
    std::mutex* mtx_;  // serialises the shared in-order queue

    Program& program() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "kfm_mergestatic", kfm_mergestatic_cl_src, px_);
    }

    // one kf_compare_frames launch per plane (upstream: one
    // kl_compare_frames / cpu_compare_frames call per plane)
    void dispatchCompareFrames(Plane* dst, Plane* src, int w, int h) {
        cl_kernel k = program().kernel("kf_compare_frames");
        oclhost::set_arg(k, 0, dst->mem());
        for (int i = 0; i < N_REFS; ++i)
            oclhost::set_arg(k, 1 + i, src[i].mem());
        oclhost::set_arg(k, 6, w);
        oclhost::set_arg(k, 7, h);
        oclhost::set_arg(k, 8, w);  // pitch == width (elements)
        oclhost::dispatch2D(sharedContext(), k, w, h);
    }

public:
    KTemporalDiffOCL(PClip clip, IScriptEnvironment* env_)
        : GenericVideoFilter(clip)
        , logUVx_(vi.GetPlaneWidthSubsampling(PLANAR_U))
        , logUVy_(vi.GetPlaneHeightSubsampling(PLANAR_U)) {
        if (!vi.IsPlanar() || !vi.IsYUV() || vi.IsYUVA())
            env_->ThrowError("[KTemporalDiffOCL] planar YUV input required");
        px_ = vi.ComponentSize();
        if (px_ != 1 && px_ != 2)
            env_->ThrowError("[KTemporalDiffOCL] Unsupported pixel format");

        static std::mutex mtx;  // one shared queue per process
        mtx_ = &mtx;

        try {
            Context& ctx = sharedContext();
            const int w = vi.width, h = vi.height;
            const int wUV = w >> logUVx_, hUV = h >> logUVy_;

            for (int i = 0; i < N_REFS; ++i) {
                srcY_[i].alloc(ctx, w, h, px_);
                srcU_[i].alloc(ctx, wUV, hUV, px_);
                srcV_[i].alloc(ctx, wUV, hUV, px_);
            }
            dstY_.alloc(ctx, w, h, px_);
            dstU_.alloc(ctx, wUV, hUV, px_);
            dstV_.alloc(ctx, wUV, hUV, px_);
            stageY_.resize((size_t)w * h * px_);
            stageUV_.resize((size_t)wUV * hUV * px_);
            program();  // build both PX instantiations up front (HOST_CONTRACT §1)
        } catch (const std::exception& e) {
            env_->ThrowError("[KTemporalDiffOCL] OpenCL init failed: %s", e.what());
        }
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            if (px_ == 1) return GetFrameT<uint8_t>(n, env_);
            return GetFrameT<uint16_t>(n, env_);
        } catch (const std::exception& e) {
            env_->ThrowError("[KTemporalDiffOCL] %s", e.what());
        }
        return PVideoFrame();
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) {
        // mirror upstream KTemporalDiff::SetCacheHints (MT_NICE_FILTER);
        // device work is serialised through the shared in-order queue + mutex.
        if (cachehints == CACHE_GET_MTMODE) return MT_NICE_FILTER;
        return 0;
    }

private:
    template <typename pixel_t>
    PVideoFrame GetFrameT(int n, IScriptEnvironment* env_) {
        const int w = vi.width, h = vi.height;
        const int wUV = w >> logUVx_, hUV = h >> logUVy_;
        const int rowsizeY = w * px_;
        const int rowsizeUV = wUV * px_;

        std::lock_guard<std::mutex> lock(*mtx_);

        // ---- fetch the clamped 5-frame window (upstream GetRefFrame, with
        // the documented deviation-1 upper-bound fix) ----
        for (int i = 0; i < N_REFS; ++i) {
            const int ref = std::max(0, std::min(n - DIST + i, vi.num_frames - 1));
            PVideoFrame f = child->GetFrame(ref, env_);
            uploadPlane(srcY_[i], f->GetReadPtr(PLANAR_Y), f->GetPitch(PLANAR_Y),
                        rowsizeY, h, stageY_);
            uploadPlane(srcU_[i], f->GetReadPtr(PLANAR_U), f->GetPitch(PLANAR_U),
                        rowsizeUV, hUV, stageUV_);
            uploadPlane(srcV_[i], f->GetReadPtr(PLANAR_V), f->GetPitch(PLANAR_V),
                        rowsizeUV, hUV, stageUV_);
        }

        // ---- one launch per plane (Y at luma dims, U/V at chroma dims) ----
        dispatchCompareFrames(&dstY_, srcY_, w, h);
        dispatchCompareFrames(&dstU_, srcU_, wUV, hUV);
        dispatchCompareFrames(&dstV_, srcV_, wUV, hUV);

        sharedContext().finish();

        // ---- download into a fresh output frame ----
        PVideoFrame dstFrame = env_->NewVideoFrame(vi);
        dstY_.buf.read(stageY_.data(), 0, dstY_.bytes());
        copyToFrame(stageY_.data(), rowsizeY, rowsizeY, h,
                    dstFrame->GetWritePtr(PLANAR_Y), dstFrame->GetPitch(PLANAR_Y));
        dstU_.buf.read(stageUV_.data(), 0, dstU_.bytes());
        copyToFrame(stageUV_.data(), rowsizeUV, rowsizeUV, hUV,
                    dstFrame->GetWritePtr(PLANAR_U), dstFrame->GetPitch(PLANAR_U));
        dstV_.buf.read(stageUV_.data(), 0, dstV_.bytes());
        copyToFrame(stageUV_.data(), rowsizeUV, rowsizeUV, hUV,
                    dstFrame->GetWritePtr(PLANAR_V), dstFrame->GetPitch(PLANAR_V));
        return dstFrame;
    }
};

// ---------------------------------------------------------------------------
// KMergeStatic — upstream "ccc" (clip60, clip30, sttclip), per plane one
// kf_merge_static over child(n), clip30(n>>1), sttclip(n>>1).
// ---------------------------------------------------------------------------
class KMergeStaticOCL : public GenericVideoFilter {
    PClip clip30_;
    PClip sttclip_;

    int logUVx_, logUVy_;
    int px_;

    Plane dstY_, dstU_, dstV_;
    Plane src60Y_, src60U_, src60V_;
    Plane src30Y_, src30U_, src30V_;
    Plane flagY_, flagU_, flagV_;

    std::vector<uint8_t> stageY_;
    std::vector<uint8_t> stageUV_;
    std::mutex* mtx_;

    Program& program() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "kfm_mergestatic", kfm_mergestatic_cl_src, px_);
    }

    // one kf_merge_static launch per plane (upstream: one kl_merge_static /
    // cpu_merge_static call per plane)
    void dispatchMergeStatic(Plane* dst, Plane* src60, Plane* src30, Plane* flag,
                             int w, int h) {
        cl_kernel k = program().kernel("kf_merge_static");
        oclhost::set_arg(k, 0, dst->mem());
        oclhost::set_arg(k, 1, src60->mem());
        oclhost::set_arg(k, 2, src30->mem());
        oclhost::set_arg(k, 3, flag->mem());
        oclhost::set_arg(k, 4, w);
        oclhost::set_arg(k, 5, h);
        oclhost::set_arg(k, 6, w);  // pitch == width (elements)
        oclhost::dispatch2D(sharedContext(), k, w, h);
    }

public:
    KMergeStaticOCL(PClip clip60, PClip clip30, PClip sttclip, IScriptEnvironment* env_)
        : GenericVideoFilter(clip60)
        , clip30_(clip30)
        , sttclip_(sttclip)
        , logUVx_(vi.GetPlaneWidthSubsampling(PLANAR_U))
        , logUVy_(vi.GetPlaneHeightSubsampling(PLANAR_U)) {
        if (!vi.IsPlanar() || !vi.IsYUV() || vi.IsYUVA())
            env_->ThrowError("[KMergeStaticOCL] planar YUV input required");
        px_ = vi.ComponentSize();
        if (px_ != 1 && px_ != 2)
            env_->ThrowError("[KMergeStaticOCL] Unsupported pixel format");

        // upstream constructor check (verbatim)
        const VideoInfo srcvi = clip30_->GetVideoInfo();
        if (vi.num_frames != srcvi.num_frames * 2)
            env_->ThrowError("[KMergeStaticOCL] Num frames don't match");
        // (upstream's GetDeviceTypes device-match check is CUDA-only —
        // documented deviation 5)

        static std::mutex mtx;  // one shared queue per process
        mtx_ = &mtx;

        try {
            Context& ctx = sharedContext();
            const int w = vi.width, h = vi.height;
            const int wUV = w >> logUVx_, hUV = h >> logUVy_;

            src60Y_.alloc(ctx, w, h, px_);
            src60U_.alloc(ctx, wUV, hUV, px_);
            src60V_.alloc(ctx, wUV, hUV, px_);
            src30Y_.alloc(ctx, w, h, px_);
            src30U_.alloc(ctx, wUV, hUV, px_);
            src30V_.alloc(ctx, wUV, hUV, px_);
            flagY_.alloc(ctx, w, h, px_);
            flagU_.alloc(ctx, wUV, hUV, px_);
            flagV_.alloc(ctx, wUV, hUV, px_);
            dstY_.alloc(ctx, w, h, px_);
            dstU_.alloc(ctx, wUV, hUV, px_);
            dstV_.alloc(ctx, wUV, hUV, px_);
            stageY_.resize((size_t)w * h * px_);
            stageUV_.resize((size_t)wUV * hUV * px_);
            program();
        } catch (const std::exception& e) {
            env_->ThrowError("[KMergeStaticOCL] OpenCL init failed: %s", e.what());
        }
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            if (px_ == 1) return GetFrameT<uint8_t>(n, env_);
            return GetFrameT<uint16_t>(n, env_);
        } catch (const std::exception& e) {
            env_->ThrowError("[KMergeStaticOCL] %s", e.what());
        }
        return PVideoFrame();
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) {
        // mirror upstream KMergeStatic::SetCacheHints (MT_NICE_FILTER)
        if (cachehints == CACHE_GET_MTMODE) return MT_NICE_FILTER;
        return 0;
    }

private:
    template <typename pixel_t>
    PVideoFrame GetFrameT(int n, IScriptEnvironment* env_) {
        const int w = vi.width, h = vi.height;
        const int wUV = w >> logUVx_, hUV = h >> logUVy_;
        const int rowsizeY = w * px_;
        const int rowsizeUV = wUV * px_;

        std::lock_guard<std::mutex> lock(*mtx_);

        // ---- fetch the three source frames (upstream verbatim) ----
        const int n30 = n >> 1;
        PVideoFrame flag = sttclip_->GetFrame(n30, env_);
        PVideoFrame frame60 = child->GetFrame(n, env_);
        PVideoFrame frame30 = clip30_->GetFrame(n30, env_);

        uploadPlane(src60Y_, frame60->GetReadPtr(PLANAR_Y), frame60->GetPitch(PLANAR_Y),
                    rowsizeY, h, stageY_);
        uploadPlane(src60U_, frame60->GetReadPtr(PLANAR_U), frame60->GetPitch(PLANAR_U),
                    rowsizeUV, hUV, stageUV_);
        uploadPlane(src60V_, frame60->GetReadPtr(PLANAR_V), frame60->GetPitch(PLANAR_V),
                    rowsizeUV, hUV, stageUV_);
        uploadPlane(src30Y_, frame30->GetReadPtr(PLANAR_Y), frame30->GetPitch(PLANAR_Y),
                    rowsizeY, h, stageY_);
        uploadPlane(src30U_, frame30->GetReadPtr(PLANAR_U), frame30->GetPitch(PLANAR_U),
                    rowsizeUV, hUV, stageUV_);
        uploadPlane(src30V_, frame30->GetReadPtr(PLANAR_V), frame30->GetPitch(PLANAR_V),
                    rowsizeUV, hUV, stageUV_);
        // the flag (coefficient) clip is read at the same per-plane dims as the
        // 60/30fps clips, exactly like upstream's shared width4/heightUV/pitch
        uploadPlane(flagY_, flag->GetReadPtr(PLANAR_Y), flag->GetPitch(PLANAR_Y),
                    rowsizeY, h, stageY_);
        uploadPlane(flagU_, flag->GetReadPtr(PLANAR_U), flag->GetPitch(PLANAR_U),
                    rowsizeUV, hUV, stageUV_);
        uploadPlane(flagV_, flag->GetReadPtr(PLANAR_V), flag->GetPitch(PLANAR_V),
                    rowsizeUV, hUV, stageUV_);

        // ---- one launch per plane (Y at luma dims, U/V at chroma dims) ----
        dispatchMergeStatic(&dstY_, &src60Y_, &src30Y_, &flagY_, w, h);
        dispatchMergeStatic(&dstU_, &src60U_, &src30U_, &flagU_, wUV, hUV);
        dispatchMergeStatic(&dstV_, &src60V_, &src30V_, &flagV_, wUV, hUV);

        sharedContext().finish();

        // ---- download into a fresh output frame ----
        PVideoFrame dstFrame = env_->NewVideoFrame(vi);
        dstY_.buf.read(stageY_.data(), 0, dstY_.bytes());
        copyToFrame(stageY_.data(), rowsizeY, rowsizeY, h,
                    dstFrame->GetWritePtr(PLANAR_Y), dstFrame->GetPitch(PLANAR_Y));
        dstU_.buf.read(stageUV_.data(), 0, dstU_.bytes());
        copyToFrame(stageUV_.data(), rowsizeUV, rowsizeUV, hUV,
                    dstFrame->GetWritePtr(PLANAR_U), dstFrame->GetPitch(PLANAR_U));
        dstV_.buf.read(stageUV_.data(), 0, dstV_.bytes());
        copyToFrame(stageUV_.data(), rowsizeUV, rowsizeUV, hUV,
                    dstFrame->GetWritePtr(PLANAR_V), dstFrame->GetPitch(PLANAR_V));
        return dstFrame;
    }
};

AVSValue __cdecl CreateKTemporalDiffOCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        return new KTemporalDiffOCL(args[0].AsClip(), env);
    } catch (const std::exception& e) {
        env->ThrowError("[KTemporalDiffOCL] %s", e.what());
    }
    return AVSValue();
}

AVSValue __cdecl CreateKMergeStaticOCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        return new KMergeStaticOCL(
            args[0].AsClip(),   // clip60
            args[1].AsClip(),   // clip30
            args[2].AsClip(),   // sttclip
            env);
    } catch (const std::exception& e) {
        env->ThrowError("[KMergeStaticOCL] %s", e.what());
    }
    return AVSValue();
}

}  // namespace

// AviSynth+ linkage: the core passes the function-table pointer as the 2nd
// argument of AvisynthPluginInit3 (no avisynth.lib import needed).
const AVS_Linkage* AVS_linkage{ nullptr };

extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
    AVS_linkage = vectors;
    try {
        // exact upstream signatures from AddFuncMergeStatic (MergeStatic.cu)
        env->AddFunction("KTemporalDiffOCL", "c", CreateKTemporalDiffOCL, 0);
        env->AddFunction("KTemporalDiff", "c", CreateKTemporalDiffOCL, 0);
        env->AddFunction("KMergeStaticOCL", "ccc", CreateKMergeStaticOCL, 0);
        env->AddFunction("KMergeStatic", "ccc", CreateKMergeStaticOCL, 0);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "KTemporalDiff + KMergeStatic (OpenCL port of KFM/MergeStatic.cu, MIT; fourth host plugin of the OpenCL port)";
}
