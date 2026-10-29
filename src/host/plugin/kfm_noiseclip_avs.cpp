// kfm_noiseclip_avs.cpp — AviSynth+ host plugin for the OpenCL port of KFM's
// KNoiseClip (upstream: rigaya/AviSynthCUDAFilters KFM/DecombeUCF.cu, MIT).
//
// The FIFTH real AviSynth+ host plugin of the port (after KEdgeLevelOCL,
// KTemporalNROCL, KDebandOCL, KTemporalDiffOCL+KMergeStaticOCL): it loads the
// ALG-VERIFIED kernel of src/opencl/kfm/kernels/kfm_noiseclip.cl
// (kf_noise_clip, embedded into the DLL at build time) through the small host
// runtime in src/host/ocl_host.* and replays the upstream filter's host
// pipeline:
//
//   GetFrameT (upstream KNoiseClip::GetFrameT):
//     fetch child(n) (src) and noiseclip->GetFrame(n) (noise); then ONE
//     kf_noise_clip launch per plane (Y at luma dims with nmin_y/range_y,
//     U/V at chroma dims with nmin_uv/range_uv), exactly upstream's three
//     kl_noise_clip / cpu_noise_clip calls.  Per pixel:
//       s   = (src - noise + 256) >> 1        // 128 == equal
//       out = dev_limitter(s, nmin, range)    // {0,56,128,199,255}
//     (the kernel's kf_limitter is the dev_limitter twin; the CUDA uchar4
//     vectorisation is over 4 scalar pixels, so the scalar kernel is
//     bit-identical).
//
// Parameters (upstream KNoiseClip::Create, verbatim defaults):
//   noise (clip)                    same-resolution second clip (the noise
//                                   estimate; usually KAnalyzeNoise output)
//   nmin_y (int, default 1)         inner dead-band half width (below 128)
//   range_y (int, default 128)      outer dead-band half width (below 128)
//   nmin_uv (int, default 1)        chroma inner dead-band half width
//   range_uv (int, default 128)     chroma outer dead-band half width
//
// UPSTREAM KERNEL INVENTORY (audited for this plugin): the upstream KNoiseClip
// class launches exactly ONE device kernel, kl_noise_clip, three times per
// frame (Y, U, V) — no helper kernels, no reductions, no persistent device
// buffers.  The ported kf_noise_clip covers 100% of what upstream KNoiseClip
// launches.  (The other kernels in DecombeUCF.cu — kl_init_uint64 /
// kl_calculate_field_diff / kl_init_block_sum / kl_add_block_sum /
// kl_block_sum_max / kl_analyze_noise / kl_analyze_diff — belong to the
// KCFieldDiff / KCFrameDiffDup / KAnalyzeNoise classes, not to KNoiseClip.)
//
// STATE MODEL (stated like the KMergeStaticOCL precedent): the upstream
// KNoiseClip class holds NO persistent state between GetFrame calls — no
// accumulation, no previous-frame memory, no device work buffers (dst is a
// fresh NewVideoFrame every call).  Output frame n is a pure function of
// child(n) and noiseclip(n) plus the class's four fixed configuration ints.
// This plugin mirrors that: its device planes are per-call scratch (overwritten
// wholesale every GetFrame), so the filter is stateless.
//
// UPSTREAM REGISTRATION (DecombeUCF.cu verbatim):
//   env->AddFunction("KNoiseClip",
//       "cc[nmin_y]i[range_y]i[nmin_uv]i[range_uv]i", KNoiseClip::Create, 0);
// This DLL registers KNoiseClipOCL with that exact signature and the upstream
// name "KNoiseClip" as an alias.
//
// Documented deviations from the upstream host code (none change any pixel
// for the golden-pinned parameter space):
//   1. Upstream's CUDA path runs uchar4 vectors over width4 = width>>2
//      columns; its constructor therefore throws unless width%4==0 and
//      height%4==0.  The scalar OpenCL kernel is exact for ANY width, but this
//      port keeps the upstream %4 checks verbatim (faithful host config; the
//      goldens pin the scalar semantics, which equal the CUDA vector
//      semantics on %4 widths).
//   2. Device planes use pitch == width plain planes (host staging copies
//      from the frame's real pitch); the kernel guards every access with
//      x < width && y < height, so no border policy is needed.
//   3. dst is a separate device plane (upstream likewise writes a distinct
//      output frame).
//   4. Upstream's `GetDeviceTypes(src) & GetDeviceTypes(noise)` CUDA device
//      match check is meaningless on the OpenCL host (both clips go through
//      this plugin's one device queue), so it is not reproduced.
//
// The plugin is a plain DLL exporting AvisynthPluginInit3 (no import lib);
// it validates autoload + registration against the installed AviSynth+ 3.7.3
// in test_kfm_noiseclip.cpp.
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "avisynth.h"
#include "../ocl_host.h"
#include "kfm_noiseclip_cl.h"  // GENERATED from src/opencl/kfm/kernels/kfm_noiseclip.cl

namespace {

using oclhost::Buffer;
using oclhost::Context;
using oclhost::Program;

// One plain plane: a w*h device image, pitch == width (KDebandOCL Plane
// discipline).
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

class KNoiseClipOCL : public GenericVideoFilter {
    PClip noiseclip_;

    int nmin_y_, range_y_, nmin_uv_, range_uv_;
    int logUVx_, logUVy_;

    // device buffers (per-call scratch; the filter itself is stateless)
    Plane srcY_, srcU_, srcV_;
    Plane noiseY_, noiseU_, noiseV_;
    Plane dstY_, dstU_, dstV_;

    std::vector<uint8_t> stageY_;    // host staging (max plane, per plane kind)
    std::vector<uint8_t> stageUV_;
    std::mutex* mtx_;                // serialises the shared in-order queue

    Program& program() {
        // 8-bit only (upstream: ComponentSize==1 throws otherwise), so the
        // kernel program is only ever instantiated at PX=uchar.
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "kfm_noiseclip", kfm_noiseclip_cl_src, 1);
    }

    // one kf_noise_clip launch per plane (upstream: one kl_noise_clip /
    // cpu_noise_clip call per plane)
    void dispatchNoiseClip(Plane* dst, Plane* src, Plane* noise, int nmin, int range) {
        cl_kernel k = program().kernel("kf_noise_clip");
        oclhost::set_arg(k, 0, dst->mem());
        oclhost::set_arg(k, 1, src->mem());
        oclhost::set_arg(k, 2, noise->mem());
        oclhost::set_arg(k, 3, dst->w);
        oclhost::set_arg(k, 4, dst->h);
        oclhost::set_arg(k, 5, dst->w);  // pitch == width
        oclhost::set_arg(k, 6, nmin);
        oclhost::set_arg(k, 7, range);
        oclhost::dispatch2D(sharedContext(), k, dst->w, dst->h);
    }

public:
    KNoiseClipOCL(PClip src, PClip noise,
                  int nmin_y, int range_y, int nmin_uv, int range_uv,
                  IScriptEnvironment* env_)
        : GenericVideoFilter(src)
        , noiseclip_(noise)
        , nmin_y_(nmin_y), range_y_(range_y), nmin_uv_(nmin_uv), range_uv_(range_uv)
        , logUVx_(vi.GetPlaneWidthSubsampling(PLANAR_U))
        , logUVy_(vi.GetPlaneHeightSubsampling(PLANAR_U)) {
        if (!vi.IsPlanar() || !vi.IsYUV() || vi.IsYUVA())
            env_->ThrowError("[KNoiseClipOCL] planar YUV input required");
        // upstream constructor checks (verbatim semantics)
        if (vi.width & 3)
            env_->ThrowError("[KNoiseClipOCL]: width must be multiple of 4");
        if (vi.height & 3)
            env_->ThrowError("[KNoiseClipOCL]: height must be multiple of 4");
        const VideoInfo noisevi = noiseclip_->GetVideoInfo();
        if (vi.width != noisevi.width || vi.height != noisevi.height)
            env_->ThrowError("[KNoiseClipOCL]: src and noiseclip must be same resoluction");
        // (upstream's GetDeviceTypes device-match check is CUDA-only —
        // documented deviation 4)
        // upstream GetFrame dispatch: only ComponentSize 1 is implemented
        if (vi.ComponentSize() != 1)
            env_->ThrowError("[KNoiseClipOCL] Unsupported pixel format");

        static std::mutex mtx;  // one shared queue per process
        mtx_ = &mtx;

        try {
            Context& ctx = sharedContext();
            const int w = vi.width, h = vi.height;
            const int wUV = w >> logUVx_, hUV = h >> logUVy_;

            srcY_.alloc(ctx, w, h, 1);
            srcU_.alloc(ctx, wUV, hUV, 1);
            srcV_.alloc(ctx, wUV, hUV, 1);
            noiseY_.alloc(ctx, w, h, 1);
            noiseU_.alloc(ctx, wUV, hUV, 1);
            noiseV_.alloc(ctx, wUV, hUV, 1);
            dstY_.alloc(ctx, w, h, 1);
            dstU_.alloc(ctx, wUV, hUV, 1);
            dstV_.alloc(ctx, wUV, hUV, 1);
            stageY_.resize((size_t)w * h);
            stageUV_.resize((size_t)wUV * hUV);
            program();  // build the uchar instantiation up front (HOST_CONTRACT §1)
        } catch (const std::exception& e) {
            env_->ThrowError("[KNoiseClipOCL] OpenCL init failed: %s", e.what());
        }
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            return GetFrameT(n, env_);
        } catch (const std::exception& e) {
            env_->ThrowError("[KNoiseClipOCL] %s", e.what());
        }
        return PVideoFrame();
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) {
        // mirror upstream KNoiseClip::SetCacheHints (MT_NICE_FILTER); device
        // work is serialised through the shared in-order queue + plugin mutex.
        if (cachehints == CACHE_GET_MTMODE) return MT_NICE_FILTER;
        return 0;
    }

private:
    PVideoFrame GetFrameT(int n, IScriptEnvironment* env_) {
        const int w = vi.width, h = vi.height;
        const int wUV = w >> logUVx_, hUV = h >> logUVy_;
        const int rowsizeY = w;
        const int rowsizeUV = wUV;

        std::lock_guard<std::mutex> lock(*mtx_);

        PVideoFrame srcFrame = child->GetFrame(n, env_);
        PVideoFrame noiseFrame = noiseclip_->GetFrame(n, env_);

        // ---- upload the three src planes and the three noise planes ----
        copyFromFrame(srcFrame->GetReadPtr(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y),
                      rowsizeY, h, stageY_.data());
        srcY_.buf.write(stageY_.data(), 0, srcY_.bytes());
        copyFromFrame(noiseFrame->GetReadPtr(PLANAR_Y), noiseFrame->GetPitch(PLANAR_Y),
                      rowsizeY, h, stageY_.data());
        noiseY_.buf.write(stageY_.data(), 0, noiseY_.bytes());
        copyFromFrame(srcFrame->GetReadPtr(PLANAR_U), srcFrame->GetPitch(PLANAR_U),
                      rowsizeUV, hUV, stageUV_.data());
        srcU_.buf.write(stageUV_.data(), 0, srcU_.bytes());
        copyFromFrame(noiseFrame->GetReadPtr(PLANAR_U), noiseFrame->GetPitch(PLANAR_U),
                      rowsizeUV, hUV, stageUV_.data());
        noiseU_.buf.write(stageUV_.data(), 0, noiseU_.bytes());
        copyFromFrame(srcFrame->GetReadPtr(PLANAR_V), srcFrame->GetPitch(PLANAR_V),
                      rowsizeUV, hUV, stageUV_.data());
        srcV_.buf.write(stageUV_.data(), 0, srcV_.bytes());
        copyFromFrame(noiseFrame->GetReadPtr(PLANAR_V), noiseFrame->GetPitch(PLANAR_V),
                      rowsizeUV, hUV, stageUV_.data());
        noiseV_.buf.write(stageUV_.data(), 0, noiseV_.bytes());

        // ---- one launch per plane (Y with nmin_y/range_y at luma dims,
        //      U/V with nmin_uv/range_uv at chroma dims) ----
        dispatchNoiseClip(&dstY_, &srcY_, &noiseY_, nmin_y_, range_y_);
        dispatchNoiseClip(&dstU_, &srcU_, &noiseU_, nmin_uv_, range_uv_);
        dispatchNoiseClip(&dstV_, &srcV_, &noiseV_, nmin_uv_, range_uv_);

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

AVSValue __cdecl CreateKNoiseClipOCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        return new KNoiseClipOCL(
            args[0].AsClip(),       // src
            args[1].AsClip(),       // noise
            args[2].AsInt(1),       // nmin_y (upstream default 1)
            args[3].AsInt(128),     // range_y (upstream default 128)
            args[4].AsInt(1),       // nmin_uv (upstream default 1)
            args[5].AsInt(128),     // range_uv (upstream default 128)
            env);
    } catch (const std::exception& e) {
        env->ThrowError("[KNoiseClipOCL] %s", e.what());
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
        // exact upstream signature from DecombeUCF.cu
        env->AddFunction("KNoiseClipOCL", "cc[nmin_y]i[range_y]i[nmin_uv]i[range_uv]i",
                         CreateKNoiseClipOCL, 0);
        // upstream function name as an alias (upstream signature identical)
        env->AddFunction("KNoiseClip", "cc[nmin_y]i[range_y]i[nmin_uv]i[range_uv]i",
                         CreateKNoiseClipOCL, 0);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "KNoiseClip (OpenCL port of KFM/DecombeUCF.cu KNoiseClip, MIT; fifth host plugin of the OpenCL port)";
}
