// kfm_deband_avs.cpp — AviSynth+ host plugin for the OpenCL port of KFM's
// KDeband (upstream: rigaya/AviSynthCUDAFilters KFM/KDeband.cu, MIT).
//
// The THIRD real AviSynth+ host plugin of the port (after KEdgeLevelOCL and
// KTemporalNROCL): it loads the ALG-VERIFIED kernel of
// src/opencl/kfm/kernels/kfm_deband.cl (kf_deband_reduce_banding, embedded
// into the DLL at build time) through the small host runtime in
// src/host/ocl_host.* and replays the upstream filter's host pipeline:
//
//   GetFrameT (upstream KDeband::GetFrameT):
//     ONE kf_deband_reduce_banding launch per plane (Y at luma dims, U/V at
//     chroma dims), exactly upstream's three cpu_reduce_banding /
//     launch_reduce_banding calls from the 3x2 sample_mode x blur_first
//     table.  The upstream table is a code-gen choice (template params); the
//     OpenCL kernel takes sample_mode/blur_first as runtime args — identical
//     arithmetic for all six entries (see the .cl header).
//
// Parameters (upstream KDeband::Create, verbatim defaults):
//   range (int, default 25)         sampling distance, host-clamped by
//                                   range_limited per pixel inside the kernel
//   thresh (float, default 1)       host-scaled via scaleParam
//                                   ((int)(thresh*(1<<(bits-8))+0.5f))
//   sample_mode (int, default 1)    0/1/2 (upstream: must be 0,1,2 — throw)
//   blur_first (bool, default false)
//
// UPSTREAM KERNEL INVENTORY (audited for this plugin): the upstream KDeband
// class launches exactly ONE device kernel, kl_reduce_banding, once per
// plane — no blur/merge pre-passes, no helper kernels.  The XorShift rand
// byte stream (CreateDebandRandom) is generated ON THE HOST in upstream, so
// it is generated on the host here too (identical algorithm, seed 0).  There
// is no kernel gap: the ported kf_deband_reduce_banding covers 100% of what
// upstream KDeband launches.  (kl_copy/kl_fill/kl_edgelevel*/kl_el_to444/
// kl_el_from444/kl_temporal_nr in the same .cu belong to KEdgeLevel /
// KTemporalNR / plugin glue, not to the KDeband filter class.)
//
// SHARED RAND BUFFER (upstream verbatim): CreateDebandRandom builds ONE
// buffer of width*height*2 bytes (LUMA dims, max_per_pixel=2, XorShift seed
// 0) in the constructor; GetFrameT passes THE SAME pointer to all three
// plane launches, each kernel indexing it with that plane's own
// rand_step = width*height.  For chroma planes the reads stay inside the
// first 2*widthUV*heightUV bytes — always in bounds of the luma-sized
// buffer.  This plugin does exactly that: one device rand buffer at luma
// dims, shared by the three dispatches.
//
// Documented deviations from the upstream host code (none change any pixel
// for the golden-pinned parameter space):
//   1. range > 127: upstream's random_range takes `char range`, so a larger
//      range would be silently TRUNCATED to a (possibly negative) byte; the
//      ported kernel's kf_random_range contract (and the goldens) pin
//      range <= 127.  This plugin throws for range < 0 or range > 127
//      instead of silently truncating — deterministic and documented; every
//      in-contract value behaves identically to upstream.
//   2. Device planes use pitch == width plain planes (host staging copies
//      from the frame's real pitch); every access is range_limited-guarded
//      inside the kernel, so no border policy is needed.
//   3. dst is a separate device plane (upstream likewise writes a distinct
//      output frame).
//
// The plugin is a plain DLL exporting AvisynthPluginInit3 (no import lib);
// it validates autoload + registration against the installed AviSynth+ 3.7.3
// in test_kfm_deband.cpp.
#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "avisynth.h"
#include "../ocl_host.h"
#include "kfm_deband_cl.h"  // GENERATED from src/opencl/kfm/kernels/kfm_deband.cl

namespace {

using oclhost::Buffer;
using oclhost::Context;
using oclhost::Program;

// upstream KDeband.cu scaleParam (verbatim semantics)
int scaleParam(float thresh, int pixelBits) {
    return (int)(thresh * (1 << (pixelBits - 8)) + 0.5f);
}

// upstream KDeband.cu XorShift (verbatim semantics) + CreateDebandRandom:
// seed 0, uint32 little-endian words streamed into a byte buffer of
// length = width*height*max_per_pixel (max_per_pixel = 2).
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

static std::vector<uint8_t> createDebandRandom(int width, int height, int seed) {
    const int max_per_pixel = 2;
    const int length = width * height * max_per_pixel;
    std::vector<uint8_t> rand_buf((size_t)length);

    XorShift rng(seed);

    size_t i = 0;
    for (; i + 4 <= (size_t)length; i += 4) {
        uint32_t r = rng.next();
        memcpy(&rand_buf[i], &r, 4);  // little-endian host order
    }
    if (i < (size_t)length) {
        uint32_t r = rng.next();
        memcpy(&rand_buf[i], &r, (size_t)length - (ptrdiff_t)i);
    }
    return rand_buf;
}

// One plain plane: a w*h device image, pitch == width (KEdgeLevelOCL/KTemporalNROCL
// Plane discipline).
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

class KDebandOCL : public GenericVideoFilter {
    int range_;
    int thresh_;  // host-scaled (scaleParam)
    int sample_mode_;
    bool blur_first_;
    int logUVx_, logUVy_;
    int bits_;
    int px_;  // ComponentSize

    // device buffers (output planes + the shared rand stream), named like
    // the upstream locals
    Plane srcY_, srcU_, srcV_;
    Plane dstY_, dstU_, dstV_;
    Buffer randBuf_;                 // width*height*2 bytes (LUMA dims, seed 0)

    std::vector<uint8_t> stageY_;    // host staging (max plane, per plane kind)
    std::vector<uint8_t> stageUV_;
    std::vector<uint8_t> randHost_;  // host-generated XorShift stream
    std::mutex* mtx_;                // serialises the shared in-order queue

    Program& program() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "kfm_deband", kfm_deband_cl_src, px_);
    }

    // one kf_deband_reduce_banding launch per plane (upstream: one
    // cpu_reduce_banding / launch_reduce_banding call per plane, same shared
    // rand pointer for all three)
    void dispatchReduceBanding(Plane* dst, Plane* src, cl_mem rand_mem,
                               int range, int thresh) {
        cl_kernel k = program().kernel("kf_deband_reduce_banding");
        oclhost::set_arg(k, 0, dst->mem());
        oclhost::set_arg(k, 1, src->mem());
        oclhost::set_arg(k, 2, rand_mem);
        oclhost::set_arg(k, 3, dst->w);
        oclhost::set_arg(k, 4, dst->h);
        oclhost::set_arg(k, 5, dst->w);  // pitch == width
        oclhost::set_arg(k, 6, range);
        oclhost::set_arg(k, 7, thresh);
        oclhost::set_arg(k, 8, sample_mode_);
        oclhost::set_arg(k, 9, blur_first_ ? 1 : 0);
        oclhost::dispatch2D(sharedContext(), k, dst->w, dst->h);
    }

public:
    KDebandOCL(PClip clip, int range, float thresh, int sample_mode, bool blur_first,
               IScriptEnvironment* env_)
        : GenericVideoFilter(clip)
        , range_(range)
        , thresh_(scaleParam(thresh, vi.BitsPerComponent()))
        , sample_mode_(sample_mode)
        , blur_first_(blur_first)
        , logUVx_(vi.GetPlaneWidthSubsampling(PLANAR_U))
        , logUVy_(vi.GetPlaneHeightSubsampling(PLANAR_U)) {
        if (!vi.IsPlanar() || !vi.IsYUV() || vi.IsYUVA())
            env_->ThrowError("[KDebandOCL] planar YUV input required");
        if (sample_mode_ != 0 && sample_mode_ != 1 && sample_mode_ != 2)
            env_->ThrowError("[KDebandOCL] sample_mode must be 0,1,2");
        // deviation 1: upstream would silently truncate a larger range to
        // `char` inside random_range; the kernel contract (and the goldens)
        // pin range <= 127, so the port throws instead.
        if (range_ < 0 || range_ > 127)
            env_->ThrowError("[KDebandOCL] range must be 0..127");
        bits_ = vi.BitsPerComponent();
        px_ = vi.ComponentSize();
        if (px_ != 1 && px_ != 2)
            env_->ThrowError("[KDebandOCL] Unsupported pixel format");

        static std::mutex mtx;  // one shared queue per process
        mtx_ = &mtx;

        try {
            Context& ctx = sharedContext();
            const int w = vi.width, h = vi.height;
            const int wUV = w >> logUVx_, hUV = h >> logUVy_;

            srcY_.alloc(ctx, w, h, px_);
            srcU_.alloc(ctx, wUV, hUV, px_);
            srcV_.alloc(ctx, wUV, hUV, px_);
            dstY_.alloc(ctx, w, h, px_);
            dstU_.alloc(ctx, wUV, hUV, px_);
            dstV_.alloc(ctx, wUV, hUV, px_);

            // upstream CreateDebandRandom(vi.width, vi.height, 0): ONE luma-
            // sized stream shared by every plane launch
            randHost_ = createDebandRandom(w, h, /*seed=*/0);
            randBuf_ = Buffer(ctx, randHost_.size(), /*zero=*/false);
            randBuf_.write(randHost_.data(), 0, randHost_.size());

            stageY_.resize((size_t)w * h * px_);
            stageUV_.resize((size_t)wUV * hUV * px_);
            program();  // build both PX instantiations up front (HOST_CONTRACT §1)
        } catch (const std::exception& e) {
            env_->ThrowError("[KDebandOCL] OpenCL init failed: %s", e.what());
        }
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            if (px_ == 1) return GetFrameT<uint8_t>(n, env_);
            return GetFrameT<uint16_t>(n, env_);
        } catch (const std::exception& e) {
            env_->ThrowError("[KDebandOCL] %s", e.what());
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
        const int rowsizeY = w * px_;
        const int rowsizeUV = wUV * px_;

        std::lock_guard<std::mutex> lock(*mtx_);

        PVideoFrame srcFrame = child->GetFrame(n, env_);

        // ---- upload the three source planes ----
        copyFromFrame(srcFrame->GetReadPtr(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y),
                      rowsizeY, h, stageY_.data());
        srcY_.buf.write(stageY_.data(), 0, srcY_.bytes());
        copyFromFrame(srcFrame->GetReadPtr(PLANAR_U), srcFrame->GetPitch(PLANAR_U),
                      rowsizeUV, hUV, stageUV_.data());
        srcU_.buf.write(stageUV_.data(), 0, srcU_.bytes());
        copyFromFrame(srcFrame->GetReadPtr(PLANAR_V), srcFrame->GetPitch(PLANAR_V),
                      rowsizeUV, hUV, stageUV_.data());
        srcV_.buf.write(stageUV_.data(), 0, srcV_.bytes());

        // ---- one launch per plane (Y at luma dims, U/V at chroma dims),
        //      sharing the single luma-dims rand buffer (upstream verbatim) ----
        dispatchReduceBanding(&dstY_, &srcY_, randBuf_.mem(), range_, thresh_);
        dispatchReduceBanding(&dstU_, &srcU_, randBuf_.mem(), range_, thresh_);
        dispatchReduceBanding(&dstV_, &srcV_, randBuf_.mem(), range_, thresh_);

        sharedContext().finish();

        // ---- download into a fresh output frame ----
        PVideoFrame dstFrame = env_->NewVideoFrame(vi);
        {
            dstY_.buf.read(stageY_.data(), 0, dstY_.bytes());
            copyToFrame(stageY_.data(), rowsizeY, rowsizeY, h,
                        dstFrame->GetWritePtr(PLANAR_Y), dstFrame->GetPitch(PLANAR_Y));
            dstU_.buf.read(stageUV_.data(), 0, dstU_.bytes());
            copyToFrame(stageUV_.data(), rowsizeUV, rowsizeUV, hUV,
                        dstFrame->GetWritePtr(PLANAR_U), dstFrame->GetPitch(PLANAR_U));
            dstV_.buf.read(stageUV_.data(), 0, dstV_.bytes());
            copyToFrame(stageUV_.data(), rowsizeUV, rowsizeUV, hUV,
                        dstFrame->GetWritePtr(PLANAR_V), dstFrame->GetPitch(PLANAR_V));
        }
        return dstFrame;
    }
};

AVSValue __cdecl CreateKDebandOCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        return new KDebandOCL(
            args[0].AsClip(),            // clip
            args[1].AsInt(25),           // range (upstream default 25)
            (float)args[2].AsFloat(1),   // thresh (upstream default 1)
            args[3].AsInt(1),            // sample_mode (upstream default 1)
            args[4].AsBool(false),       // blur_first (upstream default false)
            env);
    } catch (const std::exception& e) {
        env->ThrowError("[KDebandOCL] %s", e.what());
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
        env->AddFunction("KDebandOCL", "c[range]i[thresh]f[sample_mode]i[blur_first]b",
                         CreateKDebandOCL, 0);
        // upstream function name as an alias (upstream signature identical)
        env->AddFunction("KDeband", "c[range]i[thresh]f[sample_mode]i[blur_first]b",
                         CreateKDebandOCL, 0);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "KDeband (OpenCL port of KFM/KDeband.cu, MIT; third host plugin of the OpenCL port)";
}
