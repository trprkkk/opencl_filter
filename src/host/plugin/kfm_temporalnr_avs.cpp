// kfm_temporalnr_avs.cpp — AviSynth+ host plugin for the OpenCL port of KFM's
// KTemporalNR (upstream: rigaya/AviSynthCUDAFilters KFM/KDeband.cu, MIT).
//
// The SECOND real AviSynth+ host plugin of the port (after KEdgeLevelOCL): it
// loads the ALG-VERIFIED kernel of src/opencl/kfm/kernels/kfm_temporalnr.cl
// (kf_temporal_nr, embedded into the DLL at build time) through the small host
// runtime in src/host/ocl_host.* and replays the upstream filter's host
// pipeline:
//
//   GetFrame (upstream KTemporalNR::run / MakeFrames):
//     nframes = dist*2 + 1; fetch child frames at
//     clamp(n - dist + i, 0, vi.num_frames - 1) for i in [0, nframes) — the
//     clip-boundary clamp is upstream's (first/last frames reuse the edge
//     frame); mid = nframes/2 = dist; thresh host-scaled via scaleParam
//     ((int)(thresh*(1<<(bits-8))+0.5f)); then ONE kf_temporal_nr launch per
//     plane (Y at luma dims, U/V at chroma dims), exactly upstream's three
//     cpu_temporal_nr / kl_temporal_nr calls.
//
// CROSS-FRAME STATE MODEL (this is the port's first temporal filter, so it is
// stated explicitly): KTemporalNR holds NO persistent state between GetFrame
// calls — no previous-plane accumulation, no scene-change/reset logic.  The
// upstream class is a pure windowed filter: every output frame n is a function
// of the child clip's frames [n-dist .. n+dist] (clamped to the clip) and the
// parameters alone; the temporal "accumulation" is the sum over that window
// INSIDE one call, recomputed from scratch each time.  Consequently the
// "scene-change handling" of the upstream threshold gate is per-pixel and
// within-window only: a pixel contributes to the average iff its absolute
// difference from the window-centre pixel is <= thresh, so an abrupt content
// change simply falls out of the gate for the affected frames; no host-side
// state exists to reset.  (Ambiguity resolution per the task rules: the
// upstream class quoted above IS the authority — there is nothing else to
// mirror; the repo CPU golden sim/kfm_temporalnr_ref.cpp /
// python/run_kfm_temporalnr.py pins exactly this windowed semantics.)
//
// Kernel layout seam (documented in kfm_temporalnr.cl + KFM_PORT_SPEC.md): the
// CUDA host launches per plane passing a TemporalNRPtrs struct of nframes
// separate pointers; the OpenCL kernel takes the nframes planes PACKED into
// one buffer (plane i at ELEMENT offset i*frame_stride — see deviation 4).
// This plugin packs each plane group's fetched frames into one device buffer
// with frame_stride = width*height*px BYTES host-side (plain pitch==width
// planes, same Plane discipline as KEdgeLevelOCL) and launches once per plane
// group, passing the stride in elements.  Per-pixel arithmetic is identical.
//
// Documented deviations from the upstream host code (none change any pixel):
//   1. Upstream's CUDA path only ever processes full 4-wide vectors
//      (width4 = width/vecsize with vecsize=4 when width%8==0), leaving the
//      width%4 trailing columns UNWRITTEN (garbage in the CUDA output frame);
//      its scalar path (width%8 != 0) processes every column.  The scalar
//      OpenCL kernel processes every column always — the self-consistent
//      reading of the upstream scalar/CPU path, and the goldens pin it.
//   2. Device buffers use pitch == width plain planes (host staging copies
//      from the frame's real pitch); the kernel is launched once per plane
//      group over the whole plane (every access guarded), so no border policy
//      is needed — the kernel writes only in-range pixels.
//   3. dst is a separate device plane (not aliased into the packed frames
//      buffer); the upstream out pointer is likewise a distinct frame.
//   4. HOST-CONTRACT ISSUE FOUND (reported, kernel untouched — goldens decide):
//      kfm_temporalnr.cl's header comment says plane i sits at "byte offset
//      i*frame_stride", but the kernel indexes a `__global const PX*` —
//      POINTER arithmetic in ELEMENTS — exactly like the repo goldens
//      (python/run_kfm_temporalnr.py golden + sim mirror index
//      frames[i*frame_stride + idx] with frame_stride = pitch*height in
//      ELEMENTS).  With PX=uchar elements == bytes, so any byte-stride host
//      passes 8-bit by coincidence; at 16-bit a byte stride reads plane 2i
//      (empirically pinned: output == the 2*dist'th frame's pixel).  The
//      kernel arithmetic is ALG-VERIFIED and untouched; this host passes
//      frame_stride in ELEMENTS (width*height), which is the only
//      golden-consistent reading.  The .cl comment's "byte offset" wording is
//      the discrepancy to fix upstream of this plugin.
//
// The plugin is a plain DLL exporting AvisynthPluginInit3 (no import lib); it
// validates autoload + registration against the installed AviSynth+ 3.7.3 in
// test_kfm_temporalnr.cpp.
#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "avisynth.h"
#include "../ocl_host.h"
#include "kfm_temporalnr_cl.h"  // GENERATED from src/opencl/kfm/kernels/kfm_temporalnr.cl

namespace {

using oclhost::Buffer;
using oclhost::Context;
using oclhost::Program;

// upstream KDeband.cu scaleParam (verbatim semantics)
int scaleParam(float thresh, int pixelBits) {
    return (int)(thresh * (1 << (pixelBits - 8)) + 0.5f);
}

// A packed group of nframes identical-layout planes in ONE device buffer:
// plane i at byte offset i * frame_stride, each plane a plain w*h image with
// pitch == width (KEdgeLevelOCL Plane discipline, extended by the frame dim).
struct FrameGroup {
    Buffer buf;
    int w = 0, h = 0, px = 1;       // per-plane dims
    int nframes = 0;
    size_t frame_stride = 0;        // bytes between planes (w*h*px)

    void alloc(Context& ctx, int w_, int h_, int px_, int nframes_) {
        w = w_; h = h_; px = px_; nframes = nframes_;
        frame_stride = (size_t)w * h * px;
        buf = Buffer(ctx, frame_stride * nframes, /*zero=*/false);
    }
    cl_mem mem() const { return buf.mem(); }
};

// One output plane: plain w*h device image, pitch == width.
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

// Copies a frame plane (arbitrary source pitch) into the packed group's
// plane i (pitch == w).
void uploadPlane(FrameGroup& g, int i, const uint8_t* srcp, int srcPitch,
                 int rowsize, int height, std::vector<uint8_t>& stage) {
    for (int y = 0; y < height; ++y)
        memcpy(stage.data() + (size_t)y * rowsize, srcp + (size_t)y * srcPitch, rowsize);
    g.buf.write(stage.data(), g.frame_stride * i, g.frame_stride);
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

class KTemporalNROCL : public GenericVideoFilter {
    int dist_;
    int thresh_;  // host-scaled (scaleParam)
    int logUVx_, logUVy_;
    int bits_;
    int px_;  // ComponentSize

    // device buffers (packed frame groups + output planes), named like the
    // upstream run() locals
    FrameGroup framesY_, framesU_, framesV_;
    Plane dstY_, dstU_, dstV_;

    std::vector<uint8_t> stageY_;   // host staging (max plane, per group kind)
    std::vector<uint8_t> stageUV_;
    std::mutex* mtx_;               // serialises the shared in-order queue

    Program& program() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "kfm_temporalnr", kfm_temporalnr_cl_src, px_);
    }

    // one kf_temporal_nr launch per plane group (upstream: one kl_temporal_nr /
    // cpu_temporal_nr call per plane)
    void dispatchTemporalNR(Plane* dst, FrameGroup* frames, int thresh) {
        cl_kernel k = program().kernel("kf_temporal_nr");
        oclhost::set_arg(k, 0, dst->mem());
        oclhost::set_arg(k, 1, frames->mem());
        oclhost::set_arg(k, 2, dst->w);
        oclhost::set_arg(k, 3, dst->h);
        oclhost::set_arg(k, 4, dst->w);                 // pitch == width
        oclhost::set_arg(k, 5, dst->w * dst->h);        // frame_stride in ELEMENTS
        oclhost::set_arg(k, 6, frames->nframes);
        oclhost::set_arg(k, 7, frames->nframes / 2);    // mid = dist
        oclhost::set_arg(k, 8, thresh);
        oclhost::dispatch2D(sharedContext(), k, dst->w, dst->h);
    }

public:
    KTemporalNROCL(PClip clip, int dist, float thresh, IScriptEnvironment* env_)
        : GenericVideoFilter(clip)
        , thresh_(scaleParam(thresh, vi.BitsPerComponent()))
        , logUVx_(vi.GetPlaneWidthSubsampling(PLANAR_U))
        , logUVy_(vi.GetPlaneHeightSubsampling(PLANAR_U)) {
        if (!vi.IsPlanar() || !vi.IsYUV() || vi.IsYUVA())
            env_->ThrowError("[KTemporalNROCL] planar YUV input required");
        dist_ = dist;
        if (dist_ > 16)  // upstream MAX_TEMPORAL_DIST
            env_->ThrowError("[KTemporalNROCL] maximum dist is 16");
        bits_ = vi.BitsPerComponent();
        px_ = vi.ComponentSize();
        if (px_ != 1 && px_ != 2)
            env_->ThrowError("[KTemporalNROCL] Unsupported pixel format");

        static std::mutex mtx;  // one shared queue per process
        mtx_ = &mtx;

        try {
            Context& ctx = sharedContext();
            const int w = vi.width, h = vi.height;
            const int wUV = w >> logUVx_, hUV = h >> logUVy_;
            const int nframes = dist_ * 2 + 1;

            framesY_.alloc(ctx, w, h, px_, nframes);
            framesU_.alloc(ctx, wUV, hUV, px_, nframes);
            framesV_.alloc(ctx, wUV, hUV, px_, nframes);
            dstY_.alloc(ctx, w, h, px_);
            dstU_.alloc(ctx, wUV, hUV, px_);
            dstV_.alloc(ctx, wUV, hUV, px_);
            stageY_.resize((size_t)w * h * px_);
            stageUV_.resize((size_t)wUV * hUV * px_);
            program();  // build both PX instantiations up front (HOST_CONTRACT §1)
        } catch (const std::exception& e) {
            env_->ThrowError("[KTemporalNROCL] OpenCL init failed: %s", e.what());
        }
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            if (px_ == 1) return GetFrameT<uint8_t>(n, env_);
            return GetFrameT<uint16_t>(n, env_);
        } catch (const std::exception& e) {
            env_->ThrowError("[KTemporalNROCL] %s", e.what());
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
        const int nframes = dist_ * 2 + 1;

        std::lock_guard<std::mutex> lock(*mtx_);

        // ---- fetch the clamped temporal window (upstream verbatim) ----
        // frames[i] = child->GetFrame(clamp(n - dist + i, 0, vi.num_frames - 1))
        for (int i = 0; i < nframes; ++i) {
            const int fn = std::max(0, std::min(n - dist_ + i, vi.num_frames - 1));
            PVideoFrame f = child->GetFrame(fn, env_);
            uploadPlane(framesY_, i, f->GetReadPtr(PLANAR_Y), f->GetPitch(PLANAR_Y),
                        rowsizeY, h, stageY_);
            uploadPlane(framesU_, i, f->GetReadPtr(PLANAR_U), f->GetPitch(PLANAR_U),
                        rowsizeUV, hUV, stageUV_);
            uploadPlane(framesV_, i, f->GetReadPtr(PLANAR_V), f->GetPitch(PLANAR_V),
                        rowsizeUV, hUV, stageUV_);
        }

        // ---- one launch per plane (Y at luma dims, U/V at chroma dims) ----
        dispatchTemporalNR(&dstY_, &framesY_, thresh_);
        dispatchTemporalNR(&dstU_, &framesU_, thresh_);
        dispatchTemporalNR(&dstV_, &framesV_, thresh_);

        sharedContext().finish();

        // ---- download into a fresh output frame ----
        PVideoFrame dstFrame = env_->NewVideoFrame(vi);
        {
            // blocking reads through staging (in-order queue)
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

AVSValue __cdecl CreateKTemporalNROCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        return new KTemporalNROCL(
            args[0].AsClip(),            // clip
            args[1].AsInt(3),            // dist (upstream default 3, max 16)
            (float)args[2].AsFloat(1),   // thresh (upstream default 1)
            env);
    } catch (const std::exception& e) {
        env->ThrowError("[KTemporalNROCL] %s", e.what());
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
        env->AddFunction("KTemporalNROCL", "c[dist]i[thresh]f",
                         CreateKTemporalNROCL, 0);
        // upstream function name as an alias (upstream signature identical)
        env->AddFunction("KTemporalNR", "c[dist]i[thresh]f",
                         CreateKTemporalNROCL, 0);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "KTemporalNR (OpenCL port of KFM/KDeband.cu, MIT; second host plugin of the OpenCL port)";
}
