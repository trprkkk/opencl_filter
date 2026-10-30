// ocl_host.h — minimal OpenCL 1.2 host runtime for the OpenCL port plugins.
//
// Implements the on-rig host contract (docs/HOST_CONTRACT.md §1/§2) and the
// measured driver facts of docs/RIG_HANDOFF_BRINGUP.md §9.3:
//   - program build options: -cl-std=CL1.2 plus
//     -cl-fp32-correctly-rounded-divide-sqrt (REQUIRED on this driver for
//     float-division parity with the CPU goldens; it does NOT un-fuse FMA);
//     -fmad=false is rejected by this driver, so it is never passed.  FMA
//     contraction is controlled per-source with `#pragma OPENCL FP_CONTRACT`
//     inside the .cl files (honoured for this program family).
//   - work-group queries are unreliable on this driver (constant 256), so
//     dispatches use nullptr local size and let the driver choose (the rig
//     harnesses were verified this way).
//   - sub-buffer interior origins are broken on this driver (silently dropped
//     accesses, §9.4): the host never creates sub-buffers.  Kernels that must
//     not touch a plane border (repair style 3x3 windows over a plain, unguarded
//     plane) are dispatched over the INTERIOR only via clEnqueueNDRangeKernel's
//     global-work offset (dispatch2DOffset) — the rig's own technique
//     (rig/dev_kfm1.py).  Plane buffers are plain zero-initialised w*h images.
//
// Buffer ABI (HOST_CONTRACT §2/§6.1): every multi-component buffer the host
// shares with a kernel uses the packed logical layout.  The KEdgeLevel slice
// here only moves PX planes (uchar/ushort), so no vector/triple buffers are
// needed yet; the helpers below are structured so MV buffers (short2 vectors,
// 12-byte packed {x,y,sad} VECTOR triples — never int3) can be added without
// changing the program/queue discipline.
//
// CL 1.2 API only (CL_TARGET_OPENCL_VERSION 120).
#pragma once

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace oclhost {

// All errors carry the failing OpenCL call / build log text.
class Error : public std::runtime_error {
public:
    explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

// Build options for every program (see header comment for the rig evidence).
inline const char* buildOptions() {
    return "-cl-std=CL1.2 -cl-fp32-correctly-rounded-divide-sqrt";
}

// ---------------------------------------------------------------------------
// Context: one device (GPU preferred), one context, one in-order queue.
// ---------------------------------------------------------------------------
class Context {
public:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    cl_device_id device() const { return device_; }
    cl_context context() const { return context_; }
    // In-order queue (0 flags): kernel ordering + host blocking calls are
    // enough for the frame pipelines; no event plumbing needed.
    cl_command_queue queue() const { return queue_; }
    const std::string& deviceName() const { return device_name_; }
    const std::string& platformName() const { return platform_name_; }

    void finish() const;

private:
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    std::string device_name_;
    std::string platform_name_;
};

// ---------------------------------------------------------------------------
// Buffer: RAII cl_mem.  Whole allocations only (no sub-buffers on this driver).
// ---------------------------------------------------------------------------
class Buffer {
public:
    Buffer() {}
    Buffer(Context& ctx, size_t bytes, bool zero = false);
    ~Buffer();
    Buffer(Buffer&& o) noexcept { *this = std::move(o); }
    Buffer& operator=(Buffer&& o) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    bool empty() const { return mem_ == nullptr; }
    cl_mem mem() const { return mem_; }
    size_t bytes() const { return bytes_; }

    // Host <-> device (blocking on the in-order queue).
    void write(const void* host, size_t offset, size_t bytes) const;
    void read(void* host, size_t offset, size_t bytes) const;
    // Fill with a repeating 32-bit pattern (host word order).
    void fill32(uint32_t pattern) const;

private:
    cl_mem mem_ = nullptr;
    size_t bytes_ = 0;
    cl_command_queue queue_ = nullptr;  // queue the buffer was created on
};

// ---------------------------------------------------------------------------
// Program: one .cl source unit at one -DPX instantiation, with a kernel cache.
// ---------------------------------------------------------------------------
class Program {
public:
    // `px_bytes` 1 -> -DPX=uchar -DPX_MAX=255, 2 -> -DPX=ushort -DPX_MAX=65535.
    Program(Context& ctx, const std::string& unit_name, const std::string& source,
            int px_bytes);
    ~Program();
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;

    cl_kernel kernel(const char* name);  // cached per name

private:
    Context* ctx_ = nullptr;
    std::string unit_name_;
    cl_program program_ = nullptr;
    std::map<std::string, cl_kernel> kernels_;
};

// Programs registry (HOST_CONTRACT §1 discipline, extensible to further
// kernel units): key (unit name, px bytes) -> built program.  Building the
// same unit twice at both PX instantiations is exactly the contract.
class ProgramRegistry {
public:
    static ProgramRegistry& instance();
    // Builds (or returns cached) program for `unit_name`/`px_bytes`.
    Program& get(Context& ctx, const std::string& unit_name,
                 const std::string& source, int px_bytes);

private:
    std::map<std::pair<std::string, int>, std::unique_ptr<Program>> programs_;
};

// ---------------------------------------------------------------------------
// Kernel argument helpers (set_arg mirrors clSetKernelArg by value types).
// ---------------------------------------------------------------------------
inline void set_arg(cl_kernel k, cl_uint i, cl_mem v) {
    cl_int e = clSetKernelArg(k, i, sizeof(cl_mem), &v);
    if (e != CL_SUCCESS) throw Error("clSetKernelArg(mem) failed: " + std::to_string(e));
}
inline void set_arg(cl_kernel k, cl_uint i, int v) {
    cl_int e = clSetKernelArg(k, i, sizeof(v), &v);
    if (e != CL_SUCCESS) throw Error("clSetKernelArg(int) failed: " + std::to_string(e));
}
inline void set_arg(cl_kernel k, cl_uint i, unsigned v) {
    cl_int e = clSetKernelArg(k, i, sizeof(v), &v);
    if (e != CL_SUCCESS) throw Error("clSetKernelArg(uint) failed: " + std::to_string(e));
}
inline void set_arg(cl_kernel k, cl_uint i, float v) {
    cl_int e = clSetKernelArg(k, i, sizeof(v), &v);
    if (e != CL_SUCCESS) throw Error("clSetKernelArg(float) failed: " + std::to_string(e));
}

// Dispatch helpers: global size = the exact extents; the ported kernels guard
// every access with `x < width && y < height`, and the rig runs passed with a
// driver-chosen local size (nullptr — local-size queries are unreliable here).
void dispatch1D(Context& ctx, cl_kernel k, int n);
void dispatch2D(Context& ctx, cl_kernel k, int width, int height);
// Interior-only dispatch: global work offset (x0,y0), extents (width,height).
// The gated kernels (x < width && y < height) still guard, and kernel-side
// (x,y) map to plane samples (x0+x, y0+y) through the plain index math.
void dispatch2DOffset(Context& ctx, cl_kernel k, int x0, int y0, int width, int height);
void dispatch3D(Context& ctx, cl_kernel k, int x, int y, int z);
// Fixed-local-size dispatches for the reduction kernels whose __local tree
// layout REQUIRES an exact work-group shape (the KFM 32x16 reduce family:
// tid = lx + ly*32 over a fixed 512-slot buffer; local-size queries are
// unreliable on this driver, but an EXPLICIT local size is honoured — the rig
// harnesses launched these shapes hundreds of times, RIG_HANDOFF §9.4).
// The global size is rounded UP to whole work-groups (OOB items contribute 0
// through the width/height guards).
void dispatch2DLocal(Context& ctx, cl_kernel k, int width, int height,
                     int lx, int ly);
void dispatch2DLocalOffset(Context& ctx, cl_kernel k, int x0, int y0,
                           int width, int height, int lx, int ly);

}  // namespace oclhost
