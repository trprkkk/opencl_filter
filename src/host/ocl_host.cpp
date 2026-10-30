// ocl_host.cpp — see ocl_host.h for the contract this implements.
#include "ocl_host.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace oclhost {

namespace {

void check(cl_int e, const char* what) {
    if (e != CL_SUCCESS) {
        throw Error(std::string(what) + " failed: " + std::to_string(e));
    }
}

std::string device_info_str(cl_device_id dev, cl_device_info param) {
    char buf[512] = {0};
    check(clGetDeviceInfo(dev, param, sizeof(buf) - 1, buf, nullptr), "clGetDeviceInfo");
    return std::string(buf);
}

}  // namespace

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------
Context::Context() {
    cl_uint nplat = 0;
    check(clGetPlatformIDs(0, nullptr, &nplat), "clGetPlatformIDs(count)");
    if (nplat == 0) throw Error("no OpenCL platform found");
    std::vector<cl_platform_id> plats(nplat);
    check(clGetPlatformIDs(nplat, plats.data(), nullptr), "clGetPlatformIDs");

    // Pick (platform, device): GPU preferred, vendor-ranked exactly like the
    // working rig harness (NVIDIA > AMD > Intel > others), CPU fallback.
    struct Pick { cl_platform_id p; cl_device_id d; int score; };
    std::vector<Pick> gpus;
    cl_device_id cpu_dev = nullptr;
    cl_platform_id cpu_plat = nullptr;
    for (cl_platform_id p : plats) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) == CL_SUCCESS && nd > 0) {
            std::vector<cl_device_id> devs(nd);
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr) == CL_SUCCESS) {
                for (cl_device_id d : devs) {
                    std::string v = device_info_str(d, CL_DEVICE_VENDOR);
                    std::string n = device_info_str(d, CL_DEVICE_NAME);
                    int score = 3;
                    if (v.find("NVIDIA") != std::string::npos ||
                        n.find("NVIDIA") != std::string::npos) score = 0;
                    else if (v.find("Advanced Micro") != std::string::npos ||
                             v.find("AMD") != std::string::npos) score = 1;
                    else if (v.find("Intel") != std::string::npos ||
                             n.find("Intel") != std::string::npos) score = 2;
                    gpus.push_back({p, d, score});
                }
            }
        }
        if (!cpu_dev) {
            cl_uint ncd = 0;
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_CPU, 1, &cpu_dev, &ncd) == CL_SUCCESS && ncd > 0) {
                cpu_plat = p;
            } else {
                cpu_dev = nullptr;
            }
        }
    }

    cl_platform_id plat = nullptr;
    if (!gpus.empty()) {
        // stable lowest score
        const Pick* best = &gpus[0];
        for (const Pick& g : gpus) if (g.score < best->score) best = &g;
        plat = best->p;
        device_ = best->d;
    } else if (cpu_dev) {
        plat = cpu_plat;
        device_ = cpu_dev;
    } else {
        throw Error("no OpenCL device found");
    }

    platform_ = plat;
    {
        char pbuf[512] = {0};
        check(clGetPlatformInfo(plat, CL_PLATFORM_NAME, sizeof(pbuf) - 1, pbuf, nullptr),
              "clGetPlatformInfo");
        platform_name_ = pbuf;
    }
    device_name_ = device_info_str(device_, CL_DEVICE_NAME);

    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, nullptr);
    if (!context_) throw Error("clCreateContext failed");
    queue_ = clCreateCommandQueue(context_, device_, 0, nullptr);
    if (!queue_) { clReleaseContext(context_); context_ = nullptr; throw Error("clCreateCommandQueue failed"); }
}

Context::~Context() {
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
}

void Context::finish() const { check(clFinish(queue_), "clFinish"); }

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------
Buffer::Buffer(Context& ctx, size_t bytes, bool zero) : bytes_(bytes) {
    cl_int e = CL_SUCCESS;
    if (zero) {
        std::vector<uint8_t> z(bytes, 0);
        mem_ = clCreateBuffer(ctx.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              bytes, z.data(), &e);
    } else {
        mem_ = clCreateBuffer(ctx.context(), CL_MEM_READ_WRITE, bytes, nullptr, &e);
    }
    if (e != CL_SUCCESS || !mem_) {
        mem_ = nullptr; bytes_ = 0;
        throw Error("clCreateBuffer(" + std::to_string(bytes) + " B) failed: " + std::to_string(e));
    }
    queue_ = ctx.queue();
}

Buffer::~Buffer() { if (mem_) clReleaseMemObject(mem_); }

Buffer& Buffer::operator=(Buffer&& o) noexcept {
    if (this != &o) {
        if (mem_) clReleaseMemObject(mem_);
        mem_ = o.mem_; bytes_ = o.bytes_; queue_ = o.queue_;
        o.mem_ = nullptr; o.bytes_ = 0; o.queue_ = nullptr;
    }
    return *this;
}

void Buffer::write(const void* host, size_t offset, size_t bytes) const {
    // Blocking write on the in-order queue: the host staging buffer can be
    // reused immediately afterwards.
    check(clEnqueueWriteBuffer(queue_, mem_, CL_TRUE, offset, bytes, host,
                               0, nullptr, nullptr), "clEnqueueWriteBuffer");
}

void Buffer::read(void* host, size_t offset, size_t bytes) const {
    check(clEnqueueReadBuffer(queue_, mem_, CL_TRUE, offset, bytes, host,
                              0, nullptr, nullptr), "clEnqueueReadBuffer");
}

void Buffer::fill32(uint32_t pattern) const {
    check(clEnqueueFillBuffer(queue_, mem_, &pattern, sizeof(pattern), 0, bytes_,
                              0, nullptr, nullptr), "clEnqueueFillBuffer");
}

// ---------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------
Program::Program(Context& ctx, const std::string& unit_name, const std::string& source,
                 int px_bytes) : ctx_(&ctx), unit_name_(unit_name) {
    const char* px = (px_bytes == 1) ? "uchar" : (px_bytes == 2 ? "ushort" : nullptr);
    const char* pxmax = (px_bytes == 1) ? "255" : (px_bytes == 2 ? "65535" : nullptr);
    if (!px) throw Error("unsupported -DPX instantiation: " + std::to_string(px_bytes));

    const char* src = source.c_str();
    size_t len = source.size();
    cl_int e = CL_SUCCESS;
    program_ = clCreateProgramWithSource(ctx.context(), 1, &src, &len, &e);
    if (e != CL_SUCCESS || !program_) throw Error("clCreateProgramWithSource failed: " + std::to_string(e));

    std::string opts = std::string(buildOptions()) + " -DPX=" + px + " -DPX_MAX=" + pxmax;
    cl_device_id dev = ctx.device();
    e = clBuildProgram(program_, 1, &dev, opts.c_str(), nullptr, nullptr);
    if (e != CL_SUCCESS) {
        size_t log_len = 0;
        std::string log;
        if (clGetProgramBuildInfo(program_, ctx.device(), CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_len) == CL_SUCCESS) {
            std::vector<char> buf(log_len + 1, 0);
            clGetProgramBuildInfo(program_, ctx.device(), CL_PROGRAM_BUILD_LOG, log_len, buf.data(), nullptr);
            log = buf.data();
        }
        clReleaseProgram(program_);
        program_ = nullptr;
        throw Error("build FAILED [" + unit_name_ + " -DPX=" + px + "]: " + log);
    }
}

Program::~Program() {
    for (auto& kv : kernels_) if (kv.second) clReleaseKernel(kv.second);
    if (program_) clReleaseProgram(program_);
}

cl_kernel Program::kernel(const char* name) {
    auto it = kernels_.find(name);
    if (it != kernels_.end()) return it->second;
    cl_int e = CL_SUCCESS;
    cl_kernel k = clCreateKernel(program_, name, &e);
    if (e != CL_SUCCESS || !k) throw Error("clCreateKernel(" + unit_name_ + ":" + name + ") failed: " + std::to_string(e));
    kernels_[name] = k;
    return k;
}

ProgramRegistry& ProgramRegistry::instance() {
    static ProgramRegistry r;
    return r;
}

Program& ProgramRegistry::get(Context& ctx, const std::string& unit_name,
                              const std::string& source, int px_bytes) {
    auto key = std::make_pair(unit_name, px_bytes);
    auto it = programs_.find(key);
    if (it != programs_.end()) return *it->second;
    auto p = std::make_unique<Program>(ctx, unit_name, source, px_bytes);
    Program& ref = *p;
    programs_.emplace(std::move(key), std::move(p));
    return ref;
}

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------
void dispatch1D(Context& ctx, cl_kernel k, int n) {
    size_t g = (size_t)n;
    check(clEnqueueNDRangeKernel(ctx.queue(), k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel(1D)");
}

void dispatch2D(Context& ctx, cl_kernel k, int width, int height) {
    size_t g[2] = {(size_t)width, (size_t)height};
    check(clEnqueueNDRangeKernel(ctx.queue(), k, 2, nullptr, g, nullptr, 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel(2D)");
}

void dispatch2DOffset(Context& ctx, cl_kernel k, int x0, int y0, int width, int height) {
    if (width <= 0 || height <= 0) return;  // empty interior
    size_t off[2] = {(size_t)x0, (size_t)y0};
    size_t g[2] = {(size_t)width, (size_t)height};
    check(clEnqueueNDRangeKernel(ctx.queue(), k, 2, off, g, nullptr, 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel(2D offset)");
}

void dispatch3D(Context& ctx, cl_kernel k, int x, int y, int z) {
    size_t g[3] = {(size_t)x, (size_t)y, (size_t)z};
    check(clEnqueueNDRangeKernel(ctx.queue(), k, 3, nullptr, g, nullptr, 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel(3D)");
}

void dispatch2DLocal(Context& ctx, cl_kernel k, int width, int height, int lx, int ly) {
    dispatch2DLocalOffset(ctx, k, 0, 0, width, height, lx, ly);
}

void dispatch2DLocalOffset(Context& ctx, cl_kernel k, int x0, int y0,
                           int width, int height, int lx, int ly) {
    if (width <= 0 || height <= 0) return;  // empty interior
    size_t off[2] = {(size_t)x0, (size_t)y0};
    size_t g[2] = {
        (size_t)((width + lx - 1) / lx * lx),
        (size_t)((height + ly - 1) / ly * ly),
    };
    size_t l[2] = {(size_t)lx, (size_t)ly};
    check(clEnqueueNDRangeKernel(ctx.queue(), k, 2, off, g, l, 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel(2D local)");
}

}  // namespace oclhost
