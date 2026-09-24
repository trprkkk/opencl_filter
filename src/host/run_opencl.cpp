// OpenCL bring-up smoke test for the KTGMC kernel ports.
//
// STATUS: COMPILE-CHECKED ONLY (-fsyntax-only against Khronos OpenCL-Headers;
// no OpenCL ICD in this sandbox, so never linked or run). Its job on a rig is
// only the first bring-up
// step: pick a device, build both kernel sources at both -DPX instantiations
// (uchar and ushort), and report whether every program compiles. Per-kernel
// numeric dispatch is specified in docs/HOST_CONTRACT.md and should be added
// after this smoke test passes and the int2/int3 buffer-layout probe has run.
//
// Build (from build dir): cmake .. && make ktgmc_opencl_host
// Requires ocl-icd-opencl-dev (headers + loader) + a device ICD (vendor or
// pocl-opencl-icd). Exit code 0 == all programs built.
// 1.2 API only; the 120 target keeps queue creation un-deprecated.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string readFile(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> b(n); std::fread(b.data(), 1, (size_t)n, f); std::fclose(f);
    return std::string(b.data(), (size_t)n);
}

static void check(cl_int e, const char* what) {
    if (e != CL_SUCCESS) { std::fprintf(stderr, "OpenCL error %d at %s\n", e, what); std::exit(3); }
}

static cl_program buildProgram(cl_context ctx, cl_device_id dev, const char* srcPath,
                               const char* px, const char* pxmax) {
    std::string src = readFile(srcPath);
    const char* cstr = src.c_str();
    size_t len = src.size();
    cl_int e;
    cl_program p = clCreateProgramWithSource(ctx, 1, &cstr, &len, &e);
    check(e, "clCreateProgramWithSource");
    std::string opts = std::string("-cl-std=CL1.2 -DPX=") + px + " -DPX_MAX=" + pxmax;
    e = clBuildProgram(p, 1, &dev, opts.c_str(), nullptr, nullptr);
    if (e != CL_SUCCESS) {
        size_t logLen = 0;
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLen);
        std::vector<char> log(logLen + 1);
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, logLen, log.data(), nullptr);
        std::fprintf(stderr, "BUILD FAILED: %s (-DPX=%s):\n%s\n", srcPath, px, log.data());
        std::exit(4);
    }
    return p;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <simple.cl> <motion.cl>\n", argv[0]);
        return 2;
    }
    const char* simplePath = argv[1];
    const char* motionPath = argv[2];

    cl_uint nplat = 0;
    cl_int e = clGetPlatformIDs(0, nullptr, &nplat);
    check(e, "clGetPlatformIDs");
    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);

    cl_device_id dev = nullptr;
    char name[256] = "?";
    for (cl_platform_id p : plats) {
        cl_uint nd = 0;
        cl_device_id d = nullptr;
        // Prefer a GPU device, fall back to CPU.
        e = clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, &nd);
        if (e != CL_SUCCESS || nd == 0) {
            e = clGetDeviceIDs(p, CL_DEVICE_TYPE_CPU, 1, &d, &nd);
        }
        if (e == CL_SUCCESS && nd > 0) { dev = d; break; }
    }
    if (!dev) { std::fprintf(stderr, "no OpenCL device found\n"); return 5; }
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    std::printf("device: %s\n", name);

    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e);
    check(e, "clCreateContext");
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &e);
    check(e, "clCreateCommandQueue");

    // Build each source at both PX instantiations (compile coverage only).
    cl_program simpleU8  = buildProgram(ctx, dev, simplePath, "uchar", "255");
    cl_program simpleU16 = buildProgram(ctx, dev, simplePath, "ushort", "65535");
    cl_program motionU8  = buildProgram(ctx, dev, motionPath, "uchar", "255");
    cl_program motionU16 = buildProgram(ctx, dev, motionPath, "ushort", "65535");

    std::printf("all 4 programs built (simple uchar/ushort, motion uchar/ushort)\n");
    std::printf("smoke OK. Next: run the int2/int3 sizeof probe and per-kernel "
                "diff per docs/HOST_CONTRACT.md.\n");

    clReleaseCommandQueue(q);
    clReleaseProgram(simpleU8); clReleaseProgram(simpleU16);
    clReleaseProgram(motionU8); clReleaseProgram(motionU16);
    clReleaseContext(ctx);
    return 0;
}
