// OpenCL host harness for the two Conditional float RIG-VERIFY kernels
// (ka_sum_pixels_f32 / ka_sad_f32 in avscuda_conditional_rig.cl).
//
// STATUS: COMPILE-CHECKED ONLY. -fsyntax-only against Khronos OpenCL-Headers
// in a sandbox with no OpenCL ICD: never linked, never run. First duty on a
// rig is to build (cmake finds it when OpenCL exists) and run the A1 smoke
// below; a wrong counter there is a harness bug, not a port bug.
//
// WHAT THIS IS: the OpenCL-side half of docs/RIG_HANDOFF_AVSCUDA_CONDITIONAL.md
// §5. The CUDA twin harness (cond_cuda.cu, another agent's job) must replicate
// fill_plane() EXACTLY (same float expressions, same stream order) so both
// sides consume bit-identical inputs; the comparison criteria live in the
// handoff (§5.1/§5.2), not here. This program only launches faithfully (§4.4)
// and prints raw counter bits + the host-side f64 reference per run.
//
// Launch (handoff §4.4, all load-bearing, all honoured here):
//   local (16,16) [also enforced by reqd_work_group_size in the rig file],
//   global (round16(w), round16(h)), width mult-4 (else exit 6), pitch in
//   float elements, counter (re)written by the host before every rep,
//   single program build (-DPX=uchar; the kernels are PX-independent).
//
// Usage:
//   avscuda_rig_host <rig.cl> <sum|sad> <W> <H> --fill0 <name> [opts]
//     --fill1 <name>  second plane (sad only; REQUIRED for sad)
//     --pitch P       row stride in floats (default W; slack filled with
//                     canonical quiet NaN so stray reads scream — §5.3)
//     --seed0 S       xorshift32 seed, plane 0 (default 0x12345678, nonzero)
//     --seed1 S       xorshift32 seed, plane 1 (default seed0^0x9E3779B9)
//     --maxv M        default 255 (float path ignores it; §5.3 tries 0/INT_MAX)
//     --prefill F     counter value before each rep (default 0; §5.3 uses 42)
//     --reps N        runs (default 1; Phase A wants >=5, Phase B >=10)
//     --std 1.1|1.2   OpenCL C version flag (default 1.2)
//   Single-plane fills (applied row-major over VALID pixels; pitch slack
//   always NaN; every expression below is IEEE-754 binary32, round-to-nearest,
//   and must be transcribed verbatim into the CUDA twin):
//     zeros            0.0f
//     ones             1.0f
//     two              2.0f
//     half             0.5f
//     const:F          strtof(F) (e.g. const:3.0, const:1e10)
//     checker          ((x+y)&1) ? -1.0f : 1.0f          (A6; exact 0 on even N)
//     zerosmix         ((x+y)&1) ? -0.0f : 0.0f          (A7)
//     nan1             ordinal 0 -> 0x7fc00000, else 1.0f (A8)
//     rand01           (float)(xs&0xFFFFFF)/16777216.0f  (exact dyadic [0,1))
//     rand11           (float)((int)(xs&0xFFFFFF)-8388608)/8388608.0f (exact)
//     big              rand01-value * 1e10f              (correctly rounded)
//     small            rand01-value * 1e-10f             (correctly rounded)
//     mixed            even ordinal -> big, odd -> small (one xs draw/pixel)
//     altpm1           (ordinal&1) ? -1.0f : 1.0f        (near-zero total)
//     video            0.45f*(x/W+y/H) + 0.1f*rand01-value (gradient + noise)
//   xs = xorshift32 stream (s^=s<<13; s^=s>>17; s^=s<<5), one draw per valid
//   pixel in row-major ordinal order, reseeded per plane (seed0/seed1).
//   Handoff config map: A1-A5/A7 = sum + zeros/ones/two/half/zerosmix at the
//   §5.1 shapes; A6 = checker; A8 = nan1; A9 = sad const:3.0/const:1.0;
//   A10 = sad rand01/rand01 with --seed1 == --seed0 (identical planes).
//
// Output (stdout, one header block + one line per rep; parse with awk):
//   # ... config echo (effective seeds, build opts, device) ...
//   # f64ref=<hexfloat> (<dec>) N=<valid px> maxabs=<hexfloat> c2bound=<e>
//   run <i>: bits=0x........ val=<hexfloat> (<dec>) mean_f64=<hexfloat>
// f64ref = serial double accumulation over valid pixels in row-major order
// (sad: of fabsf(a-b)); c2bound = 4*N*2^-24*maxabs (handoff criterion 2).
//
// Build on a rig with an ICD (headers + loader + device ICD):
//   cmake -S . -B build && cmake --build build --target avscuda_rig_host
// A1 smoke: ./build/avscuda_rig_host src/opencl/avscuda/kernels/
//   avscuda_conditional_rig.cl sum 4 1 --fill0 ones --reps 5
//   (expect bits=0x40800000 i.e. 4.0f on all 5 runs, or the harness is broken)
// 1.2 API only; the 120 target keeps queue creation un-deprecated.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string readFile(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> b((size_t)n);
    if (std::fread(b.data(), 1, (size_t)n, f) != (size_t)n) {
        std::fprintf(stderr, "short read %s\n", path); std::exit(2);
    }
    std::fclose(f);
    return std::string(b.data(), (size_t)n);
}

void check(cl_int e, const char* what) {
    if (e != CL_SUCCESS) {
        std::fprintf(stderr, "OpenCL error %d at %s\n", (int)e, what);
        std::exit(3);
    }
}

uint32_t xs_next(uint32_t& s) { // xorshift32; s must stay nonzero
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

float qnan() {
    uint32_t n = 0x7fc00000u;
    float f;
    std::memcpy(&f, &n, 4);
    return f;
}

// Fill one plane per the header catalog. Returns false on unknown fill name.
// Stream order: row-major over VALID pixels only (ordinal = y*width+x);
// pitch slack is poisoned with quiet NaN afterwards.
bool fill_plane(std::vector<float>& plane, int width, int height, int pitch,
                const char* fill, uint32_t seed) {
    enum Kind {
        K_ZEROS, K_ONES, K_TWO, K_HALF, K_CONST, K_CHECKER, K_ZEROSMIX, K_NAN1,
        K_RAND01, K_RAND11, K_BIG, K_SMALL, K_MIXED, K_ALTPM1, K_VIDEO
    } kind;
    float cval = 0.0f;
    if (!std::strcmp(fill, "zeros")) kind = K_ZEROS;
    else if (!std::strcmp(fill, "ones")) kind = K_ONES;
    else if (!std::strcmp(fill, "two")) kind = K_TWO;
    else if (!std::strcmp(fill, "half")) kind = K_HALF;
    else if (!std::strncmp(fill, "const:", 6)) {
        kind = K_CONST;
        char* end = nullptr;
        cval = std::strtof(fill + 6, &end);
        if (end == fill + 6 || *end != '\0') return false;
    }
    else if (!std::strcmp(fill, "checker")) kind = K_CHECKER;
    else if (!std::strcmp(fill, "zerosmix")) kind = K_ZEROSMIX;
    else if (!std::strcmp(fill, "nan1")) kind = K_NAN1;
    else if (!std::strcmp(fill, "rand01")) kind = K_RAND01;
    else if (!std::strcmp(fill, "rand11")) kind = K_RAND11;
    else if (!std::strcmp(fill, "big")) kind = K_BIG;
    else if (!std::strcmp(fill, "small")) kind = K_SMALL;
    else if (!std::strcmp(fill, "mixed")) kind = K_MIXED;
    else if (!std::strcmp(fill, "altpm1")) kind = K_ALTPM1;
    else if (!std::strcmp(fill, "video")) kind = K_VIDEO;
    else return false;

    uint32_t s = seed;
    const float nan = qnan();
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int ordinal = y * width + x;
            float v = 0.0f;
            switch (kind) {
            case K_ZEROS: v = 0.0f; break;
            case K_ONES: v = 1.0f; break;
            case K_TWO: v = 2.0f; break;
            case K_HALF: v = 0.5f; break;
            case K_CONST: v = cval; break;
            case K_CHECKER: v = ((x + y) & 1) ? -1.0f : 1.0f; break;
            case K_ZEROSMIX: v = ((x + y) & 1) ? -0.0f : 0.0f; break;
            case K_NAN1: v = (ordinal == 0) ? nan : 1.0f; break;
            case K_RAND01:
                v = (float)(xs_next(s) & 0xFFFFFFu) / 16777216.0f; break;
            case K_RAND11:
                v = (float)((int)(xs_next(s) & 0xFFFFFFu) - 8388608)
                    / 8388608.0f; break;
            case K_BIG:
                v = ((float)(xs_next(s) & 0xFFFFFFu) / 16777216.0f) * 1e10f;
                break;
            case K_SMALL:
                v = ((float)(xs_next(s) & 0xFFFFFFu) / 16777216.0f) * 1e-10f;
                break;
            case K_MIXED: {
                float u = (float)(xs_next(s) & 0xFFFFFFu) / 16777216.0f;
                v = (ordinal & 1) ? u * 1e-10f : u * 1e10f;
                break;
            }
            case K_ALTPM1: v = (ordinal & 1) ? -1.0f : 1.0f; break;
            case K_VIDEO: {
                float gx = (float)x / (float)width;
                float gy = (float)y / (float)height;
                float u = (float)(xs_next(s) & 0xFFFFFFu) / 16777216.0f;
                v = 0.45f * (gx + gy) + 0.1f * u;
                break;
            }
            }
            plane[(size_t)y * (size_t)pitch + (size_t)x] = v;
        }
        for (int x = width; x < pitch; ++x) // poison the slack
            plane[(size_t)y * (size_t)pitch + (size_t)x] = nan;
    }
    return true;
}

void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s <rig.cl> <sum|sad> <W> <H> --fill0 <name> [--fill1 <name>]\n"
        "       [--pitch P] [--seed0 S] [--seed1 S] [--maxv M] [--prefill F]\n"
        "       [--reps N] [--std 1.1|1.2]\n"
        "fills: zeros ones two half const:F checker zerosmix nan1 rand01 rand11\n"
        "       big small mixed altpm1 video  (see file header for definitions)\n",
        argv0);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 6) { usage(argv[0]); return 2; }
    const char* clPath = argv[1];
    const bool isSum = !std::strcmp(argv[2], "sum");
    const bool isSad = !std::strcmp(argv[2], "sad");
    if (!isSum && !isSad) { usage(argv[0]); return 2; }
    const int width = std::atoi(argv[3]);
    const int height = std::atoi(argv[4]);
    if (width < 1 || height < 1) {
        std::fprintf(stderr, "W and H must be >= 1\n"); return 6;
    }
    if (width % 4 != 0) { // handoff §4.4-3: uncharted upstream otherwise
        std::fprintf(stderr, "width must be a multiple of 4 (got %d)\n", width);
        return 6;
    }

    const char* fill0 = nullptr;
    const char* fill1 = nullptr;
    int pitch = width;
    uint32_t seed0 = 0x12345678u;
    uint32_t seed1 = 0; // 0 = derive from seed0 below
    bool seed1given = false;
    int maxv = 255;
    float prefill = 0.0f;
    int reps = 1;
    const char* stdver = "1.2";
    for (int i = 5; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--fill0") && i + 1 < argc) fill0 = argv[++i];
        else if (!std::strcmp(argv[i], "--fill1") && i + 1 < argc) fill1 = argv[++i];
        else if (!std::strcmp(argv[i], "--pitch") && i + 1 < argc) pitch = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed0") && i + 1 < argc)
            seed0 = (uint32_t)std::strtoul(argv[++i], nullptr, 0);
        else if (!std::strcmp(argv[i], "--seed1") && i + 1 < argc) {
            seed1 = (uint32_t)std::strtoul(argv[++i], nullptr, 0);
            seed1given = true;
        }
        else if (!std::strcmp(argv[i], "--maxv") && i + 1 < argc)
            maxv = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--prefill") && i + 1 < argc)
            prefill = std::strtof(argv[++i], nullptr);
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--std") && i + 1 < argc) stdver = argv[++i];
        else { usage(argv[0]); return 2; }
    }
    if (!fill0) { usage(argv[0]); return 2; }
    if (isSum && fill1) {
        std::fprintf(stderr, "sum takes --fill0 only\n"); return 6;
    }
    if (isSad && !fill1) {
        std::fprintf(stderr, "sad requires --fill1\n"); return 6;
    }
    if (pitch < width) {
        std::fprintf(stderr, "pitch (%d) < width (%d)\n", pitch, width);
        return 6;
    }
    if (seed0 == 0) {
        std::fprintf(stderr, "seed0 must be nonzero\n"); return 6;
    }
    if (!seed1given) seed1 = seed0 ^ 0x9E3779B9u;
    if (isSad && seed1 == 0) {
        std::fprintf(stderr, "seed1 must be nonzero\n"); return 6;
    }
    if (reps < 1) {
        std::fprintf(stderr, "reps must be >= 1\n"); return 6;
    }
    if (std::strcmp(stdver, "1.1") && std::strcmp(stdver, "1.2")) {
        std::fprintf(stderr, "--std must be 1.1 or 1.2\n"); return 6;
    }

    std::vector<float> plane0((size_t)pitch * (size_t)height);
    if (!fill_plane(plane0, width, height, pitch, fill0, seed0)) {
        std::fprintf(stderr, "unknown --fill0 '%s'\n", fill0); return 6;
    }
    std::vector<float> plane1;
    if (isSad) {
        plane1.assign((size_t)pitch * (size_t)height, 0.0f);
        if (!fill_plane(plane1, width, height, pitch, fill1, seed1)) {
            std::fprintf(stderr, "unknown --fill1 '%s'\n", fill1); return 6;
        }
    }

    // Host-side f64 reference: serial row-major accumulation over valid
    // pixels (sad: of fabsf(a-b) — exact bit-clear, then exact to double).
    const double n = (double)width * (double)height;
    double f64ref = 0.0;
    double maxabs = 0.0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t o = (size_t)y * (size_t)pitch + (size_t)x;
            double m = isSum ? (double)plane0[o]
                             : (double)std::fabs(plane0[o] - plane1[o]);
            f64ref += m;
            if (m > maxabs) maxabs = m;
        }
    }
    const double eps = 1.0 / 16777216.0; // 2^-24, exact
    const double c2bound = 4.0 * n * eps * maxabs;

    cl_uint nplat = 0;
    cl_int e = clGetPlatformIDs(0, nullptr, &nplat);
    check(e, "clGetPlatformIDs");
    std::vector<cl_platform_id> plats(nplat);
    check(clGetPlatformIDs(nplat, plats.data(), nullptr), "clGetPlatformIDs");

    cl_device_id dev = nullptr; // prefer GPU, fall back to CPU
    for (cl_platform_id p : plats) {
        cl_uint nd = 0;
        cl_device_id d = nullptr;
        e = clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, &nd);
        if (e != CL_SUCCESS || nd == 0)
            e = clGetDeviceIDs(p, CL_DEVICE_TYPE_CPU, 1, &d, &nd);
        if (e == CL_SUCCESS && nd > 0) { dev = d; break; }
    }
    if (!dev) { std::fprintf(stderr, "no OpenCL device found\n"); return 5; }
    char devname[256] = "?";
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(devname), devname, nullptr);

    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e);
    check(e, "clCreateContext");
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &e);
    check(e, "clCreateCommandQueue");

    std::string src = readFile(clPath);
    const char* cstr = src.c_str();
    size_t srclen = src.size();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &cstr, &srclen, &e);
    check(e, "clCreateProgramWithSource");
    std::string opts = std::string("-cl-std=CL") + stdver
        + " -DPX=uchar -DPX_MAX=255";
    e = clBuildProgram(prog, 1, &dev, opts.c_str(), nullptr, nullptr);
    if (e != CL_SUCCESS) {
        size_t logLen = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                              &logLen);
        std::vector<char> log(logLen + 1, 0);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logLen,
                              log.data(), nullptr);
        std::fprintf(stderr, "BUILD FAILED: %s:\n%s\n", clPath, log.data());
        return 4;
    }

    const char* kname = isSum ? "ka_sum_pixels_f32" : "ka_sad_f32";
    cl_kernel k = clCreateKernel(prog, kname, &e);
    check(e, "clCreateKernel");

    cl_mem b0 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               plane0.size() * sizeof(float), plane0.data(),
                               &e);
    check(e, "clCreateBuffer src0");
    cl_mem b1 = nullptr;
    if (isSad) {
        b1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            plane1.size() * sizeof(float), plane1.data(), &e);
        check(e, "clCreateBuffer src1");
    }
    cl_mem bsum = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float), &prefill, &e);
    check(e, "clCreateBuffer sum");

    int ai = 0;
    check(clSetKernelArg(k, ai++, sizeof(b0), &b0), "arg src0");
    if (isSad) check(clSetKernelArg(k, ai++, sizeof(b1), &b1), "arg src1");
    check(clSetKernelArg(k, ai++, sizeof(width), &width), "arg width");
    check(clSetKernelArg(k, ai++, sizeof(height), &height), "arg height");
    check(clSetKernelArg(k, ai++, sizeof(pitch), &pitch), "arg pitch");
    check(clSetKernelArg(k, ai++, sizeof(maxv), &maxv), "arg maxv");
    check(clSetKernelArg(k, ai++, sizeof(bsum), &bsum), "arg sum");

    const size_t global[2] = {
        ((size_t)width + 15) & ~(size_t)15,
        ((size_t)height + 15) & ~(size_t)15,
    };
    const size_t local[2] = {16, 16}; // handoff §4.4-1 (reqd-enforced too)

    std::printf("# avscuda_rig_host kernel=%s width=%d height=%d pitch=%d "
                "fill0=%s fill1=%s seed0=0x%08x seed1=0x%08x maxv=%d "
                "prefill=%a reps=%d\n",
                kname, width, height, pitch, fill0,
                fill1 ? fill1 : "-", seed0, seed1, maxv, (double)prefill,
                reps);
    std::printf("# device: %s\n# build: %s\n", devname, opts.c_str());
    std::printf("# f64ref=%a (%.10g) N=%.0f maxabs=%a c2bound=%.6e\n",
                f64ref, f64ref, n, maxabs, c2bound);

    for (int r = 0; r < reps; ++r) {
        // Rewrite the counter every rep (a rep otherwise accumulates onto
        // the previous run). Blocking write => ordered before the enqueue.
        check(clEnqueueWriteBuffer(q, bsum, CL_TRUE, 0, sizeof(float),
                                   &prefill, 0, nullptr, nullptr),
              "clEnqueueWriteBuffer sum");
        check(clEnqueueNDRangeKernel(q, k, 2, nullptr, global, local, 0,
                                     nullptr, nullptr),
              "clEnqueueNDRangeKernel");
        check(clFinish(q), "clFinish");
        float out = 0.0f;
        check(clEnqueueReadBuffer(q, bsum, CL_TRUE, 0, sizeof(float), &out,
                                  0, nullptr, nullptr),
              "clEnqueueReadBuffer sum");
        uint32_t bits = 0;
        std::memcpy(&bits, &out, 4);
        std::printf("run %d: bits=0x%08x val=%a (%.10g) mean_f64=%a\n",
                    r, bits, (double)out, (double)out, (double)out / n);
    }

    clReleaseMemObject(bsum);
    if (b1) clReleaseMemObject(b1);
    clReleaseMemObject(b0);
    clReleaseKernel(k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return 0;
}
