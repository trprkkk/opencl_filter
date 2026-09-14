/* CPU mirror of the DecombeUCF.cu reduction/accumulation kernels in
 * src/opencl/kfm/kernels/kfm_decombeucf.cl.  All twins are exact upstream
 * twins; block/group reductions run serially (the reduced ops are int add
 * and int max — associative and commutative, so every tree shape, the warp
 * shuffles, and the serial golden are value-identical).  kf_add_block_sum's
 * TH_Z thread decomposition is transcribed as a serial per-pixel loop over
 * the block (order-exact int adds); the += onto the init sums is kept.
 *
 * Usage: kfm_decombeucf_ref <in> <out>
 *   All ints on one line.  Modes:
 *     U init_uint64 : U length
 *                     -> output length zeros
 *     F field_diff  : F width height pitch nt init  nP  src(nP)
 *                     src is +-2-row padded (nP = pitch*(height+4), origin
 *                     at row 2); gated 5-tap combe sum + init.
 *                     -> output single uint64
 *     A add_block_sum: A width height pitch blocks_w blocks_h block_pitch
 *                      block_size  nP nB  s0(nP) s1(nP) initAbs(nB)
 *                      initSig(nB); block_size in {4,8,16,32}.
 *                      -> output sumAbs(nB) then sumSig(nB)
 *     X block_sum_max: X blocks_w blocks_h block_pitch initMax  nQ
 *                      abs(nQ) sig(nQ); planes are interleaved int quads
 *                      (nQ = block_pitch*blocks_h*4).
 *                      -> output single int
 *     N analyze_noise: N width height pitch i0 i1 i2 i3  nP
 *                      s0(nP) s1(nP) s2(nP)
 *                      -> output 4 uint64
 *     D analyze_diff : D width height pitch i0 i1  nP  f0(nP) f1(nP)
 *                     f0/f1 are +-2-row padded (nP = pitch*(height+4),
 *                     origin at row 2).
 *                     -> output 2 uint64
 */
#include <cstdio>
#include <cstdlib>
#include <vector>

static long long g_p(const std::vector<long long>& v, size_t& i) {
    if (i >= v.size()) { std::fprintf(stderr, "short input\n"); std::exit(1); }
    return v[i++];
}

/* Upstream KFMFilterBase.cuh CalcCombe, one lane. */
static int calc_combe_5tap(int a, int b, int c, int d, int e) {
    int t = a + c * 4 + e - (b + d) * 3;
    return t >= 0 ? t : -t;
}

int main(int argc, char** argv) {
    if (argc != 3) return 1;
    std::FILE* f = std::fopen(argv[1], "r");
    if (!f) return 1;
    char mode = 0;
    if (std::fscanf(f, " %c", &mode) != 1) return 1;
    std::vector<long long> v;
    long long t;
    while (std::fscanf(f, "%lld", &t) == 1) v.push_back(t);
    std::fclose(f);
    size_t i = 0;
    std::vector<long long> out;

    if (mode == 'U') {
        long long length = g_p(v, i);
        for (long long k = 0; k < length; k++) out.push_back(0);
    } else if (mode == 'F') {
        long long width = g_p(v, i), height = g_p(v, i), pitch = g_p(v, i);
        long long nt = g_p(v, i), init = g_p(v, i);
        long long nP = g_p(v, i);
        std::vector<long long> src((size_t)nP);
        for (long long k = 0; k < nP; k++) src[(size_t)k] = g_p(v, i);
        long long base = 2 * pitch; /* origin row 2 */
        unsigned long long total = (unsigned long long)init;
        for (long long y = 0; y < height; y++) {
            for (long long x = 0; x < width; x++) {
                int s0 = (int)src[(size_t)(base + x + (y - 2) * pitch)];
                int s1 = (int)src[(size_t)(base + x + (y - 1) * pitch)];
                int s2 = (int)src[(size_t)(base + x + (y + 0) * pitch)];
                int s3 = (int)src[(size_t)(base + x + (y + 1) * pitch)];
                int s4 = (int)src[(size_t)(base + x + (y + 2) * pitch)];
                int combe = calc_combe_5tap(s0, s1, s2, s3, s4);
                if (combe > nt) total += (unsigned long long)combe;
            }
        }
        out.push_back((long long)total);
    } else if (mode == 'A') {
        long long width = g_p(v, i), height = g_p(v, i), pitch = g_p(v, i);
        long long blocks_w = g_p(v, i), blocks_h = g_p(v, i);
        long long block_pitch = g_p(v, i), block_size = g_p(v, i);
        long long nP = g_p(v, i), nB = g_p(v, i);
        std::vector<long long> s0((size_t)nP), s1((size_t)nP);
        for (long long k = 0; k < nP; k++) s0[(size_t)k] = g_p(v, i);
        for (long long k = 0; k < nP; k++) s1[(size_t)k] = g_p(v, i);
        std::vector<long long> sumAbs((size_t)nB), sumSig((size_t)nB);
        for (long long k = 0; k < nB; k++) sumAbs[(size_t)k] = g_p(v, i);
        for (long long k = 0; k < nB; k++) sumSig[(size_t)k] = g_p(v, i);
        for (long long by = 0; by < blocks_h; by++) {
            for (long long bx = 0; bx < blocks_w; bx++) {
                long long abssum = 0, sigsum = 0;
                for (long long ty = 0; ty < block_size; ty++) {
                    long long y = by * block_size + ty;
                    for (long long tx = 0; tx < block_size; tx++) {
                        long long x = bx * block_size + tx;
                        if (x >= width || y >= height) continue;
                        long long a = s0[(size_t)(y * pitch + x)];
                        long long b = s1[(size_t)(y * pitch + x)];
                        long long d = a - b;
                        abssum += d >= 0 ? d : -d;
                        sigsum += d;
                    }
                }
                long long cell = bx + by * block_pitch;
                sumAbs[(size_t)cell] += abssum;
                sumSig[(size_t)cell] += sigsum;
            }
        }
        for (long long k = 0; k < nB; k++) out.push_back(sumAbs[(size_t)k]);
        for (long long k = 0; k < nB; k++) out.push_back(sumSig[(size_t)k]);
    } else if (mode == 'X') {
        long long blocks_w = g_p(v, i), blocks_h = g_p(v, i);
        long long block_pitch = g_p(v, i), initMax = g_p(v, i);
        long long nQ = g_p(v, i);
        std::vector<long long> ab((size_t)nQ), sg((size_t)nQ);
        for (long long k = 0; k < nQ; k++) ab[(size_t)k] = g_p(v, i);
        for (long long k = 0; k < nQ; k++) sg[(size_t)k] = g_p(v, i);
        long long best = 0; /* upstream tmpmax = 0 seed: 0-floored */
        for (long long y = 0; y < blocks_h; y++) {
            for (long long x = 0; x < blocks_w; x++) {
                long long c4 = (x + y * block_pitch) * 4;
                for (int k = 0; k < 4; k++) {
                    long long m = ab[(size_t)(c4 + k)] + sg[(size_t)(c4 + k)] * 4;
                    if (m > best) best = m;
                }
            }
        }
        if (initMax > best) best = initMax; /* atomicMax onto init */
        out.push_back(best);
    } else if (mode == 'N') {
        long long width = g_p(v, i), height = g_p(v, i), pitch = g_p(v, i);
        unsigned long long r[4];
        for (int k = 0; k < 4; k++) r[k] = (unsigned long long)g_p(v, i);
        long long nP = g_p(v, i);
        std::vector<long long> s0((size_t)nP), s1((size_t)nP), s2((size_t)nP);
        for (long long k = 0; k < nP; k++) s0[(size_t)k] = g_p(v, i);
        for (long long k = 0; k < nP; k++) s1[(size_t)k] = g_p(v, i);
        for (long long k = 0; k < nP; k++) s2[(size_t)k] = g_p(v, i);
        for (long long y = 0; y < height; y++) {
            for (long long x = 0; x < width; x++) {
                long long a = s0[(size_t)(y * pitch + x)];
                long long b = s1[(size_t)(y * pitch + x)];
                long long c = s2[(size_t)(y * pitch + x)];
                long long d0 = a - 128, d1 = b - 128, d2 = b - a, d3 = c - b;
                r[0] += (unsigned long long)(d0 >= 0 ? d0 : -d0);
                r[1] += (unsigned long long)(d1 >= 0 ? d1 : -d1);
                r[2] += (unsigned long long)(d2 >= 0 ? d2 : -d2);
                r[3] += (unsigned long long)(d3 >= 0 ? d3 : -d3);
            }
        }
        for (int k = 0; k < 4; k++) out.push_back((long long)r[k]);
    } else if (mode == 'D') {
        long long width = g_p(v, i), height = g_p(v, i), pitch = g_p(v, i);
        unsigned long long r0 = (unsigned long long)g_p(v, i);
        unsigned long long r1 = (unsigned long long)g_p(v, i);
        long long nP = g_p(v, i);
        std::vector<long long> f0((size_t)nP), f1((size_t)nP);
        for (long long k = 0; k < nP; k++) f0[(size_t)k] = g_p(v, i);
        for (long long k = 0; k < nP; k++) f1[(size_t)k] = g_p(v, i);
        long long base = 2 * pitch; /* origin row 2 */
        for (long long y = 0; y < height; y++) {
            for (long long x = 0; x < width; x++) {
                int a = (int)f0[(size_t)(base + x + (y - 2) * pitch)];
                int b = (int)f0[(size_t)(base + x + (y - 1) * pitch)];
                int c = (int)f0[(size_t)(base + x + (y + 0) * pitch)];
                int d = (int)f0[(size_t)(base + x + (y + 1) * pitch)];
                int e = (int)f0[(size_t)(base + x + (y + 2) * pitch)];
                r0 += (unsigned long long)calc_combe_5tap(a, b, c, d, e);
                int m0, m1, m2, m3, m4;
                if (y & 1) {
                    m0 = (int)f0[(size_t)(base + x + (y - 2) * pitch)];
                    m1 = (int)f1[(size_t)(base + x + (y - 1) * pitch)];
                    m2 = (int)f0[(size_t)(base + x + (y + 0) * pitch)];
                    m3 = (int)f1[(size_t)(base + x + (y + 1) * pitch)];
                    m4 = (int)f0[(size_t)(base + x + (y + 2) * pitch)];
                } else {
                    m0 = (int)f1[(size_t)(base + x + (y - 2) * pitch)];
                    m1 = (int)f0[(size_t)(base + x + (y - 1) * pitch)];
                    m2 = (int)f1[(size_t)(base + x + (y + 0) * pitch)];
                    m3 = (int)f0[(size_t)(base + x + (y + 1) * pitch)];
                    m4 = (int)f1[(size_t)(base + x + (y + 2) * pitch)];
                }
                r1 += (unsigned long long)calc_combe_5tap(m0, m1, m2, m3, m4);
            }
        }
        out.push_back((long long)r0);
        out.push_back((long long)r1);
    } else {
        std::fprintf(stderr, "bad mode %c\n", mode);
        return 1;
    }

    std::FILE* o = std::fopen(argv[2], "w");
    if (!o) return 1;
    for (size_t k = 0; k < out.size(); k++)
        std::fprintf(o, "%lld\n", out[k]);
    std::fclose(o);
    return 0;
}
