/* CPU mirrors for the KFM KDeblock QP-table / show kernels in
 * src/opencl/kfm/kernels/kfm_deblock.cl: kf_make_qp_table (cpu_make_qp_table
 * twin) and kf_deblock_show (cpu_deblock_show twin).  Mirrors are the
 * authoritative scalar transcriptions of the CUDA/CPU bodies; the Python
 * golden (python/run_kfm_deblock_qp.py) re-implements them independently.
 *
 * Usage: kfm_deblock_qp_ref <in> <out>
 *   M <make_qp_table>:  in_width in_height  has_table1 has_dc0 has_dc1
 *                       in_pitch qp_scale dc_coeff_bits
 *                       qp_shift_x qp_shift_y out_width out_height out_pitch
 *                       in_width*in_height in_table0[] in_width*in_height
 *                       nonb_table0[]  [has_table1: two more planes]
 *                       nqp qp_flags...   (we pass explicit presence flags)
 *                       [dc0 plane if has_dc0]  [dc1 plane if has_dc1]
 *   S <deblock_show>:   width height dst_pitch bw bh qp_pitch
 *                       thresh_a_bits thresh_b_bits
 *                       nqp qp_table[]  (bw*bh entries)
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
using namespace std;

static int norm_qscale(int qscale, int type) {
    switch (type) {
    case 0: return qscale << 2;
    case 1: return qscale << 1;
    case 2: return qscale;
    case 3: return (63 - qscale + 2);
    }
    return qscale;
}

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
static float qp_thresh(int qp, float ta, float tb) {
    return clampf((float)qp * ta + tb, 0.0f, (float)qp);
}

static void make_qp_table(
    int in_width, int in_height,
    const unsigned char* in_table0, const unsigned char* nonb_table0,
    bool has_table1, const unsigned char* in_table1,
    const unsigned char* nonb_table1,
    int in_pitch, int qp_scale, float dc_coeff,
    bool has_dc0, const unsigned char* dc_table0,
    bool has_dc1, const unsigned char* dc_table1, int dc_pitch,
    int qp_shift_x, int qp_shift_y, int out_width, int out_height,
    unsigned short* out_table, int out_pitch) {
    for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            int qp;
            if (in_table0) {
                int qp_x = std::min(x >> qp_shift_x, in_width - 1);
                int qp_y = std::min(y >> qp_shift_y, in_height - 1);
                int in_qp = (int)in_table0[qp_x + qp_y * in_pitch];
                int nonb_qp = (int)nonb_table0[qp_x + qp_y * in_pitch];
                int dc = has_dc0 ? (int)dc_table0[qp_x + qp_y * dc_pitch] : 255;
                if (has_table1) {
                    in_qp = std::max(in_qp, (int)in_table1[qp_x + qp_y * in_pitch]);
                    nonb_qp = std::max(nonb_qp, (int)nonb_table1[qp_x + qp_y * in_pitch]);
                    dc = std::max(dc, has_dc1 ? (int)dc_table1[qp_x + qp_y * dc_pitch] : 255);
                }
                int b = norm_qscale(in_qp, qp_scale);
                int nonb = norm_qscale(nonb_qp, qp_scale);
                float b_ratio = std::min(1.0f, (float)dc * dc_coeff);
                qp = std::max(1, (int)(b * b_ratio + nonb * (1.0f - b_ratio) + 0.5f));
            } else {
                qp = qp_scale;
            }
            out_table[x + y * out_pitch] = (unsigned short)qp;
        }
    }
}

static void deblock_show(int width, int height, int dst_pitch,
    const unsigned char* dst_in /*unused; plane init handled by caller*/,
    unsigned char* dst, int bw, int bh,
    const unsigned short* qp_table, int qp_pitch,
    float thresh_a, float thresh_b) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int bx = (x + 4) >> 3;
            int by = (y + 4) >> 3;
            unsigned short qp = qp_table[bx + by * qp_pitch];
            int enabled = (qp_thresh((int)qp, thresh_a, thresh_b) >= (int)(qp >> 1));
            dst[x + y * dst_pitch] = enabled ? 230 : 16;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: kfm_deblock_qp_ref <in> <out>\n"); return 2; }
    FILE* f = fopen(argv[1], "r");
    std::vector<int> t; int x;
    while (f && fscanf(f, "%d", &x) == 1) t.push_back(x);
    if (f) fclose(f);
    size_t p = 0;
    auto rd = [&]() { return t[p++]; };
    int mode = rd();
    if (mode == 'M') {
        int in_width = rd(), in_height = rd();
        int has_table1 = rd(), has_dc0 = rd(), has_dc1 = rd();
        int in_pitch = rd(), qp_scale = rd();
        unsigned dc_bits = rd(); float dc_coeff; memcpy(&dc_coeff, &dc_bits, 4);
        int qsx = rd(), qsy = rd(), ow = rd(), oh = rd(), op = rd();
        int nin = in_width * in_height;
        std::vector<unsigned char> in0(nin), nb0(nin);
        for (int i = 0; i < nin; ++i) in0[i] = rd();
        for (int i = 0; i < nin; ++i) nb0[i] = rd();
        std::vector<unsigned char> in1, nb1;
        if (has_table1) { in1.resize(nin); nb1.resize(nin);
            for (int i = 0; i < nin; ++i) in1[i] = rd();
            for (int i = 0; i < nin; ++i) nb1[i] = rd(); }
        std::vector<unsigned char> dc0, dc1;
        if (has_dc0) { dc0.resize(nin); for (int i = 0; i < nin; ++i) dc0[i] = rd(); }
        if (has_dc1) { dc1.resize(nin); for (int i = 0; i < nin; ++i) dc1[i] = rd(); }
        std::vector<unsigned short> out((size_t)op * oh, 0);
        make_qp_table(in_width, in_height, in0.data(), nb0.data(),
            has_table1, has_table1 ? in1.data() : nullptr,
            has_table1 ? nb1.data() : nullptr,
            in_pitch, qp_scale, dc_coeff,
            has_dc0, has_dc0 ? dc0.data() : nullptr,
            has_dc1, has_dc1 ? dc1.data() : nullptr, in_pitch,
            qsx, qsy, ow, oh, out.data(), op);
        FILE* o = fopen(argv[2], "w");
        for (unsigned v : out) fprintf(o, "%d\n", v);
        fclose(o);
    } else if (mode == 'S') {
        int width = rd(), height = rd(), dst_pitch = rd();
        int bw = rd(), bh = rd(), qp_pitch = rd();
        unsigned ta_b = rd(), tb_b = rd();
        float ta; memcpy(&ta, &ta_b, 4); float tb; memcpy(&tb, &tb_b, 4);
        int nqp = rd();
        std::vector<unsigned short> qp(nqp);
        for (int i = 0; i < nqp; ++i) qp[i] = rd();
        std::vector<unsigned char> dst((size_t)dst_pitch * height, 0);
        deblock_show(width, height, dst_pitch, nullptr, dst.data(),
            bw, bh, qp.data(), qp_pitch, ta, tb);
        FILE* o = fopen(argv[2], "w");
        for (unsigned char v : dst) fprintf(o, "%d\n", v);
        fclose(o);
    } else { fprintf(stderr, "bad mode\n"); return 2; }
    return 0;
}
