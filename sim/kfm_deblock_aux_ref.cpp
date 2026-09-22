/* CPU mirror of the graduated KDeblock-helper kernels in
 * src/opencl/kfm/kernels/kfm_deblock.cl (kf_scale_qp, kf_sharpen_coeff,
 * kf_max_h, kf_max_v, kf_max_vh).  Twins transcribed from upstream
 * KFM/Deblock.cu @68aef6e: cpu_scale_qp / cpu_sharpen_coeff / cpu_max_h /
 * cpu_max_v are exact CPU twins; kf_max_vh transcribes the device-only
 * kl_max_vh directly (box form; the golden cross-checks it via the separable
 * max_h o max_v identity).  The g_sharpen_coeff bytes are verified against
 * upstream by mechanical diff (see docs/RIG_HANDOFF_KDEBLOCK.md section 6).
 *
 * Usage: kfm_deblock_aux_ref <in> <out>
 *   All ints on one line.  Modes:
 *     S scale_qp : S width height dst_pitch src_pitch scale_type  nS
 *                  src(nS); dst = norm_qscale(src, type) mod 256.
 *                  -> output width*height
 *     C sharpen  : C width height pitch qp_pitch  nQ  qp(nQ);
 *                  q = qp>>3, dst = (q>=25) ? 255 : LUT[q].
 *                  -> output width*height
 *     H max_h    : H width height pitch radius  nP  src(nP); src is the
 *                  8px/side-padded plane (nP = pitch*(height+16), origin
 *                  at 8+8*pitch); dst = max over [x+-radius].
 *                  -> output width*height (interior)
 *     V max_v    : V ... same; dst = max over [y+-radius].
 *     B max_vh   : B ... same; dst = max over the (2R+1)^2 box.
 *     G merge    : G vis_w vis_h tmp_pitch_u4 tmp_ipitch_rows out_pitch
 *                  shift maxv  nT  tmp(nT); tmp is the padded accumulator
 *                  (pitch_ushort = 4*tmp_pitch_u4, +8 ushort / +8 row host
 *                  pre-offset applied inside); per-pixel 4-slice float32
 *                  merge with Bayer dither, fmin(maxv), truncation.
 *                  -> output vis_w*vis_h
 *     P sharpen  : P width height pitch src_pitch coeff_pitch qph  nS nC
 *                  nU  src(nS) coeff(nC) unsharp(nU); device-form 3x3
 *                  window (incl. the min(x+1,height-1) quirk), manual
 *                  float32 bilinear c, unsharp on dst pitch.
 *                  -> output width*height
 *     W show     : W width height pitch coeff_pitch qph  nC  coeff(nC);
 *                  dst = (int)bilinear(coeff,x/8,y/8).
 *                  -> output width*height
 *   P/W pin the deterministic (manual-bilinear) behaviour only; vs the
 *   CUDA texture path a device run is still required (// RIG-VERIFY kept).
 */
#include <cstdio>
#include <cstdlib>
#include <vector>

static long long g_p(const std::vector<long long>& v, size_t& i) {
    if (i >= v.size()) { std::fprintf(stderr, "short input\n"); std::exit(1); }
    return v[i++];
}

/* Upstream norm_qscale (Deblock.cu), verbatim. */
static int norm_qscale(int qscale, int type) {
    switch (type) {
    case 0: return qscale << 2;
    case 1: return qscale << 1;
    case 2: return qscale;
    case 3: return (63 - qscale + 2);
    }
    return qscale;
}

/* Upstream g_sharpen_coeff (Deblock.cu), byte-verified by diff. */
/* Upstream g_ldither (Deblock.cu), byte-verified by diff; indexed
 * [y&7][X&1][L] with X = x>>2 over ushort4 columns. */
static const int LDITHER[8][2][4] = {
  { {  0,  48,  12,  60 }, {  3,  51,  15,  63 } },
  { { 32,  16,  44,  28 }, { 35,  19,  47,  31 } },
  { {  8,  56,   4,  52 }, { 11,  59,   7,  55 } },
  { { 40,  24,  36,  20 }, { 43,  27,  39,  23 } },
  { {  2,  50,  14,  62 }, {  1,  49,  13,  61 } },
  { { 34,  18,  46,  30 }, { 33,  17,  45,  29 } },
  { { 10,  58,   6,  54 }, {  9,  57,   5,  53 } },
  { { 42,  26,  38,  22 }, { 41,  25,  37,  21 } },
};

static const int SHARPEN_COEFF[30] = {
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 10,
    50, 90, 120, 150, 160,
    170, 180, 190, 200, 210,
    220, 230, 240, 245, 250,
    255, 255, 255, 255, 255,
};

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

    if (mode == 'S') {
        long long width = g_p(v, i), height = g_p(v, i);
        long long dst_pitch = g_p(v, i), src_pitch = g_p(v, i);
        long long stype = g_p(v, i), nS = g_p(v, i);
        std::vector<long long> src((size_t)nS);
        for (long long k = 0; k < nS; k++) src[(size_t)k] = g_p(v, i);
        (void)dst_pitch; /* shape-only: output is packed row-major */
        for (long long y = 0; y < height; y++)
            for (long long x = 0; x < width; x++) {
                int s = (int)src[(size_t)(x + y * src_pitch)];
                out.push_back(norm_qscale(s, (int)stype) & 0xff);
            }
    } else if (mode == 'C') {
        long long width = g_p(v, i), height = g_p(v, i), pitch = g_p(v, i);
        long long qp_pitch = g_p(v, i), nQ = g_p(v, i);
        std::vector<long long> qp((size_t)nQ);
        for (long long k = 0; k < nQ; k++) qp[(size_t)k] = g_p(v, i);
        (void)pitch; /* shape-only: output is packed row-major */
        for (long long y = 0; y < height; y++)
            for (long long x = 0; x < width; x++) {
                int q = (int)qp[(size_t)(x + y * qp_pitch)] >> 3;
                out.push_back(q >= 25 ? 255 : SHARPEN_COEFF[q]);
            }
    } else if (mode == 'H' || mode == 'V' || mode == 'B') {
        long long width = g_p(v, i), height = g_p(v, i), pitch = g_p(v, i);
        long long radius = g_p(v, i), nP = g_p(v, i);
        std::vector<long long> src((size_t)nP);
        for (long long k = 0; k < nP; k++) src[(size_t)k] = g_p(v, i);
        long long org = 8 + 8 * pitch; /* interior origin, 8px margin */
        for (long long y = 0; y < height; y++) {
            for (long long x = 0; x < width; x++) {
                int best = 0; /* upstream uint8_t sum = 0 seed */
                if (mode == 'H') {
                    for (long long d = -radius; d <= radius; d++) {
                        int s = (int)src[(size_t)(org + (x + d) + y * pitch)];
                        if (s > best) best = s;
                    }
                } else if (mode == 'V') {
                    for (long long d = -radius; d <= radius; d++) {
                        int s = (int)src[(size_t)(org + x + (y + d) * pitch)];
                        if (s > best) best = s;
                    }
                } else {
                    for (long long j = -radius; j <= radius; j++)
                        for (long long d = -radius; d <= radius; d++) {
                            int s = (int)src[(size_t)(org + (x + d) +
                                                           (y + j) * pitch)];
                            if (s > best) best = s;
                        }
                }
                out.push_back(best);
            }
        }
    } else if (mode == 'G') {
        long long vis_w = g_p(v, i), vis_h = g_p(v, i);
        long long pitch_u4 = g_p(v, i), ipitch = g_p(v, i);
        long long out_pitch = g_p(v, i), shift = g_p(v, i);
        long long maxv_i = g_p(v, i), nT = g_p(v, i);
        std::vector<long long> tmp((size_t)nT);
        for (long long k = 0; k < nT; k++) tmp[(size_t)k] = g_p(v, i);
        (void)out_pitch; /* shape-only: output is packed row-major */
        long long pitch_us = pitch_u4 * 4;
        long long org = 8 + 8 * pitch_us; /* host pre-offset: +2 u4, +8 rows */
        float maxv = (float)maxv_i;
        float inv = 1.0f / (float)(1 << (int)shift);
        const float sixth = 1.0f / 64.0f;
        for (long long y = 0; y < vis_h; y++) {
            for (long long x = 0; x < vis_w; x++) {
                long long X = x >> 2, L = x & 3;
                int sum = 0;
                for (int k = 0; k < 4; k++) {
                    long long row = ipitch * k + y;
                    sum += (int)tmp[(size_t)(org +
                        (((X + row * pitch_u4) << 2) + L))];
                }
                int d = LDITHER[y & 7][X & 1][L];
                /* upstream op order, verbatim; -ffp-contract=off, no FMA */
                float vv = (float)sum * inv + (float)d * sixth;
                vv = vv < maxv ? vv : maxv; /* fmin (no NaN: inputs >= 0) */
                out.push_back((int)vv); /* C truncation; in [0,maxv] */
            }
        }
    } else if (mode == 'P') {
        long long width = g_p(v, i), height = g_p(v, i), pitch = g_p(v, i);
        long long src_pitch = g_p(v, i), coeff_pitch = g_p(v, i);
        long long qph = g_p(v, i);
        long long nS = g_p(v, i), nC = g_p(v, i), nU = g_p(v, i);
        std::vector<long long> src((size_t)nS), cf((size_t)nC), us((size_t)nU);
        for (long long k = 0; k < nS; k++) src[(size_t)k] = g_p(v, i);
        for (long long k = 0; k < nC; k++) cf[(size_t)k] = g_p(v, i);
        for (long long k = 0; k < nU; k++) us[(size_t)k] = g_p(v, i);
        (void)qph; /* shape-only */
        for (long long y = 0; y < height; y++) {
            for (long long x = 0; x < width; x++) {
                int s = (int)src[(size_t)(x + y * src_pitch)];
                int l = s, h = s, vv;
                long long xm1 = x - 1 >= 0 ? x - 1 : 0;
                long long xp1 = x + 1 <= height - 1 ? x + 1 : height - 1;
                long long ym1 = y - 1 >= 0 ? y - 1 : 0;
                long long yp1 = y + 1 <= height - 1 ? y + 1 : height - 1;
                vv = (int)src[(size_t)(xm1 + ym1 * src_pitch)];
                if (vv < l) l = vv; if (vv > h) h = vv;
                vv = (int)src[(size_t)(x + ym1 * src_pitch)];
                if (vv < l) l = vv; if (vv > h) h = vv;
                vv = (int)src[(size_t)(xp1 + ym1 * src_pitch)];
                if (vv < l) l = vv; if (vv > h) h = vv;
                vv = (int)src[(size_t)(xm1 + y * src_pitch)];
                if (vv < l) l = vv; if (vv > h) h = vv;
                vv = (int)src[(size_t)(xp1 + y * src_pitch)];
                if (vv < l) l = vv; if (vv > h) h = vv;
                vv = (int)src[(size_t)(xm1 + yp1 * src_pitch)];
                if (vv < l) l = vv; if (vv > h) h = vv;
                vv = (int)src[(size_t)(x + yp1 * src_pitch)];
                if (vv < l) l = vv; if (vv > h) h = vv;
                vv = (int)src[(size_t)(xp1 + yp1 * src_pitch)];
                if (vv < l) l = vv; if (vv > h) h = vv;
                /* cpu-twin-verbatim manual bilinear, then /255 (device: tex) */
                float fx = (float)x * (1.0f / 8.0f);
                float fy = (float)y * (1.0f / 8.0f);
                int ix = (int)fx, iy = (int)fy;
                float c00 = (float)cf[(size_t)(ix + iy * coeff_pitch)];
                float c01 = (float)cf[(size_t)(ix + 1 + iy * coeff_pitch)];
                float c10 = (float)cf[(size_t)(ix + (iy + 1) * coeff_pitch)];
                float c11 = (float)cf[(size_t)(ix + 1 + (iy + 1) * coeff_pitch)];
                float fracx = fx - (float)ix, fracy = fy - (float)iy;
                float b = (c00 * (1.0f - fracx) + c01 * fracx) * (1.0f - fracy)
                        + (c10 * (1.0f - fracx) + c11 * fracx) * fracy;
                float c = b * (1.0f / 255.0f);
                int u = (int)us[(size_t)(x + y * pitch)];
                float r = (float)s + (float)(s - u) * c + 0.5f;
                if (r < (float)l) r = (float)l;
                else if (r > (float)h) r = (float)h;
                out.push_back((int)r);
            }
        }
    } else if (mode == 'W') {
        long long width = g_p(v, i), height = g_p(v, i), pitch = g_p(v, i);
        long long coeff_pitch = g_p(v, i), qph = g_p(v, i), nC = g_p(v, i);
        std::vector<long long> cf((size_t)nC);
        for (long long k = 0; k < nC; k++) cf[(size_t)k] = g_p(v, i);
        (void)pitch; (void)qph; /* shape-only */
        for (long long y = 0; y < height; y++) {
            for (long long x = 0; x < width; x++) {
                float fx = (float)x * (1.0f / 8.0f);
                float fy = (float)y * (1.0f / 8.0f);
                int ix = (int)fx, iy = (int)fy;
                float c00 = (float)cf[(size_t)(ix + iy * coeff_pitch)];
                float c01 = (float)cf[(size_t)(ix + 1 + iy * coeff_pitch)];
                float c10 = (float)cf[(size_t)(ix + (iy + 1) * coeff_pitch)];
                float c11 = (float)cf[(size_t)(ix + 1 + (iy + 1) * coeff_pitch)];
                float fracx = fx - (float)ix, fracy = fy - (float)iy;
                float b = (c00 * (1.0f - fracx) + c01 * fracx) * (1.0f - fracy)
                        + (c10 * (1.0f - fracx) + c11 * fracx) * fracy;
                out.push_back((int)b);
            }
        }
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
