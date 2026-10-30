// kfm_analyze_avs.cpp — AviSynth+ host plugin for the OpenCL port of KFM's
// CombingAnalyze.cu + DecombeUCF.cu reduction/analysis filters (upstream:
// rigaya/AviSynthCUDAFilters KFM, MIT).
//
// The SEVENTH real AviSynth+ host plugin of the port (after KEdgeLevelOCL,
// KTemporalNROCL, KDebandOCL, KTemporalDiffOCL+KMergeStaticOCL, KNoiseClipOCL
// and KDeblockOCL): it loads the ALG-VERIFIED kernels of
//   src/opencl/kfm/kernels/kfm_combinganalyze.cl  (15/15 kernels),
//   src/opencl/kfm/kernels/kfm_decombeucf.cl      (8/8 kernels)
// (both embedded into the DLL at build time) through the small host runtime in
// src/host/ocl_host.* and replays the upstream filters' host assembly.
//
// REGISTERED FILTERS (exact upstream signatures/defaults; the upstream name
// registered as an alias for each):
//
//   KFMSuperOCL / KFMSuper               "cc"
//   KCleanSuperOCL / KCleanSuper         "c[thY]i[thC]i"
//   KPreCycleAnalyzeOCL / KPreCycleAnalyze
//                             "c[threshMY]i[threshSY]i[threshMC]i[threshSC]i"
//   KPreCycleAnalyzeShowOCL / KPreCycleAnalyzeShow   "cc"
//   KContainsCombeOCL / KContainsCombe   "c"
//   KCombeMaskOCL / KCombeMask           "cc"
//   KRemoveCombeOCL / KRemoveCombe       "cc[thY]f[thC]f"
//   KCFieldDiffOCL / KCFieldDiff         "c[nt]f[chroma]b"      (conditional)
//   KCFrameDiffDupOCL / KCFrameDiffDup   "c[chroma]b[blksize]i" (conditional)
//   KAnalyzeNoiseOCL / KAnalyzeNoise     "cc[s4uper]c"
//
// LAUNCH SEQUENCES per GetFrame (upstream verbatim; "dev" = OpenCL launch):
//
//   KFMSuper (upstream KFMSuper::GetFrameT + AnalyzeFrame): parity =
//   child->GetParity(n); f0/f1 = Frame(padclip->GetFrame(n / n+1), VPAD)
//   (the +4-row ROI of a +-4-row mirror-padded pad clip); then per plane
//   (Y, U, V) ONE dev kf_super_analyze over the (nBlkX-1, nBlkY-1) block grid
//   (4x4-stride cells, the 8-tap calc_combe/calc_diff sums, clamp(sum>>shift)
//   into the flag pairs at (bx+1, 2*(by+1)+{0,1}); shift = bits-8+4).  The
//   uchar2 flag frame is byte-packed in AviSynth (a w/2-wide 8-bit plane
//   holding (flag_x, flag_y) pairs): the host stages two uchar planes and
//   interleaves them on download (DEVIATION 1).  Create then chains
//   env->Invoke("SeparateFields") and env->Invoke("Align") like upstream
//   (Align is skipped, with a printed note, when the core does not provide
//   it — DEVIATION 10).
//
//   KCleanSuper (upstream KCleanSuper::GetFrame/CleanSuper): prev =
//   child->GetFrame(n-1), cur = child->GetFrame(n); ONE dev kf_clean_super
//   per plane (Y with thY, U and V with thC) over the uchar2 element grid:
//   dst.y = cur.y; dst.x = (prev.y <= th && cur.y <= th) ? 0 : cur.x.
//   (Upstream fuses U+V over blockIdx.z with dstV as the z=1 buffer; the
//   values are plane-independent, so the host launches U and V separately —
//   exactly upstream's own cpu twin structure.)  The pair-packed planes are
//   deinterleaved before upload and interleaved on download (DEVIATION 1).
//
//   KPreCycleAnalyze (upstream KPreCycleAnalyze::GetFrame): parity =
//   child->GetParity(0); combe0/1 = child->GetFrame(2n / 2n+1); ONE dev
//   kf_init_fmcount (<<<1,2>>>) zeroing the two FMCount records; ONE dev
//   kf_count_cmflags over the Y pair planes and ONE dev
//   kf_count_cmflags_2planes over the U+V pairs — both over the upstream
//   interior (combe pointers +pitch+1) expressed as a global-work offset of
//   (1,1) with the full-plane guard (DEVIATION 2), and both with the
//   REQUIRED 32x16 work-group shape (dispatch2DLocalOffset; rig §9.4).
//   Thresholds prmY = (threshMY, threshSY, threshSY*3), prmC = (threshMC,
//   threshSC, threshSC*3).  The 24-byte result (FMCount[2] = {move, shima,
//   lshima} x 2, slot i ^ !parity) lands in the BGR32 6x1 output frame.
//
//   KPreCycleAnalyzeShow (upstream: pure host): reads the fmclip frame's
//   FMCount[2] and DrawTexts "Pre: [0](m,s,l) [1](m,s,l)" at (0,0) with the
//   bit-exact 10x20 font port (kfm_textout_font.h, mechanically extracted
//   from TextOut.cu; same twin as the KDeblock show path).
//
//   KContainsCombe (upstream KContainsCombe::GetFrame/ContainsDurtyBlock):
//   scans the WHOLE flag frame (upstream loops bx<flag.GetWidth<uint8_t>(),
//   by<flag.GetHeight(); the WrapSwitchFragFrame crop belongs to KCombeMask,
//   NOT here); ONE dev kf_init_contains_durty_block (<<<1,1>>>) + ONE dev
//   kf_contains_durty_block over the full plane (plain (0,0) launch over a
//   tight staging — no crop); the resulting int (0/1 OR-reduction) lands in
//   the BGR32 4x1 output frame.
//
//   KCombeMask (upstream KCombeMask::GetFrame/MakeMask/BilinearImage): flag
//   ROI = WrapSwitchFragFrame(4, 2); the 1px padv/padh mirror ring around the
//   ROI and the separable bilinear upscale (Y at DC_BLOCK_SIZE 8x8, U/V at
//   scaleUV = widthUV/fwidth x heightUV/fheight, asserted 4 or 8; the
//   upstream (SCALE,SHIFT) table entries (8,3) and (4,2)) run as the exact
//   upstream CPU twin (cpu_bilinear_h/v over the padv/padh ring) in the host
//   staging (DEVIATION 3 — the KDeblock precedent: the device path offsets
//   the h-pass source one row up and the v-pass source one row down and its
//   x0=(x-half)>>shift reaches the ring's negative column, an interior-origin
//   pointer argument a whole-buffer cl_mem cannot express on this driver,
//   RIG_HANDOFF §9.4).
//
//   KRemoveCombe (upstream KRemoveCombe::GetFrameT/RemoveCombe): src =
//   Frame(padclip->GetFrame(n), VPAD) (the pad clip supplies the +-4-row
//   halo), combe = superclip->GetFrame(n) (.x lane of the pair-packed super
//   frame); per plane the combe-gated vertical binomial merge
//   (a + 2b + c + 2) >> 2 over the vi.height-8 interior.  DEVIATION 4: the
//   op runs as the exact CPU twin (upstream's own cpu_remove_combe2,
//   verbatim transcription) in the host staging — kf_remove_combe2's
//   per-pixel write is indexed by the OUTPUT row y while its halo reads
//   (src y-1..y+1) AND its combe row index (y>>2) are both tied to that
//   same y, so no launch offset can rebase a whole-buffer argument without
//   corrupting the combe row mapping (the interior-origin pointer argument
//   this needs is exactly what sub-buffers cannot express on this driver,
//   §9.4).  The kernel remains ported and ALG-VERIFIED; upstream's own CPU
//   path runs the same twin.
//
//   KCFieldDiff (upstream KFieldDiff::CFunc/ConditionalFieldDiff/
//   CalcFieldDiff): conditional function — reads
//   env->GetVar("current_frame") (throws "This filter can only be used
//   within ConditionalFilter!" when absent), builds the filter with
//   (nt=3, chroma=true) and returns the double.  Per frame: stage the child
//   frame with the mirror-pad rows the kernel taps (+-2; bit-identical to
//   upstream's VPAD=4 CopyFrameAndPad for the taps the kernel reads —
//   DEVIATION 5); ONE dev kf_init_uint64, then per plane (Y; U, V when
//   chroma) ONE dev kf_calculate_field_diff (gated CalcCombe 5-tap sum,
//   REQUIRED 32x16 work-groups) accumulating into ONE ulong sum; the
//   kernel's (y-2)..(y+2) taps are kept verbatim by launching through a
//   global work offset of +2 rows over the padded staging (a pure reducer,
//   so the row attribution is irrelevant — DEVIATION 5); read back,
//   raw /= 6, result = (double)(raw >> (bits-8)).
//
//   KCFrameDiffDup (upstream KFrameDiffDup::CFunc/ConditionalFrameDiff/
//   CalcFrameDiff): conditional function (defaults chroma=true,
//   blocksize=32); src0/src1 = child frames clamp(n-1 / n, 0, num-1); ONE
//   dev kf_init_block_sum, per plane (Y; U, V when chroma) ONE dev
//   kf_add_block_sum (per-block-cell sumAbs/sumSig accumulation into the
//   SHARED work arrays, block_size runtime arg), then ONE dev
//   kf_block_sum_max over the int4-quad view (blocks_w4 = blocks_w>>2,
//   REQUIRED 32x16 work-groups, atomic max); read maxSum and return
//   maxSum / (2.0 * (235 << (bits-8)) * blocksize^2) * 100.0.  blocks_w/h =
//   nblocks(dim, blocksize); block_pitch aligned to max(4, th_uv_z) exactly
//   like upstream's ctor.
//
//   KAnalyzeNoise (upstream KAnalyzeNoise::GetFrameT; 8-bit only): noise0/1/2
//   = noiseclip frames 2n / 2n+1 / 2n+2; f0p/f1p = superclip frames n/n+1
//   (VPAD ROI) or child frames staged with the mirror-pad rows (DEVIATION 5);
//   ONE dev kf_init_uint64 over the result accumulators; per plane (Y, U, V)
//   ONE dev kf_analyze_noise (4-way |.-128| / |delta| sums) and ONE dev
//   kf_analyze_diff (CalcCombe sum0 + the TFF field-mixed weave sum1,
//   offset-launched over the padded staging), both with REQUIRED 32x16
//   work-groups; the NoiseResult[2] (Y record at [0], UV record at [1]) is
//   written into the BGR32 16x2 output frame, whose vi is tagged with the
//   upstream UCFNoiseMeta (sample_type magic + meta pointer in
//   num_audio_samples) for the downstream KDecombUCF* consumers (NOT ported
//   — see GAPS).
//
// STATE MODEL (stated like the KMergeStaticOCL/KDeblockOCL precedents): every
// filter here holds NO persistent state between GetFrame calls; all per-frame
// data arrives from the child/auxiliary clips (the frame indices each filter
// reads are pure functions of (n, fixed config)) and every device buffer is
// per-call scratch, overwritten (or zero-filled where the kernels do not
// write every element) on every GetFrame.  KCFieldDiff / KCFrameDiffDup
// construct their state on every evaluation exactly like upstream's CFunc.
// The filters are stateless.
//
// UPSTREAM KERNEL INVENTORY (audited for this plugin): the nine device-launching
// implemented classes launch exactly kf_super_analyze, kf_clean_super,
// kf_init_fmcount, kf_count_cmflags, kf_count_cmflags_2planes,
// kf_init_contains_durty_block, kf_contains_durty_block, kf_init_uint64,
// kf_calculate_field_diff, kf_init_block_sum, kf_add_block_sum,
// kf_block_sum_max, kf_analyze_noise, kf_analyze_diff — every one of which
// has an ALG-VERIFIED OpenCL twin (kfm_combinganalyze.cl / kfm_decombeucf.cl);
// KCombeMask and KRemoveCombe run their upstream CPU twins in host staging
// (DEVIATIONS 3/4).  The remaining CombingAnalyze/DecombeUCF kernels
// (kf_copy_first, kf_combe_to_flag, kf_sum_box3x3, kf_binary_flag,
// kf_temporal_soften, kf_remove_combe2, kf_bilinear_h/v) belong to KSwitchFlag /
// KFMSuperShow / KRemoveCombe's CPU path — see GAPS at the file bottom.
//
// Documented deviations from the upstream host code (none change any pixel or
// counter for the golden-pinned parameter space):
//   1. The uchar2 "super" frames (pair-packed 8-bit planes) are staged as
//      two separate uchar .x/.y planes and interleaved on download.  The
//      ported kernels address .x/.y as separate element buffers (upstream
//      addresses one uchar2 array); pure layout bookkeeping, bit-identical
//      bytes.
//   2. Interior-origin pointer arguments (KPreCycleAnalyze's combe0Y +=
//      pitch+1, KCFieldDiff/KAnalyzeNoise/KFMSuper's pad-plane taps) are
//      expressed as global-work OFFSETS over the full-plane buffers, with the
//      guard argument widened by the offset so the same absolute index set is
//      processed (count kernels: offset (1,1) + guard (We,He) admits exactly
//      the upstream ROI cells 1..We-1; field_diff/analyze_diff: offset (0,2) +
//      guard (W,H+2)).  Sub-buffers cannot express interior origins on this
//      driver (RIG_HANDOFF §9.4); the reducer/writer index math is otherwise
//      untouched.
//   3. KCombeMask's 1px padv/padh mirror ring around the flag ROI AND its
//      separable bilinear upscale run as the exact upstream CPU twin
//      (cpu_bilinear_h / cpu_bilinear_v over the padv/padh ring) in the host
//      staging — no device launch.  The device path offsets the h-pass source
//      one row up (flagp - fpitch) and the v-pass source one row down
//      (dsttmp + dpitch), and its x0 = (x-half)>>shift reaches the ring's
//      negative column: an interior-origin pointer argument a whole-buffer
//      cl_mem cannot express on this driver (same limitation as 2), and the
//      v-pass y0 mapping goes negative for the first `half` output rows.  The
//      upscaled planes are small, so the CPU twin is the faithful choice.
//   4. KRemoveCombe runs the exact upstream CPU twin cpu_remove_combe2 in
//      the host staging (see the launch-sequence paragraph; no device
//      launch).  Upstream itself ships this twin as its CPU path.
//   5. The DecombeUCF padded planes are staged with the mirror-pad rows the
//      kernels tap (+-2) instead of upstream's full VPAD=4 ring — the
//      kernels only ever read (y-2)..(y+2), and the mirror rule (row -1 =
//      row 0, row -2 = row 1) is identical for any pad depth >= 2, so the
//      staged planes are bit-identical where it matters.  The kernel's
//      (y-2)..(y+2) tap text is kept verbatim by launching through a global
//      work offset of +2 rows over the padded staging (pure reducers: the
//      row attribution is irrelevant to the accumulated scalar).
//   6. KAnalyzeNoise's result accumulators are four separate device buffers
//      (Y-noise, UV-noise, Y-diff, UV-diff) instead of one NoiseResult[2]
//      array, and KPreCycleAnalyze's FMCount[2] is three separate 2-slot
//      field arrays — the kernels index their accumulators from the base
//      pointer, which a whole-buffer cl_mem argument cannot offset; the
//      assembled output bytes are identical.  The KAnalyzeNoise output
//      frame's 32 trailing bytes (upstream: uninitialized) are zero-filled.
//   7. Upstream's GetDeviceTypes CUDA device-match checks are CUDA-only and
//      not reproduced (all clips go through this plugin's one queue).
//   8. KCFrameDiffDup: upstream's blocksize switch leaves f_idx
//      uninitialized for blocksize 4 (its table has a 4th entry but the
//      switch does not map it); the port maps 4 -> the table's 4th entry
//      deterministically.  blocksize outside {4,8,16,32} throws.
//   9. KPreCycleAnalyzeShow draws on a fresh copy of the source frame
//      (upstream MakeWritable's the source frame in place — same pixels,
//      the KDeblock show precedent), and is 8-bit-only (upstream
//      instantiates DrawText<uint8_t> only).
//  10. KFMSuper::Create invokes the core Align filter like upstream, but
//      only when the host provides one (FunctionExists); otherwise the
//      SeparateFields result is returned unchanged (AviSynth+ 3.7.3 ships
//      no Align filter).
//
// GAPS (upstream registrations NOT implemented by this DLL, with reasons):
//   - KSwitchFlag("c[thY]f[thC]f"): its host pipeline is not faithfully
//     implementable.  combe_to_flag's launch grid (fwidth-1, fheight-1)
//     reads combe rows up to 2*(fheight-2)+1, while the combe staging has
//     combvi.height = childH rows and the ctor's own flag dims are
//     vi.height = nblocks(combvi.height, 2) + 4 (combvi.width /= 2 but NOT
//     combvi.height).  So the declared flag height is twice the row count a
//     2x2 row-downsample of the combe plane can fill: the kernel's vertical
//     read walks ~2x past the combe plane's last row (an upstream
//     host-side dims bug — out-of-bounds reads with undefined content) and
//     the flag's lower half is never written.  The undefined content cannot
//     be reproduced; the flag rows beyond the first half are garbage or
//     zero depending on allocation.  (kf_copy_first / kf_temporal_soften /
//     kf_max / kf_max_extend_blocks_h/v / kf_sum_box3x3 / kf_binary_flag /
//     kf_combe_to_flag kernels themselves are all ported and verified;
//     KSwitchFlag is the only consumer of kf_copy_first / kf_combe_to_flag /
//     kf_sum_box3x3 / kf_binary_flag / kf_temporal_soften.)
//   - KFMSuperShow("c[threshMY]i[...]i"): debug-only visualiser
//     (ExtendBlocks<uchar4> + a CPU colour loop); its only device work
//     (kf_max_extend_blocks_h/v) adds no kernel coverage beyond the KFM
//     chain, and its ExtendBlocks composition is shared with the skipped
//     KSwitchFlag.  Listed, not implemented.
//   - KTelecine("cc[show]b") / KTelecineSuper("cc"): require an fmclip
//     producing the KFMResult records of the KFMKernel.cu analysis family
//     (no ported kernels; no kfm_kernel.cl exists in this repo), plus the
//     CycleAnalyzeInfo magic-key VideoInfo plumbing.
//   - KDecombUCFParam / KDecombUCF / KDecombUCF24 / KDecombUCF60Flag /
//     KDecombUCF60 (DecombeUCF.cu): pure host decision machinery over
//     NoiseResult frames + magic-key VideoInfo + frame-props plumbing — no
//     device kernels at all (KDecombUCF24 additionally needs the unported
//     KFMKernel fmclip); zero kernel coverage, not ported.
//   - KPatchCombe("ccccc") / KFMSwitch(...) (KFMKernel.cu): their host
//     assembly consumes the KFMResult `fmclip` records produced by the
//     KFMKernel.cu analysis family (KCCombingAnalyze/KFMCycleAnalyze/
//     KFMDumpFM), which is not ported (no kfm_kernel.cl exists in this repo);
//     KPatchCombe additionally needs the 24/30/60-fps src/flag/mask/ucf clip
//     matrix.  The blender kernel they drive (kf_merge_block) is ported in
//     kfm_filterbase.cl but has no consumer here.
//   - KFMPad("c") (KFMKernel.cu): its device kernel (kf_copy_pad / the pad
//     pair in kfm_filterbase.cl) is ported and ALG-VERIFIED, but KFMPad is a
//     plain whole-frame mirror-pad filter with no analysis role; deferred
//     (trivially host-staged if needed later), not needed by this family.
//   - KFMDecimate("c[filepath]s") / AssumeDevice("ci") (KFMKernel.cu): host
//     bookkeeping (frame decimation by frame-props, device tagging); no
//     device kernels; not ported.
//   - KAnalyzeStatic("c[thcombe]f[thdiff]f[pad]c", MergeStatic.cu): kernels
//     100% ported, but five of its seven stages are halo-reading writers or
//     internally-clamped chain stages whose shifted-buffer formulation
//     (global-work offsets with per-plane row shifts that differ by
//     subsampling, border pre-fills for the internally-clamped
//     kf_extend_coef2, and a re-upload before kf_apply_uvcoefs_420 for the
//     444 family) is deferred to a later slice; documented in the port
//     notes.
//
// GRunT: no KFM class implemented here requires GRunT.  The GRunT pairings
// in the upstream workflows feed ConditionalFilter/ScriptFunction around the
// KCFieldDiff/KCFrameDiffDup conditional functions and the KDecombUCF*
// consumers — the conditional functions themselves are plain env-variable
// readers and ARE implemented; the KDecombUCF* consumers are not (GAPS).
// No GRunT dependency.
//
// The plugin is a plain DLL exporting AvisynthPluginInit3 (no import lib);
// it validates autoload + registration against the installed AviSynth+ 3.7.3
// in test_kfm_analyze.cpp.
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "avisynth.h"
#include "../ocl_host.h"
#include "kfm_combinganalyze_cl.h"  // GENERATED from src/opencl/kfm/kernels/kfm_combinganalyze.cl
#include "kfm_decombeucf_cl.h"      // GENERATED from src/opencl/kfm/kernels/kfm_decombeucf.cl
#include "kfm_textout_font.h"

namespace {

using oclhost::Buffer;
using oclhost::Context;
using oclhost::Program;

// upstream KFM.h enums (verbatim)
enum {
    VPAD = 4,
    COMBE_FLAG_PAD_W = 4,
    COMBE_FLAG_PAD_H = 2,
};

struct FMCount {
    int move, shima, lshima;
};

int nblocks(int a, int b) { return (a + b - 1) / b; }

// upstream KFMFilterBase.cu scaleParam (verbatim)
int scaleParam(float thresh, int pixelBits) {
    return (int)(thresh * (1 << (pixelBits - 8)) + 0.5f);
}

// upstream Get8BitType (KFMFilterBase.cu; the planar YUV family this port
// supports — the non-planar fallback of upstream is unreachable here because
// the ctors reject non-planar YUV first)
int get8BitType(const VideoInfo& vi) {
    if (vi.Is420()) return VideoInfo::CS_YV12;
    if (vi.Is422()) return VideoInfo::CS_YV16;
    return VideoInfo::CS_YV24;
}

// One plain plane: a w*h device image, pitch == width (Plane discipline).
struct Plane {
    Buffer buf;
    int w = 0, h = 0, px = 1;
    size_t bytes() const { return (size_t)w * h * px; }
    void alloc(Context& ctx, int w_, int h_, int px_) {
        w = w_; h = h_; px = px_;
        buf = Buffer(ctx, bytes(), /*zero=*/false);
    }
    void fillZero(Context& ctx) {
        buf.fill32(0);
        size_t tail = bytes() & 3;
        if (tail) {
            static const uint8_t z[3] = { 0, 0, 0 };
            buf.write(z, bytes() - tail, tail);
        }
        ctx.finish();
    }
    cl_mem mem() const { return buf.mem(); }
};

Context& sharedContext() {
    static Context ctx;  // thread-safe magic static; one device/queue per process
    return ctx;
}

std::mutex& queueMutex() {
    static std::mutex mtx;  // one shared in-order queue for the whole DLL
    return mtx;
}

void copyFromFrame(const uint8_t* srcp, int srcPitch, int rowsize, int height, uint8_t* dst,
                   int dstPitch) {
    for (int y = 0; y < height; ++y)
        memcpy(dst + (size_t)y * dstPitch, srcp + (size_t)y * srcPitch, rowsize);
}

// stage the visible rows of one plane into a staging vector (pitch = width)
template <typename T>
void stagePlane(std::vector<T>& dst, const T* srcp, int srcPitch, int w, int h) {
    dst.resize((size_t)w * h);
    for (int y = 0; y < h; ++y)
        memcpy(dst.data() + (size_t)y * w, srcp + (size_t)y * srcPitch, (size_t)w * sizeof(T));
}

template <typename T>
void downloadPlane(PVideoFrame& f, int plane, const std::vector<T>& src, int w, int h) {
    int pitch = f->GetPitch(plane) / (int)sizeof(T);
    T* dstp = (T*)f->GetWritePtr(plane);
    for (int y = 0; y < h; ++y)
        memcpy(dstp + (size_t)y * pitch, src.data() + (size_t)y * w, (size_t)w * sizeof(T));
}

// deinterleave the uchar2 pair planes: .x = even bytes, .y = odd bytes
// (element-pitch views of the pair-packed plane; DEVIATION 1)
void deinterleave(std::vector<uint8_t>& x, std::vector<uint8_t>& y,
                  const uint8_t* srcp, int srcPitchBytes, int elems, int rows) {
    x.resize((size_t)elems * rows);
    y.resize((size_t)elems * rows);
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row = srcp + (size_t)r * srcPitchBytes;
        for (int e = 0; e < elems; ++e) {
            x[(size_t)r * elems + e] = row[2 * e];
            y[(size_t)r * elems + e] = row[2 * e + 1];
        }
    }
}

void interleave(std::vector<uint8_t>& dst, const std::vector<uint8_t>& x,
                const std::vector<uint8_t>& y, int elems, int rows) {
    dst.resize((size_t)elems * 2 * rows);
    for (int r = 0; r < rows; ++r)
        for (int e = 0; e < elems; ++e) {
            dst[(size_t)r * elems * 2 + 2 * e] = x[(size_t)r * elems + e];
            dst[(size_t)r * elems * 2 + 2 * e + 1] = y[(size_t)r * elems + e];
        }
}

// mirror index (kf_padv/kf_padh / cpu twins; same expression as KDeblock's)
int mirrorIdx(int v, int n) {
    if (v < 0) return -v - 1;
    if (v >= n) return 2 * n - v - 1;
    return v;
}

// host-staged mirror pad with independent horizontal/vertical ring depths
// (bit-identical to the cpu_padv/cpu_padh composition: dst[-y-1] = dst[y])
template <typename T>
void stageMirrorPad2(std::vector<T>& dst, const T* srcp, int srcPitch, int w, int h,
                     int padv, int padh) {
    const int pw = w + 2 * padh, ph = h + 2 * padv;
    dst.resize((size_t)pw * ph);
    for (int y = 0; y < ph; ++y) {
        int sy = mirrorIdx(y - padv, h);
        for (int x = 0; x < pw; ++x) {
            int sx = mirrorIdx(x - padh, w);
            dst[(size_t)y * pw + x] = srcp[(size_t)sy * srcPitch + sx];
        }
    }
}

// vertical-only mirror pad of exactly `pad` rows (the DecombeUCF ring;
// bit-identical to the VPAD=4 ring for the +-2 taps the kernels read)
template <typename T>
void stageMirrorPadV(std::vector<T>& dst, const T* srcp, int srcPitch, int w, int h, int pad) {
    const int ph = h + 2 * pad;
    dst.resize((size_t)w * ph);
    for (int y = 0; y < ph; ++y) {
        int sy = mirrorIdx(y - pad, h);
        memcpy(dst.data() + (size_t)y * w, srcp + (size_t)sy * srcPitch, (size_t)w * sizeof(T));
    }
}

// ---------------------------------------------------------------------------
// cpu_draw_text / DrawText port (upstream TextOut.cu, bit-exact — the same
// twin the KDeblock plugin verified): glyph cell 10x20 from kfmtext::kFont;
// luma set 235<<shift, chroma set 128<<shift; unset pixels dimmed.
// ---------------------------------------------------------------------------
template <typename T>
void drawTextPlane(T* dst, int width, int height, int pitch,
                   const char* charmap, int mapw, int maph, int mappitch,
                   int logx, int logy, int pixelshift, bool isUV) {
    for (int by = 0; by < maph; ++by) {
        for (int bx = 0; bx < mapw; ++bx) {
            for (int ty = 0; ty < 20; ty += (1 << logy)) {
                for (int tx = 0; tx < 10; tx += (1 << logx)) {
                    int x = (tx + bx * 10) >> logx;
                    int y = (ty + by * 20) >> logy;
                    if (x < width && y < height) {
                        int c = charmap[bx + by * mappitch] - ' ';
                        if (c >= 0 && c <= 'z' - ' ') {
                            if (kfmtext::kFont[c][ty] & (1 << (15 - tx))) {
                                dst[(size_t)x + (size_t)y * pitch] =
                                    (T)(isUV ? (128 << pixelshift) : (235 << pixelshift));
                            } else {
                                dst[(size_t)x + (size_t)y * pitch] =
                                    (T)(((int)dst[(size_t)x + (size_t)y * pitch] +
                                         (isUV ? (128 << pixelshift) : 0)) >> 1);
                            }
                        }
                    }
                }
            }
        }
    }
}

// upstream DrawText (single-line form at (0,0))
template <typename T>
void drawTextMessage(PVideoFrame& f, int bits, const std::string& msg) {
    int maxlen = (int)msg.size();
    if (maxlen <= 0) return;
    std::vector<char> charmap((size_t)maxlen, 0);
    memcpy(charmap.data(), msg.data(), msg.size());

    T* dstY = (T*)f->GetWritePtr(PLANAR_Y);
    T* dstU = (T*)f->GetWritePtr(PLANAR_U);
    T* dstV = (T*)f->GetWritePtr(PLANAR_V);
    int width = f->GetRowSize(PLANAR_Y) / (int)sizeof(T);
    if (width > maxlen * 10) width = maxlen * 10;
    int height = f->GetHeight(PLANAR_Y);
    if (height > 20) height = 20;
    int logx = (f->GetRowSize(PLANAR_U) / (int)sizeof(T) < width) ? 1 : 0;
    int logy = (f->GetHeight(PLANAR_U) < height) ? 1 : 0;
    int shift = bits - 8;
    drawTextPlane(dstY, width, height, f->GetPitch(PLANAR_Y) / (int)sizeof(T),
                  charmap.data(), maxlen, 1, maxlen, 0, 0, shift, false);
    drawTextPlane(dstU, width >> logx, height >> logy, f->GetPitch(PLANAR_U) / (int)sizeof(T),
                  charmap.data(), maxlen, 1, maxlen, logx, logy, shift, true);
    drawTextPlane(dstV, width >> logx, height >> logy, f->GetPitch(PLANAR_U) / (int)sizeof(T),
                  charmap.data(), maxlen, 1, maxlen, logx, logy, shift, true);
}

// ===========================================================================
// CombingAnalyze.cu family
// ===========================================================================

// ---------------------------------------------------------------------------
// KFMSuperOCL — upstream KFMSuper (AnalyzeFrame per plane).
// ---------------------------------------------------------------------------
class KFMSuperOCL : public GenericVideoFilter {
public:
    PClip padclip;
    VideoInfo srcvi;
    int logUVx, logUVy;
    int px;  // source ComponentSize

    KFMSuperOCL(PClip clip, PClip pad, IScriptEnvironment* env)
        : GenericVideoFilter(clip), padclip(pad) {
        if (!vi.IsPlanar() || !vi.IsYUV() || vi.IsYUVA())
            env->ThrowError("[KFMSuper]: planar YUV input required");
        srcvi = vi;
        logUVx = vi.GetPlaneWidthSubsampling(PLANAR_U);
        logUVy = vi.GetPlaneHeightSubsampling(PLANAR_U);
        px = vi.ComponentSize();
        if (vi.width & 7) env->ThrowError("[KFMSuper]: width must be multiple of 8");
        if (vi.height & 7) env->ThrowError("[KFMSuper]: height must be multiple of 8");
        if (px != 1 && px != 2)
            env->ThrowError("[KFMSuper] Unsupported pixel format");
        if (vi.pixel_type == VideoInfo::CS_YV411)
            env->ThrowError("[KFMSuper] Unsupported pixel format (no 8-bit AviSynth+ type)");

        // upstream ctor vi mutation (verbatim)
        vi.width = vi.width / 4 * 2;
        vi.height = vi.height / 4 * 2;
        vi.pixel_type = get8BitType(srcvi);

        try {
            program();  // build the PX instantiation up front (HOST_CONTRACT §1)
        } catch (const std::exception& e) {
            env->ThrowError("[KFMSuperOCL] OpenCL init failed: %s", e.what());
        }
    }

    Program& program() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "kfm_combinganalyze", kfm_combinganalyze_cl_src, px);
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            return GetFrameT(n, env_);
        } catch (const std::exception& e) {
            env_->ThrowError("[KFMSuperOCL] %s", e.what());
        }
        return PVideoFrame();
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) {
        if (cachehints == CACHE_GET_MTMODE) return MT_NICE_FILTER;
        return 0;
    }

private:
    // upstream AnalyzeFrame for one plane: f0/f1 are plain-pitch staged
    // planes (width x height, px-sized elements); the flag pairs land in
    // flagX/flagY (element pitch fpitch; upstream's uchar2 view split).
    void analyzePlane(Plane& flagX, Plane& flagY, int fpitch,
                      const void* f0p, const void* f1p, int pitchElems,
                      int nBlkX, int nBlkY, int shift, int parity) {
        Context& ctx = sharedContext();
        Plane f0, f1;
        f0.alloc(ctx, nBlkX * 4, nBlkY * 4, px);
        f1.alloc(ctx, nBlkX * 4, nBlkY * 4, px);
        if (px == 1) {
            stagePlane<uint8_t>(stage8_, (const uint8_t*)f0p, pitchElems, nBlkX * 4, nBlkY * 4);
            f0.buf.write(stage8_.data(), 0, f0.bytes());
            stagePlane<uint8_t>(stage8_, (const uint8_t*)f1p, pitchElems, nBlkX * 4, nBlkY * 4);
            f1.buf.write(stage8_.data(), 0, f1.bytes());
        } else {
            stagePlane<uint16_t>(stage16_, (const uint16_t*)f0p, pitchElems, nBlkX * 4, nBlkY * 4);
            f0.buf.write(stage16_.data(), 0, f0.bytes());
            stagePlane<uint16_t>(stage16_, (const uint16_t*)f1p, pitchElems, nBlkX * 4, nBlkY * 4);
            f1.buf.write(stage16_.data(), 0, f1.bytes());
        }

        cl_kernel k = program().kernel("kf_super_analyze");
        int a = 0;
        oclhost::set_arg(k, a++, flagX.mem());
        oclhost::set_arg(k, a++, flagY.mem());
        oclhost::set_arg(k, a++, fpitch);
        oclhost::set_arg(k, a++, f0.mem());
        oclhost::set_arg(k, a++, f1.mem());
        oclhost::set_arg(k, a++, nBlkX * 4);  // pixel pitch (plain staged plane)
        oclhost::set_arg(k, a++, nBlkX);
        oclhost::set_arg(k, a++, nBlkY);
        oclhost::set_arg(k, a++, shift);
        oclhost::set_arg(k, a++, parity);
        oclhost::dispatch2D(ctx, k, nBlkX - 1, nBlkY - 1);
    }

    PVideoFrame GetFrameT(int n, IScriptEnvironment* env_) {
        const int w = srcvi.width, h = srcvi.height;
        const int wUV = w >> logUVx, hUV = h >> logUVy;
        const int shift = srcvi.BitsPerComponent() - 8 + 4;

        std::lock_guard<std::mutex> lock(queueMutex());
        Context& ctx = sharedContext();

        int parity = child->GetParity(n) ? 1 : 0;

        // f0/f1 = Frame(padclip->GetFrame(n / n+1), VPAD): the +4-row ROI.
        PVideoFrame f0f = padclip->GetFrame(n, env_);
        PVideoFrame f1f = padclip->GetFrame(n + 1, env_);

        // flag element dims (the uchar2 views of the output planes):
        //   Y: feW x feH = (w/4, h/2);  U/V: (w/4>>logUVx, h/2>>logUVy)
        const int feW = w / 4, feH = h / 2;
        const int feWUV = feW >> logUVx, feHUV = feH >> logUVy;

        Plane flagXY, flagYY, flagXU, flagYU, flagXV, flagYV;
        flagXY.alloc(ctx, feW, feH, 1);
        flagYY.alloc(ctx, feW, feH, 1);
        flagXU.alloc(ctx, feWUV, feHUV, 1);
        flagYU.alloc(ctx, feWUV, feHUV, 1);
        flagXV.alloc(ctx, feWUV, feHUV, 1);
        flagYV.alloc(ctx, feWUV, feHUV, 1);

        const uint8_t* f0Y = f0f->GetReadPtr(PLANAR_Y) + (size_t)VPAD * f0f->GetPitch(PLANAR_Y);
        const uint8_t* f1Y = f1f->GetReadPtr(PLANAR_Y) + (size_t)VPAD * f1f->GetPitch(PLANAR_Y);
        // upstream AnalyzeFrame: nBlkX/nBlkY = the uchar2 element dims >> 1
        // per plane (4x4-stride block cells); the flag buffers keep the full
        // packed-plane element grids
        analyzePlane(flagXY, flagYY, feW, f0Y, f1Y,
                     f0f->GetPitch(PLANAR_Y) / px, feW, feH >> 1, shift, parity);
        const uint8_t* f0U = f0f->GetReadPtr(PLANAR_U) +
            (size_t)(VPAD >> logUVy) * f0f->GetPitch(PLANAR_U);
        const uint8_t* f1U = f1f->GetReadPtr(PLANAR_U) +
            (size_t)(VPAD >> logUVy) * f1f->GetPitch(PLANAR_U);
        analyzePlane(flagXU, flagYU, feWUV, f0U, f1U,
                     f0f->GetPitch(PLANAR_U) / px, feWUV, feHUV >> 1, shift, parity);
        const uint8_t* f0V = f0f->GetReadPtr(PLANAR_V) +
            (size_t)(VPAD >> logUVy) * f0f->GetPitch(PLANAR_V);
        const uint8_t* f1V = f1f->GetReadPtr(PLANAR_V) +
            (size_t)(VPAD >> logUVy) * f1f->GetPitch(PLANAR_V);
        analyzePlane(flagXV, flagYV, feWUV, f0V, f1V,
                     f0f->GetPitch(PLANAR_V) / px, feWUV, feHUV >> 1, shift, parity);

        ctx.finish();

        // download + interleave the flag pairs into the packed output planes
        // (the flag pairs are always bytes; only the SOURCE staging differs)
        PVideoFrame dst = env_->NewVideoFrame(vi);
        {
            std::vector<uint8_t> yv;
            stB_.resize(flagYY.bytes());
            if (stage8_.size() < flagXY.bytes()) stage8_.resize(flagXY.bytes());
            flagXY.buf.read(stage8_.data(), 0, flagXY.bytes());
            flagYY.buf.read(stB_.data(), 0, flagYY.bytes());
            interleave(yv, stage8_, stB_, feW, feH);
            downloadPlane<uint8_t>(dst, PLANAR_Y, yv, feW * 2, feH);
            flagXU.buf.read(stage8_.data(), 0, flagXU.bytes());
            flagYU.buf.read(stB_.data(), 0, flagYU.bytes());
            interleave(yv, stage8_, stB_, feWUV, feHUV);
            downloadPlane<uint8_t>(dst, PLANAR_U, yv, feWUV * 2, feHUV);
            flagXV.buf.read(stage8_.data(), 0, flagXV.bytes());
            flagYV.buf.read(stB_.data(), 0, flagYV.bytes());
            interleave(yv, stage8_, stB_, feWUV, feHUV);
            downloadPlane<uint8_t>(dst, PLANAR_V, yv, feWUV * 2, feHUV);
        }
        return dst;
    }

    std::vector<uint8_t> stage8_;    // staging (max plane)
    std::vector<uint16_t> stage16_;  // 16-bit staging (max plane)
    std::vector<uint8_t> stB_;       // second download scratch
};

// ---------------------------------------------------------------------------
// KCleanSuperOCL — upstream KCleanSuper (CleanSuper per plane).
// Input/output: pair-packed 8-bit "super" frames (uchar2 views).
// ---------------------------------------------------------------------------
class KCleanSuperOCL : public GenericVideoFilter {
public:
    int thY, thC;

    KCleanSuperOCL(PClip clip, int thY_, int thC_, IScriptEnvironment* env)
        : GenericVideoFilter(clip), thY(thY_), thC(thC_) {
        if (!vi.IsPlanar() || !vi.IsYUV() || vi.IsYUVA())
            env->ThrowError("[KCleanSuper]: planar YUV input required");
        logUVx = vi.GetPlaneWidthSubsampling(PLANAR_U);
        logUVy = vi.GetPlaneHeightSubsampling(PLANAR_U);
        try {
            program();
        } catch (const std::exception& e) {
            env->ThrowError("[KCleanSuperOCL] OpenCL init failed: %s", e.what());
        }
    }

    Program& program() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "kfm_combinganalyze", kfm_combinganalyze_cl_src, 1);
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            return GetFrameT(n, env_);
        } catch (const std::exception& e) {
            env_->ThrowError("[KCleanSuperOCL] %s", e.what());
        }
        return PVideoFrame();
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) {
        if (cachehints == CACHE_GET_MTMODE) return MT_NICE_FILTER;
        return 0;
    }

private:
    // upstream CleanSuper for one plane (prev/cur = the .x/.y split planes)
    void cleanSuper(Plane& dstX, Plane& dstY,
                    const std::vector<uint8_t>& prevX, const std::vector<uint8_t>& prevY,
                    const std::vector<uint8_t>& curX, const std::vector<uint8_t>& curY,
                    int width, int height, int thresh) {
        Context& ctx = sharedContext();
        Plane pvX, pvY, cuX, cuY;
        pvX.alloc(ctx, width, height, 1);
        pvY.alloc(ctx, width, height, 1);
        cuX.alloc(ctx, width, height, 1);
        cuY.alloc(ctx, width, height, 1);
        pvX.buf.write(prevX.data(), 0, pvX.bytes());
        pvY.buf.write(prevY.data(), 0, pvY.bytes());
        cuX.buf.write(curX.data(), 0, cuX.bytes());
        cuY.buf.write(curY.data(), 0, cuY.bytes());

        cl_kernel k = program().kernel("kf_clean_super");
        int a = 0;
        oclhost::set_arg(k, a++, dstX.mem());
        oclhost::set_arg(k, a++, dstY.mem());
        oclhost::set_arg(k, a++, pvX.mem());
        oclhost::set_arg(k, a++, pvY.mem());
        oclhost::set_arg(k, a++, cuX.mem());
        oclhost::set_arg(k, a++, cuY.mem());
        oclhost::set_arg(k, a++, width);
        oclhost::set_arg(k, a++, height);
        oclhost::set_arg(k, a++, width);  // pitch == width (plain planes)
        oclhost::set_arg(k, a++, thresh);
        oclhost::dispatch2D(ctx, k, width, height);
    }

    PVideoFrame GetFrameT(int n, IScriptEnvironment* env_) {
        std::lock_guard<std::mutex> lock(queueMutex());
        Context& ctx = sharedContext();

        PVideoFrame prevf = child->GetFrame(n - 1, env_);
        PVideoFrame curf = child->GetFrame(n, env_);
        PVideoFrame dst = env_->NewVideoFrame(vi);

        // The pair-packed "super" frames are uchar2 views: the element grid
        // is (row bytes / 2) x height, exactly upstream's
        // cur.GetWidth<uchar2>(PLANAR_*)/cur.GetHeight(PLANAR_*) (DEVIATION 1).
        const int eW = prevf->GetRowSize(PLANAR_Y) / 2, eH = prevf->GetHeight(PLANAR_Y);
        const int eWUV = prevf->GetRowSize(PLANAR_U) / 2, eHUV = prevf->GetHeight(PLANAR_U);

        // per plane: deinterleave the pair-packed planes into .x/.y
        deinterleave(prevX_, prevY_, prevf->GetReadPtr(PLANAR_Y),
                     prevf->GetPitch(PLANAR_Y), eW, eH);
        deinterleave(curX_, curY_, curf->GetReadPtr(PLANAR_Y),
                     curf->GetPitch(PLANAR_Y), eW, eH);
        dstX_.alloc(ctx, eW, eH, 1);
        dstY_.alloc(ctx, eW, eH, 1);
        cleanSuper(dstX_, dstY_, prevX_, prevY_, curX_, curY_, eW, eH, thY);
        outX_.resize(dstX_.bytes()); outY_.resize(dstY_.bytes());
        dstX_.buf.read(outX_.data(), 0, dstX_.bytes());
        dstY_.buf.read(outY_.data(), 0, dstY_.bytes());
        interleave(packed_, outX_, outY_, eW, eH);
        downloadPlane<uint8_t>(dst, PLANAR_Y, packed_, eW * 2, eH);

        deinterleave(prevX_, prevY_, prevf->GetReadPtr(PLANAR_U),
                     prevf->GetPitch(PLANAR_U), eWUV, eHUV);
        deinterleave(curX_, curY_, curf->GetReadPtr(PLANAR_U),
                     curf->GetPitch(PLANAR_U), eWUV, eHUV);
        dstX_.alloc(ctx, eWUV, eHUV, 1);
        dstY_.alloc(ctx, eWUV, eHUV, 1);
        cleanSuper(dstX_, dstY_, prevX_, prevY_, curX_, curY_, eWUV, eHUV, thC);
        outX_.resize(dstX_.bytes()); outY_.resize(dstY_.bytes());
        dstX_.buf.read(outX_.data(), 0, dstX_.bytes());
        dstY_.buf.read(outY_.data(), 0, dstY_.bytes());
        interleave(packed_, outX_, outY_, eWUV, eHUV);
        downloadPlane<uint8_t>(dst, PLANAR_U, packed_, eWUV * 2, eHUV);

        deinterleave(prevX_, prevY_, prevf->GetReadPtr(PLANAR_V),
                     prevf->GetPitch(PLANAR_V), eWUV, eHUV);
        deinterleave(curX_, curY_, curf->GetReadPtr(PLANAR_V),
                     curf->GetPitch(PLANAR_V), eWUV, eHUV);
        dstX_.alloc(ctx, eWUV, eHUV, 1);
        dstY_.alloc(ctx, eWUV, eHUV, 1);
        cleanSuper(dstX_, dstY_, prevX_, prevY_, curX_, curY_, eWUV, eHUV, thC);
        outX_.resize(dstX_.bytes()); outY_.resize(dstY_.bytes());
        dstX_.buf.read(outX_.data(), 0, dstX_.bytes());
        dstY_.buf.read(outY_.data(), 0, dstY_.bytes());
        interleave(packed_, outX_, outY_, eWUV, eHUV);
        downloadPlane<uint8_t>(dst, PLANAR_V, packed_, eWUV * 2, eHUV);

        ctx.finish();
        return dst;
    }

    int logUVx, logUVy;
    std::vector<uint8_t> prevX_, prevY_, curX_, curY_;
    std::vector<uint8_t> outX_, outY_, packed_;
    Plane dstX_, dstY_;
};

// ---------------------------------------------------------------------------
// KPreCycleAnalyzeOCL — upstream KPreCycleAnalyze (FMCount census).
// ---------------------------------------------------------------------------
class KPreCycleAnalyzeOCL : public GenericVideoFilter {
public:
    VideoInfo combevi;
    int threshMY, threshSY, threshMC, threshSC;

    KPreCycleAnalyzeOCL(PClip super, int threshMY_, int threshSY_, int threshMC_,
                        int threshSC_, IScriptEnvironment* env)
        : GenericVideoFilter(super), threshMY(threshMY_), threshSY(threshSY_),
          threshMC(threshMC_), threshSC(threshSC_) {
        if (!vi.IsPlanar() || !vi.IsYUV() || vi.IsYUVA())
            env->ThrowError("[KPreCycleAnalyze]: planar YUV input required");
        combevi = vi;

        // upstream ctor vi mutation (verbatim)
        int out_bytes = (int)sizeof(FMCount) * 2;
        vi.pixel_type = VideoInfo::CS_BGR32;
        vi.width = 6;
        vi.height = nblocks(out_bytes, vi.width * 4);
        vi.MulDivFPS(1, 2);
        vi.num_frames /= 2;

        try {
            program();
        } catch (const std::exception& e) {
            env->ThrowError("[KPreCycleAnalyzeOCL] OpenCL init failed: %s", e.what());
        }
    }

    Program& program() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "kfm_combinganalyze", kfm_combinganalyze_cl_src, 1);
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            return GetFrameT(n, env_);
        } catch (const std::exception& e) {
            env_->ThrowError("[KPreCycleAnalyzeOCL] %s", e.what());
        }
        return PVideoFrame();
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) {
        if (cachehints == CACHE_GET_MTMODE) return MT_NICE_FILTER;
        return 0;
    }

private:
    PVideoFrame GetFrameT(int n, IScriptEnvironment* env_) {
        std::lock_guard<std::mutex> lock(queueMutex());
        Context& ctx = sharedContext();

        int parity = child->GetParity(0) ? 1 : 0;
        PVideoFrame combe0 = child->GetFrame(2 * n + 0, env_);
        PVideoFrame combe1 = child->GetFrame(2 * n + 1, env_);

        PVideoFrame dst = env_->NewVideoFrame(vi);

        // upstream: width/height = the uchar2 element dims - 1; the interior
        // (+1,+1) origin is expressed as a launch offset (DEVIATION 2).
        const int We = combe0->GetRowSize(PLANAR_Y) / 2;
        const int He = combe0->GetHeight(PLANAR_Y);
        const int WeU = combe0->GetRowSize(PLANAR_U) / 2;
        const int HeU = combe0->GetHeight(PLANAR_U);
        const int width = We - 1, height = He - 1;
        const int widthUV = WeU - 1, heightUV = HeU - 1;

        deinterleave(c0x_, c0y_, combe0->GetReadPtr(PLANAR_Y),
                     combe0->GetPitch(PLANAR_Y), We, He);
        deinterleave(c1x_, c1y_, combe1->GetReadPtr(PLANAR_Y),
                     combe1->GetPitch(PLANAR_Y), We, He);
        deinterleave(c0Ux_, c0Uy_, combe0->GetReadPtr(PLANAR_U),
                     combe0->GetPitch(PLANAR_U), WeU, HeU);
        deinterleave(c1Ux_, c1Uy_, combe1->GetReadPtr(PLANAR_U),
                     combe1->GetPitch(PLANAR_U), WeU, HeU);
        deinterleave(c0Vx_, c0Vy_, combe0->GetReadPtr(PLANAR_V),
                     combe0->GetPitch(PLANAR_V), WeU, HeU);
        deinterleave(c1Vx_, c1Vy_, combe1->GetReadPtr(PLANAR_V),
                     combe1->GetPitch(PLANAR_V), WeU, HeU);

        Plane p0x, p0y, p1x, p1y, p0Ux, p0Uy, p1Ux, p1Uy, p0Vx, p0Vy, p1Vx, p1Vy;
        p0x.alloc(ctx, We, He, 1);  p0y.alloc(ctx, We, He, 1);
        p1x.alloc(ctx, We, He, 1);  p1y.alloc(ctx, We, He, 1);
        p0x.buf.write(c0x_.data(), 0, p0x.bytes());
        p0y.buf.write(c0y_.data(), 0, p0y.bytes());
        p1x.buf.write(c1x_.data(), 0, p1x.bytes());
        p1y.buf.write(c1y_.data(), 0, p1y.bytes());
        p0Ux.alloc(ctx, WeU, HeU, 1);  p0Uy.alloc(ctx, WeU, HeU, 1);
        p1Ux.alloc(ctx, WeU, HeU, 1);  p1Uy.alloc(ctx, WeU, HeU, 1);
        p0Vx.alloc(ctx, WeU, HeU, 1);  p0Vy.alloc(ctx, WeU, HeU, 1);
        p1Vx.alloc(ctx, WeU, HeU, 1);  p1Vy.alloc(ctx, WeU, HeU, 1);
        p0Ux.buf.write(c0Ux_.data(), 0, p0Ux.bytes());
        p0Uy.buf.write(c0Uy_.data(), 0, p0Uy.bytes());
        p1Ux.buf.write(c1Ux_.data(), 0, p1Ux.bytes());
        p1Uy.buf.write(c1Uy_.data(), 0, p1Uy.bytes());
        p0Vx.buf.write(c0Vx_.data(), 0, p0Vx.bytes());
        p0Vy.buf.write(c0Vy_.data(), 0, p0Vy.bytes());
        p1Vx.buf.write(c1Vx_.data(), 0, p1Vx.bytes());
        p1Vy.buf.write(c1Vy_.data(), 0, p1Vy.bytes());

        // FMCount[2]: the .cl port splits the struct into THREE int arrays of
        // the two parity slots (move/shima/lshima) — three separate device
        // buffers, because a whole-buffer cl_mem argument cannot express the
        // upstream +8/+16-byte field offsets inside one FMCount[2] allocation
        // (the assembled output bytes are identical).
        Plane fmMove, fmShima, fmLshima;
        fmMove.alloc(ctx, 2, 1, 4);
        fmShima.alloc(ctx, 2, 1, 4);
        fmLshima.alloc(ctx, 2, 1, 4);

        // upstream prmY = (threshMY, threshSY, threshSY*3),
        //          prmC = (threshMC, threshSC, threshSC*3)
        int thMY = threshMY, thSY = threshSY, thLSY = threshSY * 3;
        int thMC = threshMC, thSC = threshSC, thLSC = threshSC * 3;

        {
            cl_kernel k = program().kernel("kf_init_fmcount");
            oclhost::set_arg(k, 0, fmMove.mem());
            oclhost::set_arg(k, 1, fmShima.mem());
            oclhost::set_arg(k, 2, fmLshima.mem());
            oclhost::dispatch1D(ctx, k, 2);  // <<<1,2>>> upstream
        }

        // ---- Y census: ONE kf_count_cmflags over the interior ----
        {
            cl_kernel k = program().kernel("kf_count_cmflags");
            int a = 0;
            oclhost::set_arg(k, a++, fmMove.mem());
            oclhost::set_arg(k, a++, fmShima.mem());
            oclhost::set_arg(k, a++, fmLshima.mem());
            oclhost::set_arg(k, a++, p0x.mem());
            oclhost::set_arg(k, a++, p0y.mem());
            oclhost::set_arg(k, a++, p1x.mem());
            oclhost::set_arg(k, a++, p1y.mem());
            oclhost::set_arg(k, a++, We);
            oclhost::set_arg(k, a++, We);   // guard-defeat width (DEVIATION 2)
            oclhost::set_arg(k, a++, He);   // guard-defeat height
            oclhost::set_arg(k, a++, parity);
            oclhost::set_arg(k, a++, thMY);
            oclhost::set_arg(k, a++, thSY);
            oclhost::set_arg(k, a++, thLSY);
            oclhost::dispatch2DLocalOffset(ctx, k, 1, 1, width, height, 32, 16);
        }

        // ---- U+V census: ONE kf_count_cmflags_2planes ----
        {
            cl_kernel k = program().kernel("kf_count_cmflags_2planes");
            int a = 0;
            oclhost::set_arg(k, a++, fmMove.mem());
            oclhost::set_arg(k, a++, fmShima.mem());
            oclhost::set_arg(k, a++, fmLshima.mem());
            oclhost::set_arg(k, a++, p0Ux.mem());
            oclhost::set_arg(k, a++, p0Uy.mem());
            oclhost::set_arg(k, a++, p1Ux.mem());
            oclhost::set_arg(k, a++, p1Uy.mem());
            oclhost::set_arg(k, a++, p0Vx.mem());
            oclhost::set_arg(k, a++, p0Vy.mem());
            oclhost::set_arg(k, a++, p1Vx.mem());
            oclhost::set_arg(k, a++, p1Vy.mem());
            oclhost::set_arg(k, a++, WeU);
            oclhost::set_arg(k, a++, WeU);
            oclhost::set_arg(k, a++, HeU);
            oclhost::set_arg(k, a++, parity);
            oclhost::set_arg(k, a++, thMC);
            oclhost::set_arg(k, a++, thSC);
            oclhost::set_arg(k, a++, thLSC);
            oclhost::dispatch2DLocalOffset(ctx, k, 1, 1, widthUV, heightUV, 32, 16);
        }

        ctx.finish();

        // read the three field arrays and assemble FMCount[2] (upstream
        // layout: FMCount[k] = {move[k], shima[k], lshima[k]}, k = parity
        // slot = i ^ !parity inside the kernels)
        int mv[2], sh[2], ls[2];
        fmMove.buf.read(mv, 0, sizeof(mv));
        fmShima.buf.read(sh, 0, sizeof(sh));
        fmLshima.buf.read(ls, 0, sizeof(ls));
        FMCount* out = (FMCount*)dst->GetWritePtr();
        out[0].move = mv[0]; out[0].shima = sh[0]; out[0].lshima = ls[0];
        out[1].move = mv[1]; out[1].shima = sh[1]; out[1].lshima = ls[1];
        return dst;
    }

    std::vector<uint8_t> c0x_, c0y_, c1x_, c1y_;
    std::vector<uint8_t> c0Ux_, c0Uy_, c1Ux_, c1Uy_, c0Vx_, c0Vy_, c1Vx_, c1Vy_;
};

// ---------------------------------------------------------------------------
// KPreCycleAnalyzeShowOCL — upstream KPreCycleAnalyzeShow (pure host).
// ---------------------------------------------------------------------------
class KPreCycleAnalyzeShowOCL : public GenericVideoFilter {
public:
    PClip fmclip;

    KPreCycleAnalyzeShowOCL(PClip fmclip_, PClip source, IScriptEnvironment* env)
        : GenericVideoFilter(source), fmclip(fmclip_) {
        if (vi.pixel_type != VideoInfo::CS_YV12 && vi.pixel_type != VideoInfo::CS_YV16 &&
            vi.pixel_type != VideoInfo::CS_YV24)
            env->ThrowError("[KPreCycleAnalyzeShow]: 8-bit planar YUV input required "
                            "(upstream instantiates DrawText<uint8_t> only)");
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            PVideoFrame fmframe = fmclip->GetFrame(n, env_);
            const FMCount* fmcount = (const FMCount*)fmframe->GetReadPtr();

            PVideoFrame srcf = child->GetFrame(n, env_);
            // upstream MakeWritable's the source frame in place; this port
            // draws on a fresh copy (same pixels — DEVIATION 9).
            PVideoFrame dst = env_->NewVideoFrame(vi);
            stage_.resize((size_t)vi.width * vi.height);
            copyFromFrame(srcf->GetReadPtr(PLANAR_Y), srcf->GetPitch(PLANAR_Y),
                          vi.width, vi.height, stage_.data(), vi.width);
            downloadPlane<uint8_t>(dst, PLANAR_Y, stage_, vi.width, vi.height);
            const int wUV = vi.width >> vi.GetPlaneWidthSubsampling(PLANAR_U);
            const int hUV = vi.height >> vi.GetPlaneHeightSubsampling(PLANAR_U);
            copyFromFrame(srcf->GetReadPtr(PLANAR_U), srcf->GetPitch(PLANAR_U),
                          wUV, hUV, stage_.data(), wUV);
            downloadPlane<uint8_t>(dst, PLANAR_U, stage_, wUV, hUV);
            copyFromFrame(srcf->GetReadPtr(PLANAR_V), srcf->GetPitch(PLANAR_V),
                          wUV, hUV, stage_.data(), wUV);
            downloadPlane<uint8_t>(dst, PLANAR_V, stage_, wUV, hUV);

            char buf[100];
            sprintf(buf, "Pre: [0](%d,%d,%d) [1](%d,%d,%d)",
                    fmcount[0].move, fmcount[0].shima, fmcount[0].lshima,
                    fmcount[1].move, fmcount[1].shima, fmcount[1].lshima);
            drawTextMessage<uint8_t>(dst, vi.BitsPerComponent(), buf);
            return dst;
        } catch (const std::exception& e) {
            env_->ThrowError("[KPreCycleAnalyzeShowOCL] %s", e.what());
        }
        return PVideoFrame();
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) {
        if (cachehints == CACHE_GET_MTMODE) return MT_NICE_FILTER;
        return 0;
    }

private:
    std::vector<uint8_t> stage_;
};

// ---------------------------------------------------------------------------
// KContainsCombeOCL — upstream KContainsCombe (combe-present scan).
// ---------------------------------------------------------------------------
class KContainsCombeOCL : public GenericVideoFilter {
public:
    VideoInfo workvi;

    KContainsCombeOCL(PClip flag, IScriptEnvironment* env)
        : GenericVideoFilter(flag) {
        if (vi.pixel_type != VideoInfo::CS_Y8)
            env->ThrowError("[KContainsCombe]: Y8 flag input required");
        // upstream ctor (verbatim)
        int work_bytes = (int)sizeof(int);
        workvi.pixel_type = VideoInfo::CS_BGR32;
        workvi.width = 4;
        workvi.height = nblocks(work_bytes, workvi.width * 4);
        try {
            program();
        } catch (const std::exception& e) {
            env->ThrowError("[KContainsCombeOCL] OpenCL init failed: %s", e.what());
        }
    }

    Program& program() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "kfm_combinganalyze", kfm_combinganalyze_cl_src, 1);
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            std::lock_guard<std::mutex> lock(queueMutex());
            Context& ctx = sharedContext();

            // flag frame (scanned in full by upstream)
            PVideoFrame flagf = child->GetFrame(n, env_);
            const int fpitch = flagf->GetPitch(PLANAR_Y);

            PVideoFrame dst = env_->NewVideoFrame(workvi);
            Plane work;
            work.alloc(ctx, 1, 1, 4);

            {
                cl_kernel k = program().kernel("kf_init_contains_durty_block");
                oclhost::set_arg(k, 0, work.mem());
                oclhost::dispatch1D(ctx, k, 1);  // <<<1,1>>> upstream
            }
            // scan over the WHOLE flag frame (upstream ContainsDurtyBlock loops
            // bx<width, by<height with width = flag.GetWidth<uint8_t>()), i.e.
            // the crop that WrapSwitchFragFrame applies is NOT part of
            // KContainsCombe.  The kernel's interior-origin crop form is
            // avoided entirely here: a plain (0,0)-origin launch over the tight
            // staging scans exactly the upstream cell set.
            {
                Plane flagp;
                flagp.alloc(ctx, flagf->GetRowSize(PLANAR_Y), flagf->GetHeight(PLANAR_Y), 1);
                stage_.resize((size_t)flagp.w * flagp.h);
                copyFromFrame(flagf->GetReadPtr(PLANAR_Y), fpitch,
                              flagp.w, flagp.h, stage_.data(), flagp.w);
                flagp.buf.write(stage_.data(), 0, flagp.bytes());

                cl_kernel k = program().kernel("kf_contains_durty_block");
                int a = 0;
                oclhost::set_arg(k, a++, flagp.mem());
                oclhost::set_arg(k, a++, flagp.w);   // width  (logical row bytes)
                oclhost::set_arg(k, a++, flagp.h);   // height
                oclhost::set_arg(k, a++, flagp.w);   // pitch (full)
                oclhost::set_arg(k, a++, work.mem());
                oclhost::dispatch2DLocal(ctx, k, flagp.w, flagp.h, 32, 16);
            }

            ctx.finish();
            int workv = 0;
            work.buf.read(&workv, 0, sizeof(workv));
            *(int*)dst->GetWritePtr() = workv;
            return dst;
        } catch (const std::exception& e) {
            env_->ThrowError("[KContainsCombeOCL] %s", e.what());
        }
        return PVideoFrame();
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) {
        if (cachehints == CACHE_GET_MTMODE) return MT_NICE_FILTER;
        return 0;
    }

private:
    std::vector<uint8_t> stage_;
};

// ---------------------------------------------------------------------------
// KCombeMaskOCL — upstream KCombeMask (flag -> mask bilinear upscale).
// ---------------------------------------------------------------------------
class KCombeMaskOCL : public GenericVideoFilter {
public:
    PClip flagclip;

    KCombeMaskOCL(PClip source, PClip flag, IScriptEnvironment* env)
        : GenericVideoFilter(source), flagclip(flag) {
        if (!vi.IsPlanar() || !vi.IsYUV() || vi.IsYUVA())
            env->ThrowError("[KCombeMask]: planar YUV input required");
        logUVx = vi.GetPlaneWidthSubsampling(PLANAR_U);
        logUVy = vi.GetPlaneHeightSubsampling(PLANAR_U);
        VideoInfo flagvi = flagclip->GetVideoInfo();
        // upstream ctor checks (verbatim)
        if ((flagvi.width - COMBE_FLAG_PAD_W * 2) * 8 != vi.width) {
            env->ThrowError("[KCombeMask] width unmatch");
        }
        if ((flagvi.height - COMBE_FLAG_PAD_H * 2) * 8 != vi.height) {
            env->ThrowError("[KCombeMask] height unmatch");
        }
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            // flag ROI = the WrapSwitchFragFrame crop (4, 2)
            PVideoFrame flagf = flagclip->GetFrame(n, env_);
            const int fpitchFull = flagf->GetRowSize(PLANAR_Y);
            const int fwidth = fpitchFull - COMBE_FLAG_PAD_W * 2;
            const int fheight = flagf->GetHeight(PLANAR_Y) - COMBE_FLAG_PAD_H * 2;

            PVideoFrame dst = env_->NewVideoFrame(vi);
            const int widthUV = vi.width >> logUVx, heightUV = vi.height >> logUVy;
            const int scaleUVw = widthUV / fwidth;   // 4 or 8 upstream
            const int scaleUVh = heightUV / fheight;

            // ---- stage the ROI with its 1px mirror ring ----
            stage_.resize((size_t)fpitchFull * flagf->GetHeight(PLANAR_Y));
            copyFromFrame(flagf->GetReadPtr(PLANAR_Y), flagf->GetPitch(PLANAR_Y),
                          fpitchFull, flagf->GetHeight(PLANAR_Y), stage_.data(), fpitchFull);
            const uint8_t* roi = stage_.data() +
                (size_t)COMBE_FLAG_PAD_H * fpitchFull + COMBE_FLAG_PAD_W;
            stageMirrorPad2<uint8_t>(padded_, roi, fpitchFull, fwidth, fheight, 1, 1);

            // ---- Y group (DC_BLOCK_SIZE 8x8) then UV group ----
            bilinearGroupHost(dst, PLANAR_Y, fwidth, fheight, 8, 8, vi.width, vi.height);
            bilinearGroupHost(dst, PLANAR_U, fwidth, fheight, scaleUVw, scaleUVh,
                              widthUV, heightUV);
            bilinearGroupHost(dst, PLANAR_V, fwidth, fheight, scaleUVw, scaleUVh,
                              widthUV, heightUV);
            return dst;
        } catch (const std::exception& e) {
            env_->ThrowError("[KCombeMaskOCL] %s", e.what());
        }
        return PVideoFrame();
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) {
        if (cachehints == CACHE_GET_MTMODE) return MT_NICE_FILTER;
        return 0;
    }

private:
    // Exact upstream CPU twin: cpu_bilinear_h then cpu_bilinear_v over the
    // padv/padh(1) ring.  The device path offsets the h-pass source one row up
    // (flagp - fpitch) and the v-pass source one row down (dsttmp + dpitch),
    // and its x0 = (x-half)>>shift reaches the ring's negative column — an
    // interior-origin pointer argument this driver cannot express with
    // whole-buffer cl_mem args (RIG_HANDOFF §9.4), so both separable passes
    // run in the host staging (bit-identical; DEVIATION 3).  Ring layout:
    // padded_[r][c] = ROI(c-1, r-1); h pass reads ROI(x0, y-1) = R[y][x0+1..+2].
    void bilinearGroupHost(PVideoFrame& dst, int plane, int fwidth, int fheight,
                           int scalew, int scaleh, int dwidth, int dheight) {
        const int dwidthW = scalew * fwidth;
        const int dheightW = scaleh * fheight;
        const int shiftw = (scalew == 8) ? 3 : 2;
        const int shifth = (scaleh == 8) ? 3 : 2;
        const int halfw = scalew / 2, halfh = scaleh / 2;
        const int pw = fwidth + 2;

        std::vector<uint8_t> hout((size_t)dwidthW * (fheight + 2));
        for (int y = 0; y < fheight + 2; ++y) {
            const uint8_t* R = padded_.data() + (size_t)y * pw;
            uint8_t* h = hout.data() + (size_t)y * dwidthW;
            for (int x = 0; x < dwidthW; ++x) {
                int x0 = (x - halfw) >> shiftw;
                int c0 = ((x0 + 1) << shiftw) - (x - halfw);
                int c1 = scalew - c0;
                int s0 = R[x0 + 1];
                int s1 = R[x0 + 2];
                h[x] = (uint8_t)((s0 * c0 + s1 * c1 + halfw) >> shiftw);
            }
        }

        outDev_.resize((size_t)dwidthW * dheightW);
        for (int y = 0; y < dheightW; ++y) {
            int y0 = (y - halfh) >> shifth;
            int c0 = ((y0 + 1) << shifth) - (y - halfh);
            int c1 = scaleh - c0;
            const uint8_t* h0 = hout.data() + (size_t)(y0 + 1) * dwidthW;
            const uint8_t* h1 = hout.data() + (size_t)(y0 + 2) * dwidthW;
            uint8_t* d = outDev_.data() + (size_t)y * dwidthW;
            for (int x = 0; x < dwidthW; ++x)
                d[x] = (uint8_t)((h0[x] * c0 + h1[x] * c1 + halfh) >> shifth);
        }
        downloadPlane<uint8_t>(dst, plane, outDev_, dwidthW, dheightW);
    }

    int logUVx, logUVy;
    std::vector<uint8_t> stage_, padded_, outDev_;
};

// ---------------------------------------------------------------------------
// KRemoveCombeOCL — upstream KRemoveCombe.  DEVIATION 4: the combe-gated
// binomial merge runs as the exact upstream CPU twin (cpu_remove_combe2) in
// the host staging (no device launch — see the header for the reason).
// ---------------------------------------------------------------------------
class KRemoveCombeOCL : public GenericVideoFilter {
public:
    PClip superclip;
    float thY, thC;

    KRemoveCombeOCL(PClip pad, PClip super, float thY_, float thC_, IScriptEnvironment* env)
        : GenericVideoFilter(pad), superclip(super), thY(thY_), thC(thC_) {
        if (!vi.IsPlanar() || !vi.IsYUV() || vi.IsYUVA())
            env->ThrowError("[KRemoveCombe]: planar YUV input required");
        logUVx = vi.GetPlaneWidthSubsampling(PLANAR_U);
        logUVy = vi.GetPlaneHeightSubsampling(PLANAR_U);
        if (vi.width & 7) env->ThrowError("[KRemoveCombe]: width must be multiple of 8");
        if (vi.height & 7) env->ThrowError("[KRemoveCombe]: height must be multiple of 8");

        // superclip check (verbatim)
        VideoInfo supervi = superclip->GetVideoInfo();
        if (supervi.num_frames != vi.num_frames) {
            env->ThrowError("[KRemoveCombe]: padclip and superclip should have the same num_frames");
        }

        // upstream: vi.height -= VPAD * 2 (the pad clip supplies the halo)
        vi.height -= VPAD * 2;
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            const int px = vi.ComponentSize();
            if (px == 1) return GetFrameT<uint8_t>(n, env_);
            if (px == 2) return GetFrameT<uint16_t>(n, env_);
            env_->ThrowError("[KRemoveCombe] Unsupported pixel format");
        } catch (const std::exception& e) {
            env_->ThrowError("[KRemoveCombeOCL] %s", e.what());
        }
        return PVideoFrame();
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) {
        if (cachehints == CACHE_GET_MTMODE) return MT_NICE_FILTER;
        return 0;
    }

private:
    // upstream cpu_remove_combe2 (verbatim twin) for one plane; srcp points
    // at the pad frame's VPAD interior (real row 0); combe bytes are the
    // pair-packed super frame's .x lanes (even bytes), cpitch in pairs.
    template <typename T>
    void removeCombe(T* dstp, int dpitch, const T* srcp, int spitchElems,
                     int width, int height, const uint8_t* combeBytes,
                     int combePitchBytes, int thcombe) {
        const int c_pitch = combePitchBytes / 2;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int score = combeBytes[2 * ((x >> 2) + (y >> 2) * c_pitch)];
                int v;
                if (score >= thcombe) {
                    v = (int)srcp[x + (y - 1) * spitchElems]
                        + 2 * (int)srcp[x + y * spitchElems]
                        + (int)srcp[x + (y + 1) * spitchElems] + 2;
                    v >>= 2;
                } else {
                    v = srcp[x + y * spitchElems];
                }
                dstp[x + y * dpitch] = (T)v;
            }
        }
    }

    template <typename T>
    PVideoFrame GetFrameT(int n, IScriptEnvironment* env_) {
        const int w = vi.width, h = vi.height;
        const int wUV = w >> logUVx, hUV = h >> logUVy;

        PVideoFrame srcf = child->GetFrame(n, env_);       // pad clip (VPAD ring)
        PVideoFrame combef = superclip->GetFrame(n, env_); // pair-packed super
        PVideoFrame dst = env_->NewVideoFrame(vi);

        const T* srcY = (const T*)srcf->GetReadPtr(PLANAR_Y) + (size_t)VPAD * (srcf->GetPitch(PLANAR_Y) / (int)sizeof(T));
        const uint8_t* combY = combef->GetReadPtr(PLANAR_Y);
        removeCombe<T>((T*)dst->GetWritePtr(PLANAR_Y), dst->GetPitch(PLANAR_Y) / (int)sizeof(T),
                       srcY, srcf->GetPitch(PLANAR_Y) / (int)sizeof(T),
                       w, h, combY, combef->GetPitch(PLANAR_Y), (int)thY);
        const T* srcU = (const T*)srcf->GetReadPtr(PLANAR_U) +
            (size_t)(VPAD >> logUVy) * (srcf->GetPitch(PLANAR_U) / (int)sizeof(T));
        removeCombe<T>((T*)dst->GetWritePtr(PLANAR_U), dst->GetPitch(PLANAR_U) / (int)sizeof(T),
                       srcU, srcf->GetPitch(PLANAR_U) / (int)sizeof(T),
                       wUV, hUV, combef->GetReadPtr(PLANAR_U), combef->GetPitch(PLANAR_U),
                       (int)thC);
        const T* srcV = (const T*)srcf->GetReadPtr(PLANAR_V) +
            (size_t)(VPAD >> logUVy) * (srcf->GetPitch(PLANAR_V) / (int)sizeof(T));
        removeCombe<T>((T*)dst->GetWritePtr(PLANAR_V), dst->GetPitch(PLANAR_V) / (int)sizeof(T),
                       srcV, srcf->GetPitch(PLANAR_V) / (int)sizeof(T),
                       wUV, hUV, combef->GetReadPtr(PLANAR_V), combef->GetPitch(PLANAR_V),
                       (int)thC);
        return dst;
    }

    int logUVx, logUVy;
};

// ===========================================================================
// DecombeUCF.cu family
// ===========================================================================

// ---------------------------------------------------------------------------
// KFieldDiffOCL — upstream KFieldDiff (KCFieldDiff conditional function).
// ---------------------------------------------------------------------------
class KFieldDiffOCL {
public:
    int nt6;
    bool chroma;
    VideoInfo vi;
    int logUVx, logUVy;

    KFieldDiffOCL(PClip clip, float nt, bool chroma_) : chroma(chroma_) {
        vi = clip->GetVideoInfo();
        logUVx = vi.GetPlaneWidthSubsampling(PLANAR_U);
        logUVy = vi.GetPlaneHeightSubsampling(PLANAR_U);
        nt6 = scaleParam(nt * 6, vi.BitsPerComponent());
    }

    Program& program() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "kfm_decombeucf", kfm_decombeucf_cl_src,
            vi.ComponentSize());
    }

    // upstream CalcFieldDiff: one ulong sum over the (chroma-extended) frame
    unsigned long long calcFieldDiff(PClip child, int n, IScriptEnvironment* env_) {
        std::lock_guard<std::mutex> lock(queueMutex());
        Context& ctx = sharedContext();
        Program& prog = program();

        PVideoFrame src = child->GetFrame(n, env_);

        const int w = vi.width, h = vi.height;
        const int wUV = w >> logUVx, hUV = h >> logUVy;
        const int px = vi.ComponentSize();

        Plane sum;
        sum.alloc(ctx, 1, 1, 8);
        {
            cl_kernel k = prog.kernel("kf_init_uint64");
            oclhost::set_arg(k, 0, sum.mem());
            oclhost::set_arg(k, 1, 1);
            oclhost::dispatch1D(ctx, k, 1);
        }

        // per plane: stage the mirror ring (+-2 rows) and launch the gated
        // CalcCombe sum through a +2-row work offset (DEVIATION 5)
        fieldDiffPlane(prog, sum, src, PLANAR_Y, w, h, px, stage_, stage16_);
        if (chroma) {
            fieldDiffPlane(prog, sum, src, PLANAR_U, wUV, hUV, px, stage_, stage16_);
            fieldDiffPlane(prog, sum, src, PLANAR_V, wUV, hUV, px, stage_, stage16_);
        }
        ctx.finish();

        unsigned long long raw = 0;
        sum.buf.read(&raw, 0, sizeof(raw));
        return raw;
    }

    // upstream InternalFieldDiff
    double internalFieldDiff(PClip child, int n, IScriptEnvironment* env_) {
        unsigned long long raw = calcFieldDiff(child, n, env_);
        raw /= 6;  // upstream comment: 概算補正
        int shift = vi.BitsPerComponent() - 8;
        return (double)(raw >> shift);
    }

private:
    void fieldDiffPlane(Program& prog, Plane& sum, PVideoFrame& src, int plane,
                        int width, int height, int px,
                        std::vector<uint8_t>& st8, std::vector<uint16_t>& st16) {
        Context& ctx = sharedContext();
        // stage the plane with a +-2-row mirror ring; the buffer's row j holds
        // the padded row j (real row j-2), so the kernel's (y-2)..(y+2) taps
        // at global y = real row r + 2 are exactly the real rows r-2..r+2.
        const int pitchPx = width;
        Plane padded;
        padded.alloc(ctx, width, height + 4, px);
        if (px == 1) {
            stageMirrorPadV<uint8_t>(st8, src->GetReadPtr(plane), src->GetPitch(plane),
                                     width, height, 2);
            padded.buf.write(st8.data(), 0, padded.bytes());
        } else {
            stageMirrorPadV<uint16_t>(st16, (const uint16_t*)src->GetReadPtr(plane),
                                      src->GetPitch(plane) / 2, width, height, 2);
            padded.buf.write(st16.data(), 0, padded.bytes());
        }

        cl_kernel k = prog.kernel("kf_calculate_field_diff");
        int a = 0;
        oclhost::set_arg(k, a++, padded.mem());
        oclhost::set_arg(k, a++, nt6);
        oclhost::set_arg(k, a++, width);
        oclhost::set_arg(k, a++, height + 2);  // guard-defeat (DEVIATION 5)
        oclhost::set_arg(k, a++, pitchPx);
        oclhost::set_arg(k, a++, sum.mem());
        oclhost::dispatch2DLocalOffset(ctx, k, 0, 2, width, height, 32, 16);
    }

    std::vector<uint8_t> stage_;
    std::vector<uint16_t> stage16_;
};

// ---------------------------------------------------------------------------
// KFrameDiffDupOCL — upstream KFrameDiffDup (KCFrameDiffDup conditional).
// ---------------------------------------------------------------------------
class KFrameDiffDupOCL {
public:
    bool chroma;
    int blocksize;

    int logUVx, logUVy;
    int th_z, th_uv_z;
    int blocks_w, blocks_h, block_pitch;
    VideoInfo vi;

    enum { THREADS = 256 };

    KFrameDiffDupOCL(PClip clip, bool chroma_, int blocksize_) : chroma(chroma_),
        blocksize(blocksize_) {
        vi = clip->GetVideoInfo();
        logUVx = vi.GetPlaneWidthSubsampling(PLANAR_U);
        logUVy = vi.GetPlaneHeightSubsampling(PLANAR_U);
        if (blocksize != 4 && blocksize != 8 && blocksize != 16 && blocksize != 32)
            throw std::runtime_error("[KFrameDiffDup] blocksize must be 4/8/16/32");

        blocks_w = nblocks(vi.width, blocksize);
        blocks_h = nblocks(vi.height, blocksize);

        th_z = THREADS / (blocksize * (blocksize / 4));
        th_uv_z = th_z * (chroma ? (1 << (logUVx + logUVy)) : 1);

        int block_align = th_uv_z > 4 ? th_uv_z : 4;
        block_pitch = nblocks(blocks_w, block_align) * block_align;
    }

    Program& program() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "kfm_decombeucf", kfm_decombeucf_cl_src,
            vi.ComponentSize());
    }

    // upstream CalcFrameDiff: returns argmax(subAbs + sumSig * 4)
    int calcFrameDiff(PClip child, int n, IScriptEnvironment* env_) {
        std::lock_guard<std::mutex> lock(queueMutex());
        Context& ctx = sharedContext();
        Program& prog = program();

        PVideoFrame src0 = child->GetFrame(n - 1 < 0 ? 0 : (n - 1 > vi.num_frames - 1 ? vi.num_frames - 1 : n - 1), env_);
        PVideoFrame src1 = child->GetFrame(n < 0 ? 0 : (n > vi.num_frames - 1 ? vi.num_frames - 1 : n), env_);

        const int w = vi.width, h = vi.height;
        const int wUV = w >> logUVx, hUV = h >> logUVy;
        const int px = vi.ComponentSize();
        const int len = block_pitch * blocks_h;

        // work: sumAbs[len] | sumSig[len] | maxSum[1] — one int buffer; the
        // three sections are addressed from the base in the kernel args via
        // SEPARATE buffers (a whole-buffer arg cannot express the +len
        // offset), assembled identically.
        Plane sumAbsP, sumSigP, maxSumP;
        sumAbsP.alloc(ctx, len, 1, 4);
        sumSigP.alloc(ctx, len, 1, 4);
        maxSumP.alloc(ctx, 1, 1, 4);
        {
            cl_kernel k = prog.kernel("kf_init_block_sum");
            oclhost::set_arg(k, 0, sumAbsP.mem());
            oclhost::set_arg(k, 1, sumSigP.mem());
            oclhost::set_arg(k, 2, maxSumP.mem());
            oclhost::set_arg(k, 3, len);
            oclhost::dispatch1D(ctx, k, len);  // upstream <<<64, nblocks(len,64)>>>
        }

        addBlockSum(prog, src0, src1, PLANAR_Y, w, h, px, sumAbsP, sumSigP);
        if (chroma) {
            addBlockSum(prog, src0, src1, PLANAR_U, wUV, hUV, px, sumAbsP, sumSigP);
            addBlockSum(prog, src0, src1, PLANAR_V, wUV, hUV, px, sumAbsP, sumSigP);
        }

        // block_sum_max over the int4-quad view (blocks_w4 = blocks_w>>2)
        {
            int blocks_w4 = blocks_w >> 2;
            int block_pitch4 = block_pitch >> 2;
            cl_kernel k = prog.kernel("kf_block_sum_max");
            oclhost::set_arg(k, 0, sumAbsP.mem());
            oclhost::set_arg(k, 1, sumSigP.mem());
            oclhost::set_arg(k, 2, blocks_w4);
            oclhost::set_arg(k, 3, blocks_h);
            oclhost::set_arg(k, 4, block_pitch4);
            oclhost::set_arg(k, 5, maxSumP.mem());
            if (blocks_w4 > 0)
                oclhost::dispatch2DLocal(ctx, k, blocks_w4, blocks_h, 32, 16);
        }

        ctx.finish();
        int result = 0;
        maxSumP.buf.read(&result, 0, sizeof(result));
        return result;
    }

    // upstream InternalFrameDiff
    double internalFrameDiff(PClip child, int n, IScriptEnvironment* env_) {
        int diff = calcFrameDiff(child, n, env_);
        int shift = vi.BitsPerComponent() - 8;
        // upstream verbatim (the corrected dup232a form)
        return diff / (2.0 * (235 << shift) * blocksize * blocksize) * 100.0;
    }

private:
    void addBlockSum(Program& prog, PVideoFrame& s0f, PVideoFrame& s1f, int plane,
                     int width, int height, int px, Plane& sumAbsP, Plane& sumSigP) {
        Context& ctx = sharedContext();
        Plane p0, p1;
        p0.alloc(ctx, width, height, px);
        p1.alloc(ctx, width, height, px);
        if (px == 1) {
            stagePlane<uint8_t>(st8_, s0f->GetReadPtr(plane), s0f->GetPitch(plane), width, height);
            p0.buf.write(st8_.data(), 0, p0.bytes());
            stagePlane<uint8_t>(st8_, s1f->GetReadPtr(plane), s1f->GetPitch(plane), width, height);
            p1.buf.write(st8_.data(), 0, p1.bytes());
        } else {
            stagePlane<uint16_t>(st16_, (const uint16_t*)s0f->GetReadPtr(plane),
                                 s0f->GetPitch(plane) / 2, width, height);
            p0.buf.write(st16_.data(), 0, p0.bytes());
            stagePlane<uint16_t>(st16_, (const uint16_t*)s1f->GetReadPtr(plane),
                                 s1f->GetPitch(plane) / 2, width, height);
            p1.buf.write(st16_.data(), 0, p1.bytes());
        }

        cl_kernel k = prog.kernel("kf_add_block_sum");
        int a = 0;
        oclhost::set_arg(k, a++, p0.mem());
        oclhost::set_arg(k, a++, p1.mem());
        oclhost::set_arg(k, a++, width);
        oclhost::set_arg(k, a++, height);
        oclhost::set_arg(k, a++, width);  // pitch == width
        oclhost::set_arg(k, a++, blocks_w);
        oclhost::set_arg(k, a++, blocks_h);
        oclhost::set_arg(k, a++, block_pitch);
        oclhost::set_arg(k, a++, sumAbsP.mem());
        oclhost::set_arg(k, a++, sumSigP.mem());
        oclhost::set_arg(k, a++, blocksize);
        oclhost::dispatch2D(ctx, k, blocks_w, blocks_h);
    }

    std::vector<uint8_t> st8_;
    std::vector<uint16_t> st16_;
};

// ---------------------------------------------------------------------------
// KAnalyzeNoiseOCL — upstream KAnalyzeNoise (8-bit only).
// ---------------------------------------------------------------------------
struct NoiseResult {
    unsigned long long noise0, noise1;
    unsigned long long noiseR0, noiseR1;
    unsigned long long diff0, diff1;
};

struct UCFNoiseMeta {
    enum {
        VERSION = 1,
        MAGIC_KEY = 0x39EDF8,
    };
    int nMagicKey;
    int nVersion;

    int srcw, srch;
    int srcUVw, srcUVh;
    int noisew, noiseh;
    int noiseUVw, noiseUVh;

    UCFNoiseMeta()
        : nMagicKey(MAGIC_KEY), nVersion(VERSION) {}
};

class KAnalyzeNoiseOCL : public GenericVideoFilter {
public:
    PClip noiseclip;
    PClip superclip;

    UCFNoiseMeta meta;

    VideoInfo srcvi;

    KAnalyzeNoiseOCL(PClip src, PClip noise, PClip pad, IScriptEnvironment* env)
        : GenericVideoFilter(src), noiseclip(noise), superclip(pad) {
        if (!vi.IsPlanar() || !vi.IsYUV() || vi.IsYUVA())
            env->ThrowError("[KAnalyzeNoise]: planar YUV input required");
        srcvi = vi;
        if (srcvi.pixel_type != VideoInfo::CS_YV12 && srcvi.pixel_type != VideoInfo::CS_YV16 &&
            srcvi.pixel_type != VideoInfo::CS_YV24)
            env->ThrowError("[KAnalyzeNoise] Unsupported pixel format (8-bit only upstream)");
        if (srcvi.width & 3) env->ThrowError("[KAnalyzeNoise]: width must be multiple of 4");
        if (srcvi.height & 3) env->ThrowError("[KAnalyzeNoise]: height must be multiple of 4");

        // upstream ctor (verbatim dims + the UCFNoiseMeta tag)
        int out_bytes = (int)sizeof(NoiseResult) * 2;
        vi.pixel_type = VideoInfo::CS_BGR32;
        vi.width = 16;
        vi.height = nblocks(out_bytes, vi.width * 4);

        VideoInfo noisevi = noiseclip->GetVideoInfo();
        meta.srcw = srcvi.width;
        meta.srch = srcvi.height;
        meta.srcUVw = srcvi.width >> srcvi.GetPlaneWidthSubsampling(PLANAR_U);
        meta.srcUVh = srcvi.height >> srcvi.GetPlaneHeightSubsampling(PLANAR_U);
        meta.noisew = noisevi.width;
        meta.noiseh = noisevi.height;
        meta.noiseUVw = noisevi.width >> noisevi.GetPlaneWidthSubsampling(PLANAR_U);
        meta.noiseUVh = noisevi.height >> noisevi.GetPlaneHeightSubsampling(PLANAR_U);
        // upstream UCFNoiseMeta::SetParam: kill audio, tag sample_type, stash
        // the meta pointer in num_audio_samples (for the unported KDecombUCF*
        // consumers)
        vi.audio_samples_per_second = 0;
        vi.sample_type = UCFNoiseMeta::MAGIC_KEY;
        vi.num_audio_samples = (size_t)&meta;

        try {
            program();
        } catch (const std::exception& e) {
            env->ThrowError("[KAnalyzeNoiseOCL] OpenCL init failed: %s", e.what());
        }
    }

    Program& program() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "kfm_decombeucf", kfm_decombeucf_cl_src, 1);
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            return GetFrameT(n, env_);
        } catch (const std::exception& e) {
            env_->ThrowError("[KAnalyzeNoiseOCL] %s", e.what());
        }
        return PVideoFrame();
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) {
        if (cachehints == CACHE_GET_MTMODE) return MT_NICE_FILTER;
        return 0;
    }

private:
    PVideoFrame GetFrameT(int n, IScriptEnvironment* env_) {
        std::lock_guard<std::mutex> lock(queueMutex());
        Context& ctx = sharedContext();
        Program& prog = program();

        PVideoFrame noise0 = noiseclip->GetFrame(2 * n + 0, env_);
        PVideoFrame noise1 = noiseclip->GetFrame(2 * n + 1, env_);
        PVideoFrame noise2 = noiseclip->GetFrame(2 * n + 2, env_);

        const int w = srcvi.width, h = srcvi.height;
        const int wUV = w >> srcvi.GetPlaneWidthSubsampling(PLANAR_U);
        const int hUV = h >> srcvi.GetPlaneHeightSubsampling(PLANAR_U);

        PVideoFrame dst = env_->NewVideoFrame(vi);

        // DEVIATION 6: four separate u64 accumulators (Y-noise 4, UV-noise 4,
        // Y-diff 2, UV-diff 2 slots) instead of one NoiseResult[2] array
        Plane resYn, resUVn, resYd, resUVd;
        resYn.alloc(ctx, 4, 1, 8);
        resUVn.alloc(ctx, 4, 1, 8);
        resYd.alloc(ctx, 2, 1, 8);
        resUVd.alloc(ctx, 2, 1, 8);
        initUint64(prog, resYn, 4);
        initUint64(prog, resUVn, 4);
        initUint64(prog, resYd, 2);
        initUint64(prog, resUVd, 2);

        // ---- AnalyzeNoise (per plane: Y -> Y record, U/V -> UV record) ----
        analyzeNoisePlane(prog, resYn, noise0, noise1, noise2, PLANAR_Y, w, h);
        analyzeNoisePlane(prog, resUVn, noise0, noise1, noise2, PLANAR_U, wUV, hUV);
        analyzeNoisePlane(prog, resUVn, noise0, noise1, noise2, PLANAR_V, wUV, hUV);

        // ---- f0p/f1p: superclip VPAD ROIs or staged mirror-padded child ----
        PVideoFrame f0f, f1f;
        bool staged = false;
        if (superclip) {
            f0f = superclip->GetFrame(n, env_);
            f1f = superclip->GetFrame(n + 1, env_);
        } else {
            PVideoFrame f0 = child->GetFrame(n, env_);
            PVideoFrame f1 = child->GetFrame(n + 1, env_);
            stageMirrorPadV<uint8_t>(pad0_, f0->GetReadPtr(PLANAR_Y), f0->GetPitch(PLANAR_Y), w, h, 2);
            stageMirrorPadV<uint8_t>(pad1_, f1->GetReadPtr(PLANAR_Y), f1->GetPitch(PLANAR_Y), w, h, 2);
            stageMirrorPadV<uint8_t>(pad0U_, f0->GetReadPtr(PLANAR_U), f0->GetPitch(PLANAR_U), wUV, hUV, 2);
            stageMirrorPadV<uint8_t>(pad1U_, f1->GetReadPtr(PLANAR_U), f1->GetPitch(PLANAR_U), wUV, hUV, 2);
            stageMirrorPadV<uint8_t>(pad0V_, f0->GetReadPtr(PLANAR_V), f0->GetPitch(PLANAR_V), wUV, hUV, 2);
            stageMirrorPadV<uint8_t>(pad1V_, f1->GetReadPtr(PLANAR_V), f1->GetPitch(PLANAR_V), wUV, hUV, 2);
            staged = true;
        }

        // ---- AnalyzeDiff (per plane) ----
        analyzeDiffPlane(prog, resYd, f0f, f1f, PLANAR_Y, w, h, staged);
        analyzeDiffPlane(prog, resUVd, f0f, f1f, PLANAR_U, wUV, hUV, staged);
        analyzeDiffPlane(prog, resUVd, f0f, f1f, PLANAR_V, wUV, hUV, staged);

        ctx.finish();

        // assemble NoiseResult[2] into the output frame (96 bytes; the 32
        // trailing bytes of the 128-byte frame are zero-filled, DEVIATION 6)
        std::vector<uint8_t> out((size_t)dst->GetRowSize() * dst->GetHeight(), 0);
        NoiseResult* nr = (NoiseResult*)out.data();
        resYn.buf.read(&nr[0].noise0, 0, 4 * 8);
        resUVn.buf.read(&nr[1].noise0, 0, 4 * 8);
        resYd.buf.read(&nr[0].diff0, 0, 2 * 8);
        resUVd.buf.read(&nr[1].diff0, 0, 2 * 8);
        memcpy(dst->GetWritePtr(), out.data(), out.size());
        return dst;
    }

    void initUint64(Program& prog, Plane& sum, int length) {
        Context& ctx = sharedContext();
        cl_kernel k = prog.kernel("kf_init_uint64");
        oclhost::set_arg(k, 0, sum.mem());
        oclhost::set_arg(k, 1, length);
        oclhost::dispatch1D(ctx, k, length);
    }

    void analyzeNoisePlane(Program& prog, Plane& result,
                           PVideoFrame& n0, PVideoFrame& n1, PVideoFrame& n2,
                           int plane, int width, int height) {
        Context& ctx = sharedContext();
        Plane p0, p1, p2;
        p0.alloc(ctx, width, height, 1);
        p1.alloc(ctx, width, height, 1);
        p2.alloc(ctx, width, height, 1);
        stagePlane<uint8_t>(st8_, n0->GetReadPtr(plane), n0->GetPitch(plane), width, height);
        p0.buf.write(st8_.data(), 0, p0.bytes());
        stagePlane<uint8_t>(st8_, n1->GetReadPtr(plane), n1->GetPitch(plane), width, height);
        p1.buf.write(st8_.data(), 0, p1.bytes());
        stagePlane<uint8_t>(st8_, n2->GetReadPtr(plane), n2->GetPitch(plane), width, height);
        p2.buf.write(st8_.data(), 0, p2.bytes());

        cl_kernel k = prog.kernel("kf_analyze_noise");
        int a = 0;
        oclhost::set_arg(k, a++, result.mem());
        oclhost::set_arg(k, a++, p0.mem());
        oclhost::set_arg(k, a++, p1.mem());
        oclhost::set_arg(k, a++, p2.mem());
        oclhost::set_arg(k, a++, width);
        oclhost::set_arg(k, a++, height);
        oclhost::set_arg(k, a++, width);
        oclhost::dispatch2DLocal(ctx, k, width, height, 32, 16);
    }

    void analyzeDiffPlane(Program& prog, Plane& result,
                          PVideoFrame& f0f, PVideoFrame& f1f, int plane,
                          int width, int height, bool staged) {
        Context& ctx = sharedContext();
        // Buffer row j holds the row the kernel's (y-2)..(y+2) tap text reads
        // for global y = j+2, so taps for real row r land on buffer rows
        // r..r+4:
        //   staged (child) path: row j = mirror-padded real row j-2 (the
        //     staged ring supplies the halo; DEVIATION 5);
        //   superclip path: row j = super-frame row j+4 (the VPAD ROI plus
        //     the clip's own bottom halo rows — the halo comes from the clip,
        //     exactly as upstream's Frame(..., VPAD) pointer math).
        Plane p0, p1;
        p0.alloc(ctx, width, height + 4, 1);
        p1.alloc(ctx, width, height + 4, 1);
        if (staged) {
            if (plane == PLANAR_Y) { p0.buf.write(pad0_.data(), 0, p0.bytes()); p1.buf.write(pad1_.data(), 0, p1.bytes()); }
            else if (plane == PLANAR_U) { p0.buf.write(pad0U_.data(), 0, p0.bytes()); p1.buf.write(pad1U_.data(), 0, p1.bytes()); }
            else { p0.buf.write(pad0V_.data(), 0, p0.bytes()); p1.buf.write(pad1V_.data(), 0, p1.bytes()); }
        } else {
            const int vshift = plane == PLANAR_Y
                ? (VPAD - 2)
                : ((VPAD >> srcvi.GetPlaneHeightSubsampling(PLANAR_U)) - 2);
            st8_.resize((size_t)width * (height + 4));
            for (int j = 0; j < height + 4; ++j)
                memcpy(st8_.data() + (size_t)j * width,
                       f0f->GetReadPtr(plane) + (size_t)(j + vshift) * f0f->GetPitch(plane),
                       width);
            p0.buf.write(st8_.data(), 0, p0.bytes());
            for (int j = 0; j < height + 4; ++j)
                memcpy(st8_.data() + (size_t)j * width,
                       f1f->GetReadPtr(plane) + (size_t)(j + vshift) * f1f->GetPitch(plane),
                       width);
            p1.buf.write(st8_.data(), 0, p1.bytes());
        }

        cl_kernel k = prog.kernel("kf_analyze_diff");
        int a = 0;
        oclhost::set_arg(k, a++, result.mem());
        oclhost::set_arg(k, a++, p0.mem());
        oclhost::set_arg(k, a++, p1.mem());
        oclhost::set_arg(k, a++, width);
        oclhost::set_arg(k, a++, height + 2);  // guard-defeat (DEVIATION 5)
        oclhost::set_arg(k, a++, width);
        oclhost::dispatch2DLocalOffset(ctx, k, 0, 2, width, height, 32, 16);
    }

    std::vector<uint8_t> st8_;
    std::vector<uint8_t> pad0_, pad1_, pad0U_, pad1U_, pad0V_, pad1V_;
};

// ===========================================================================
// registration
// ===========================================================================

AVSValue __cdecl CreateKFMSuperOCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        PClip clip = args[0].AsClip();
        PClip pad = args[1].AsClip();
        PClip super = new KFMSuperOCL(clip, pad, env);
        AVSValue separate = env->Invoke("SeparateFields", AVSValue(super));
        // upstream: env->Invoke("Align", ...) — skipped when the core does
        // not provide it (DEVIATION 10)
        if (env->FunctionExists("Align"))
            return env->Invoke("Align", separate);
        return separate;
    } catch (const std::exception& e) {
        env->ThrowError("[KFMSuperOCL] %s", e.what());
    }
    return AVSValue();
}

AVSValue __cdecl CreateKCleanSuperOCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        return new KCleanSuperOCL(
            args[0].AsClip(),
            args[1].AsInt(10),   // thY (upstream default 10)
            args[2].AsInt(8),    // thC (upstream default 8)
            env);
    } catch (const std::exception& e) {
        env->ThrowError("[KCleanSuperOCL] %s", e.what());
    }
    return AVSValue();
}

AVSValue __cdecl CreateKPreCycleAnalyzeOCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        return new KPreCycleAnalyzeOCL(
            args[0].AsClip(),
            args[1].AsInt(20),   // threshMY (upstream default 20)
            args[2].AsInt(12),   // threshSY (upstream default 12)
            args[3].AsInt(24),   // threshMC (upstream default 24)
            args[4].AsInt(16),   // threshSC (upstream default 16)
            env);
    } catch (const std::exception& e) {
        env->ThrowError("[KPreCycleAnalyzeOCL] %s", e.what());
    }
    return AVSValue();
}

AVSValue __cdecl CreateKPreCycleAnalyzeShowOCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        return new KPreCycleAnalyzeShowOCL(
            args[0].AsClip(),    // fmclip
            args[1].AsClip(),    // source
            env);
    } catch (const std::exception& e) {
        env->ThrowError("[KPreCycleAnalyzeShowOCL] %s", e.what());
    }
    return AVSValue();
}

AVSValue __cdecl CreateKContainsCombeOCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        return new KContainsCombeOCL(args[0].AsClip(), env);
    } catch (const std::exception& e) {
        env->ThrowError("[KContainsCombeOCL] %s", e.what());
    }
    return AVSValue();
}

AVSValue __cdecl CreateKCombeMaskOCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        return new KCombeMaskOCL(
            args[0].AsClip(),    // source
            args[1].AsClip(),    // flag
            env);
    } catch (const std::exception& e) {
        env->ThrowError("[KCombeMaskOCL] %s", e.what());
    }
    return AVSValue();
}

AVSValue __cdecl CreateKRemoveCombeOCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        return new KRemoveCombeOCL(
            args[0].AsClip(),              // pad (source with the VPAD halo)
            args[1].AsClip(),              // super
            (float)args[2].AsFloat(6),     // thY (upstream default 6)
            (float)args[3].AsFloat(6),     // thC (upstream default 6)
            env);
    } catch (const std::exception& e) {
        env->ThrowError("[KRemoveCombeOCL] %s", e.what());
    }
    return AVSValue();
}

// KCFieldDiff — upstream KFieldDiff::CFunc (verbatim semantics)
AVSValue __cdecl CreateKCFieldDiffOCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        AVSValue cnt = env->GetVar("current_frame");
        if (!cnt.IsInt()) {
            env->ThrowError("[KCFieldDiff] This filter can only be used within ConditionalFilter!");
        }
        int n = cnt.AsInt();
        std::unique_ptr<KFieldDiffOCL> f(new KFieldDiffOCL(
            args[0].AsClip(),
            (float)args[1].AsFloat(3),   // nt (upstream default 3)
            args[2].AsBool(true)         // chroma (upstream default true)
        ));
        return f->internalFieldDiff(args[0].AsClip(), n, env);
    } catch (const std::exception& e) {
        env->ThrowError("[KCFieldDiffOCL] %s", e.what());
    }
    return AVSValue();
}

// KCFrameDiffDup — upstream KFrameDiffDup::CFunc (verbatim semantics)
AVSValue __cdecl CreateKCFrameDiffDupOCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        AVSValue cnt = env->GetVar("current_frame");
        if (!cnt.IsInt()) {
            env->ThrowError("[KFrameDiffDup] This filter can only be used within ConditionalFilter!");
        }
        int n = cnt.AsInt();
        std::unique_ptr<KFrameDiffDupOCL> f(new KFrameDiffDupOCL(
            args[0].AsClip(),
            args[1].AsBool(true),  // chroma (upstream default true)
            args[2].AsInt(32)      // blksize (upstream default 32)
        ));
        return f->internalFrameDiff(args[0].AsClip(), n, env);
    } catch (const std::exception& e) {
        env->ThrowError("[KCFrameDiffDupOCL] %s", e.what());
    }
    return AVSValue();
}

AVSValue __cdecl CreateKAnalyzeNoiseOCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        return new KAnalyzeNoiseOCL(
            args[0].AsClip(),    // src
            args[1].AsClip(),    // noise
            args[2].Defined() ? args[2].AsClip() : PClip(),  // s4uper (optional)
            env);
    } catch (const std::exception& e) {
        env->ThrowError("[KAnalyzeNoiseOCL] %s", e.what());
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
        // CombingAnalyze.cu AddFuncCombingAnalyze (verbatim signatures)
        env->AddFunction("KFMSuperOCL", "cc", CreateKFMSuperOCL, 0);
        env->AddFunction("KFMSuper", "cc", CreateKFMSuperOCL, 0);
        env->AddFunction("KCleanSuperOCL", "c[thY]i[thC]i", CreateKCleanSuperOCL, 0);
        env->AddFunction("KCleanSuper", "c[thY]i[thC]i", CreateKCleanSuperOCL, 0);
        env->AddFunction("KPreCycleAnalyzeOCL",
                         "c[threshMY]i[threshSY]i[threshMC]i[threshSC]i",
                         CreateKPreCycleAnalyzeOCL, 0);
        env->AddFunction("KPreCycleAnalyze",
                         "c[threshMY]i[threshSY]i[threshMC]i[threshSC]i",
                         CreateKPreCycleAnalyzeOCL, 0);
        env->AddFunction("KPreCycleAnalyzeShowOCL", "cc", CreateKPreCycleAnalyzeShowOCL, 0);
        env->AddFunction("KPreCycleAnalyzeShow", "cc", CreateKPreCycleAnalyzeShowOCL, 0);
        env->AddFunction("KContainsCombeOCL", "c", CreateKContainsCombeOCL, 0);
        env->AddFunction("KContainsCombe", "c", CreateKContainsCombeOCL, 0);
        env->AddFunction("KCombeMaskOCL", "cc", CreateKCombeMaskOCL, 0);
        env->AddFunction("KCombeMask", "cc", CreateKCombeMaskOCL, 0);
        env->AddFunction("KRemoveCombeOCL", "cc[thY]f[thC]f", CreateKRemoveCombeOCL, 0);
        env->AddFunction("KRemoveCombe", "cc[thY]f[thC]f", CreateKRemoveCombeOCL, 0);

        // DecombeUCF.cu AddFuncDecombeUCF (verbatim signatures; the two
        // conditional functions read current_frame like upstream)
        env->AddFunction("KCFieldDiffOCL", "c[nt]f[chroma]b", CreateKCFieldDiffOCL, 0);
        env->AddFunction("KCFieldDiff", "c[nt]f[chroma]b", CreateKCFieldDiffOCL, 0);
        env->AddFunction("KCFrameDiffDupOCL", "c[chroma]b[blksize]i", CreateKCFrameDiffDupOCL, 0);
        env->AddFunction("KCFrameDiffDup", "c[chroma]b[blksize]i", CreateKCFrameDiffDupOCL, 0);
        env->AddFunction("KAnalyzeNoiseOCL", "cc[s4uper]c", CreateKAnalyzeNoiseOCL, 0);
        env->AddFunction("KAnalyzeNoise", "cc[s4uper]c", CreateKAnalyzeNoiseOCL, 0);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "KFM CombingAnalyze + DecombeUCF analysis filters (OpenCL port, MIT; seventh host plugin of the OpenCL port)";
}

