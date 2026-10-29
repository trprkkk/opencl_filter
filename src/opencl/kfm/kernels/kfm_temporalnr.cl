/* ============================================================================
 * kfm_temporalnr.cl — OpenCL port of the KFM `KTemporalNR` filter kernel.
 *
 * Faithful port of rigaya/AviSynthCUDAFilters -> KFM/KDeband.cu (KFM is MIT).
 * KTemporalNR is a temporal noise reducer: each output pixel is an average of
 * the pixels in the same spatial location across nframes frames (dist either
 * side of the centre frame mid), keeping only the frames whose pixel is within
 * `thresh` of the centre pixel.  It ships an exact CPU twin `cpu_temporal_nr`
 * (== the CUDA `kl_temporal_nr` arithmetic), which is the reference this kernel
 * is cross-checked against.
 *
 * Per-pixel algorithm (identical for every output element; purely column-wise,
 * so the CUDA 1-wide/4-wide vectorisation makes no difference — a scalar port
 * is bit-identical for any width):
 *   center = frames[mid][x,y]
 *   count = 0, sum = 0
 *   for i in [0, nframes):  ref = frames[i][x,y]
 *       diff = abs(ref - center)
 *       if diff <= thresh:  count += 1;  sum += ref
 *   avg = (float)sum / count + 0.5f     // float32, count>=1 (i==mid gives 0)
 *   dst = (int)avg
 *
 * NOTE on division: the CPU reference (and this OpenCL transliteration) uses a
 * normal float32 division `(float)sum/count`.  The real CUDA *device* kernel
 * instead calls the approximate intrinsic `__fdividef` (max ~2 ulp error),
 * which can differ from correctly-rounded division at a rounding boundary.
 * That is a CUDA-specific hardware-division nuance with no OpenCL equivalent;
 * OpenCL `/` mirrors the CPU reference, so ALG-VERIFIED is against that.  A rig
 * comparing against the literal CUDA __global__ output would need
 * `-cl-fp32-correctly-rounded-enabled` (best-effort) and is a // RIG-VERIFY
 * item — the difference is at most a couple of ulp, not an algorithm change.
 *
 * DEVICE RECORD (2026-10-28, rig/dev_kfm1.py + rig/probe_fdiv2.py, NVIDIA
 * RTX 2060 SUPER): the "OpenCL `/` mirrors the CPU reference" claim holds
 * on this driver ONLY with `-cl-fp32-correctly-rounded-divide-sqrt` — the
 * default build mis-rounds 360/800 probe values (one-ulp), so the f32
 * division kernels (edgelevel + temporalnr) MUST be built with that flag;
 * with it the device is bit-exact vs the golden (temporalnr PASS).  A CUDA
 * comparison against the literal `__fdividef` output remains open.
 *
 * NOTE on types: pixel_t is uchar/ushort.  count/sum fit in int
 * (count <= nframes = 2*dist+1 <= 33;  sum <= 33*65535), and
 * (float)sum/count lands back in [0,maxv] so no explicit clamp is needed
 * (matches the CPU twin, which does not clamp either).
 *
 * Status:
 *   // ALG-VERIFIED (python/run_kfm_temporalnr.py, 300 cases) vs the CPU
 *   // mirror sim/kfm_temporalnr_ref.cpp, float32-exact Python golden.
 *
 * Layout seam: the CUDA host launches this kernel once PER PLANE, passing a
 * TemporalNRPtrs struct holding nframes separate input-plane pointers.  OpenCL
 * cannot take a runtime-count array of image args, so this kernel instead takes
 * the nframes planes packed contiguously in one buffer `frames`, plane i at
 * ELEMENT offset i*frame_stride (frame_stride = host-chosen stride between
 * planes in PX elements, normally pitch*height of that plane; element units
 * because `frames` is typed __global const PX* — a byte stride would only
 * work for PX=uchar and read plane 2i at 16-bit, pinned empirically
 * 2026-10-29).  Each plane is processed by its own
 * launch with that plane's own width/height/pitch.  Per-pixel arithmetic is
 * identical to cpu_temporal_nr; only the pointer-array vs packed-frame host
 * layout differs (recorded in docs/KFM_PORT_SPEC.md).
 * ==========================================================================*/

#ifndef PX
#error "compile with -DPX=uchar|-DPX=ushort and -DPX_MAX=255|65535"
#endif

/* thresh is passed host-scaled (KTemporalNR::scaleParam:
 * (int)(thresh*(1<<(bits-8))+0.5f)). */
kernel void kf_temporal_nr(
    __global       PX* __restrict dst,
    __global const PX* __restrict frames,
    int width, int height, int pitch,
    int frame_stride,
    int nframes, int mid,
    int thresh)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x < width && y < height) {
        size_t idx = (size_t)x + (size_t)y * (size_t)pitch;
        PX center = frames[(size_t)mid * (size_t)frame_stride + idx];

        int count = 0;
        int sum = 0;
        for (int i = 0; i < nframes; ++i) {
            int ref = (int)frames[(size_t)i * (size_t)frame_stride + idx];
            int diff = ref - (int)center;
            if (diff < 0) diff = -diff;
            if (diff <= thresh) {
                count += 1;
                sum += ref;
            }
        }

        /* (float)sum / count + 0.5f, then truncate.  count>=1 (i==mid). */
        float avg = (float)sum / (float)count + 0.5f;
        dst[idx] = (PX)(int)avg;
    }
}
