# masktools port spec (CUDA surface → OpenCL)

Upstream: `rigaya/AviSynthCUDAFilters` @ `68aef6e`, submodule
`rigaya/masktools` @ `24ba826` (GPL). Line numbers are 1-based at that
commit.

## 1. Census — the CUDA surface is 5 kernels in 2 files (complete)

masktools is a large CPU filter suite; only these carry `__global__` code.

| # | CUDA kernel | File:line | OpenCL port | Status |
|---|---|---|---|---|
| 1 | `kl_fill` | `common/functions/functions_cuda.cu:16` | `km_fill`, `km_fill_f32` | ALG-VERIFIED |
| 2 | `kl_copy` | `common/functions/functions_cuda.cu:62` | `km_copy_bytes` | ALG-VERIFIED |
| 3 | `kl_lut_x` | `masktools/filters/lut/lut_kernel.cu:15` | `km_lut_x` | ALG-VERIFIED |
| 4 | `kl_lut_xy` | `masktools/filters/lut/lut_kernel.cu:48` | `km_lut_xy` | ALG-VERIFIED |
| 5 | `kl_lut_xyz` | `masktools/filters/lut/lut_kernel.cu:88` | `km_lut_xyz` | ALG-VERIFIED |

Launch wrappers: `Functions::memset_plane{,_16,_32}_cuda` (:26/:37/:48),
`Functions::copy_plane_cuda` (:72), `lut_cuda` (:120) and its depth
dispatcher `lut_cuda_16` (:157).

All five are elementwise with 2D bounds guards — no `__local`, no
reductions, no work-group-size constraint, no floating-point arithmetic
(the f32 fill only moves bit patterns). That makes this the cheapest
family in the suite to port and the easiest to prove.

## 2. Conventions

- Widths/pitches are in 4-pixel VECTORS (`width4`, `pitch4`), as upstream's
  uchar4/ushort4/float4 kernels use them; the port indexes 4 scalar lanes.
- **Tail truncation is inherited**: the launchers compute `w4 = width >> 2`
  (`rowsize >> 2` for the copy), so widths that are not multiples of 4
  leave the tail pixels untouched. That is upstream behaviour, not an
  artifact of the port.
- `km_copy_bytes` is byte-granular (upstream always instantiates uchar4
  from a BYTE rowsize), so it is format-independent.
- `km_fill_f32` takes the fill value as a bit pattern, so no host-side
  float argument conversion can perturb it.
- The mask branch is `sizeof(PX) == 1`, exactly mirroring upstream's
  `sizeof(pixel_t) == 1` test (compile-time folded in both).

## 3. Upstream defect found and transcribed (do not "fix" blindly)

`lut_cuda_16` (lut_kernel.cu:156-165) instantiates the kernels with
`bits_per_pixel = 8` for **all** of 10/12/14/16-bit, so the 16-bit kernels
run with shift 8 and `mask = 255`. But the host builds the table with the
REAL depth — `idx = (x << bits_per_pixel) + y` over `1 << bits` entries
(`lut_data.cpp:27`, loop at :25-30) — and the CPU path indexes it that way
(`lutxy.cpp:30`). On the CUDA path this means:

- `lut_x` 16-bit reads only `lut[X & 255]`, the first 256 entries.
- `lut_xy` 16-bit: `((X << 8) + Y) & 255 == Y & 255` — **X is dropped
  entirely**; the output depends only on the second clip.
- `lut_xyz` 16-bit collapses to `Z & 255`; the 16-bit 3-input table is not
  even built (`lut_data.cpp:32`, `case 3` is commented out).

The port transcribes this faithfully and takes `lut_bits`/`mask` as host
arguments (a host mirroring upstream passes 8/255 at every 16-bit depth).
The runner **pins the collapse with dedicated assertions** so it cannot
change silently: swapping the leading clip must not change the 16-bit
`lut_xy` output, and the result must equal `lut[Z & 255]` for `lut_xyz`.
Whether to correct the depth on a rig is a separate, evidence-backed
decision — flagged, not taken.

The masking is also load-bearing for memory safety: a mutant that removed
it segfaulted the mirror, because the 16-bit index would otherwise run off
the 256-entry region the CUDA path actually addresses.

## 4. Verification

`src/opencl/masktools/kernels/masktools_lut.cl` +
`sim/masktools_lut_ref.cpp` + `python/run_masktools_lut.py` (610 cases:
fill 120, f32 fill 60, copy 100, lut_x 100, lut_xy 100, lut_xyz 100, plus
30 collapse pins; both PX widths, slack pitches, saturated edge values).

- The golden derives the LUT index in **mixed radix** (`x*B^2 + y*B + z`,
  `B = 2**lut_bits`) with masking as a modulo, rather than repeating the
  port's shift/AND spelling.
- Index spaces above 2^16 (8-bit `lut_xyz` reaches 2^24) use a
  **procedural table** shared by mirror and golden instead of shipping 16M
  entries, so the full index range is exercised.
- Mutation-tested, all caught: wrong `xy` shift, wrong `xyz` shift,
  masking applied on the 8-bit path, masking removed (segfault), fill
  writing 3 of 4 lanes, copy reusing the dst pitch for src, and swapped
  `xy` operands (caught by the collapse pins).

## 5. Status

The masktools CUDA surface is fully ported (5/5). Nothing rig-side is
pending beyond the usual real-compiler build and the open question in §3.
