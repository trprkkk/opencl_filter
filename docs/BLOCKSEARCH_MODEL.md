# KTGMC block-search host model (grounds the remaining MV kernel ports)

This documents, from `AviSynthCUDAFilters/KTGMC/{MVKernel.cu,MV.cpp}`, the exact
host data structures and execution model the not-yet-ported search / degrain /
compensate block kernels depend on. Everything here is read from source; the
goal is that each later OpenCL kernel port can be transliterated against this
one model instead of re-deriving it.

Status: **source-derived reference**. None of the structures below are wired to
an OpenCL host in this sandbox (no device). They are the contract a rig host
must build before the block kernels can run/diff (see `HOST_CONTRACT.md`).

## 0. The key fact: the shipped kernels run the CPU-ordered scalar path

`MVKernel.cu` launches every `kl_search<...>` instantiation with
`#define CPU_EMU true` (lines ~2625–2657). When `CPU_EMU` is true:

- `dev_reduce_result<LEN,true>` reduces candidate costs by a **serial scan in
  lane 0 in fixed index order** (`for i in 1..LEN MinCost(tmp[0],tmp[i])`),
  not by a warp-shuffle tree — so the surviving candidate is deterministic and
  matches the author's CPU reference.
- FetchPredictors writes predictor `i` to slot `i==3? (i+1) : i` (slot 3 is the
  median), then computes the median predictor in slots 4,5,6 — CPU-order.

So although the CUDA block `kl_search` is warp-shaped, its *results* are the
deterministic output of a serial algorithm, which a faithful scalar OpenCL port
can reproduce exactly. This is the basis for the `RIG-VERIFY` on-rig diff.

## 1. SearchBatch host struct → OpenCL

CUDA (MVKernel.cu:832):
```cpp
template<typename pixel_t> struct SearchBatch {
  VECTOR *out;                 // int3 (x,y,sad) per block, result
  int* dst_sad;                // int per block
  const SearchBlock* blocks;   // 17 ints per block (data[12]+dataf[5])
  short2* vectors;             // int2, per MV row incl. sentinels
  volatile int* prog;          // per column search progress
  int* next;                   // work-stealing counter
  const pixel_t *pSrcY,*pSrcU,*pSrcV;   // source (search) plane block base
  const pixel_t *pRefY,*pRefU,*pRefV;   // reference (super-frame) planes
};
// passed to device as: union { SearchBatch d; int data[LEN]; }
```

OpenCL translation: pass these as plain kernel pointer args (a `union`-as-int-
array trick is unnecessary). Host keeps one `SearchBatch`-equivalent per batch.

`SearchBlock` (already used by `kt_prepare_search`): `data[12]`+`dataf[5]`,
carried here as two flat int arrays, stride 12 / 5 per block.

Per-MV-row layout (matching `kt_init_const_vec` + `kt_prepare_search`):
`row_base = vectors + row*vectorsPitch`, with `[-2]=zero=(0,0)`,
`[-1]=global=globalMV*nPel`, `[0..nBlkX*nBlkY)` = that row's blocks,
then a **copy region** of `nBlkX*nBlkY` more elements holding the previous
(coarse) level's vectors (indices offset `+nBlkX*nBlkY`), which left / up /
bottom-right predictors read so searching a block does not race with its
overwritten neighbour.

## 2. SearchBlock fields (as `kt_prepare_search` fills them)

- `data[0..3] = CLIP_RECT = {DxMax,DyMax,DxMin,DyMin}` block-search bounds.
- `data[4..9] = REF_VECTOR_INDEX[]` MV indices (also predictor slot indices):
  - `data[4] = -2` (zero), `data[5] = -1` (global),
    `data[6] = blkIdx` (own predictor), `data[7]=p1` (left),
    `data[8]=p2` (up), `data[9]=p3` (bottom-right from coarse level copy region).
    Only `data[6]` is guaranteed valid; `data[7..9]` may be -2 when no neighbour.
    Each references the `vectors` array (sentinel or block or copy region).
- `data[10..11] = PRED_X/Y`: the "predictor" seed vector = this block's coarse
  MV, used as the centre for the motion-model cost term.
- `dataf[0..3] = {penaltyZero, penaltyGlobal, 0, penaltyNew}` and
  `dataf[4] = lambda` (0 on block row 0) — cost offsets / MV-length weighting.

Cost model used throughout the search device code:
`cost = candidateSAD + penalties + (lambda * sq_norm(candidate, PRED))>>8`
(`dev_sq_norm`), subject to `dev_check_mv` against `CLIP_RECT` and clipping via
`dev_clip_mv`. `LARGE_COST = INT_MAX` marks invalid.

## 3. Super-frame / reference layout (`dev_get_ref_block`)

The reference buffer is treated as **NPEL×NPEL sub-pel planes** stacked
`nImgPitch` elements apart, each plane being `nPitch`-wide rows of the padded
image. A vector `(vx,vy)` (integer, in sub-pel units) selects:
```
NPEL==1: index = pRef[vx + vy*nPitch]
NPEL==2: sx=vx&1, sy=vy&1, si=sx+sy*2, x=vx>>1, y=vy>>1
         index = pRef[x + y*nPitch + si*nImgPitch]
NPEL==4: sx=vx&3, sy=vy&3, si=sx+sy*4, x=vx>>2, y=vy>>2
         index = pRef[x + y*nPitch + si*nImgPitch]
```
`pRef` here is a pointer already advanced by the block's base offset
(`&plane[offx + offy*nPitch]`). MVKernel.cu declares `short2` vectors but motion
values (x,y) can be wider after scaling; `dev_get_ref_block` takes ints.

Which planes exist and their `nPitch`/`nImgPitch` are fixed by the host's
`KMSuper` frame buffer (see §5 of MV_PORT_SPEC.md). The chroma component is
fetched with `(vx>>1,vy>>1)` and `nPitchUV`/`nImgPitchUV`.

## 4. Dispatched kernel instantiations (what a host must be able to launch)

`kl_calc_all_sad`: threads `BLK_SIZE*8`; blocks `(nBlkX,nBlkY,batch)`.
`kl_search<BLK_SIZE,SEARCH,NPEL,CHROMA,CPU_EMU=true>`: threads `BLK_SIZE*8`;
blocks `(batch, min(nBlkX,nBlkY))`. Combinations present in the launcher table
(MVKernel.cu ~2629–2655): BLK_SIZE ∈ {8,16,32}; SEARCH ∈ {1 exhaustive,
2 hex+expanding}; NPEL ∈ {1,2}; CHROMA ∈ {true,false}. (Exhaustive SEARCH=1 rows
only use NPEL=1; hex SEARCH=2 covers NPEL 1 & 2.)

`BLK_SIZE_UV = BLK_SIZE/2`. Block grid step `BLK_STEP = BLK_SIZE/2`, base
`offx = nPad + bx*BLK_STEP`, `offy = nPad + by*BLK_STEP`. `nPad` and the
extended-frame size come from the level pyramid.

## 5. Pure device helpers to transliterate first

These are used by every block kernel and are pure integer functions:

```c
void  k_dev_clip_mv(int2* v, const int rect[4]);      // clamp v to [rect2..0],[rect3..1]
bool  k_dev_check_mv(int x,int y, const int rect[4]); // x in [rect2,rect0], y in [rect3,rect1]
int   k_dev_sq_norm(ax,ay,bx,by);                     // (ax-bx)^2+(ay-by)^2
// ref-element offset for NPEL sub-pel plane stack (see §3)
int   k_dev_ref_offset(int vx,int vy,int nPitch,int nImgPitch,int NPEL);
```

`dev_clip_mv`: `v.x = v.x>rect[0]?rect[0] : v.x<rect[2]?rect[2] : v.x`
(symmetric for y with rect[1]/rect[3]).
`dev_check_mv`: `(x<=rect[0]) && (y<=rect[1]) && (x>=rect[2]) && (y>=rect[3])`.
`dev_sq_norm`: `(ax-bx)*(ax-bx) + (ay-by)*(ay-by)`.

## 6. Block SAD (`dev_calc_sad` / `kl_calc_all_sad`)

Per block, `sad = Σ |src - ref|` over the BLK_SIZE×BLK_SIZE luma window, plus
chroma windows when `CHROMA` (U,V each BLK_SIZE_UV×BLK_SIZE_UV). src is a plain
padded plane (indexed by BLK_SIZE row stride for src tile, `nPitchY` for ref);
ref base = `dev_get_ref_block(pRef@blockoff, nPitchY, nImgPitchY, vx, vy)` for
the block's MV. Absolute-difference sums are order-independent integers, so an
OpenCL scalar double loop reproduces `__sad`/`__vabsdiff4` exactly (the packed/
funnel-shift loads are pure load optimizations). CUDA reduces the per-thread
partials over `BLK_SIZE*8` threads; OpenCL can serialise or use a `__local`
reduction — identical integer result.

`kl_calc_all_sad` then writes, per block, `dst_sad[blk] = sad` and
`out[blk] = {xy.x, xy.y, sad}` where `xy` is the block's current MV read from
`vectors[bx+by*nBlkX]`.

## 7. Search driver `kl_search` shape (deferred port)

Per batch block, iterates `blkx` via a shared `next` counter and every `blky`,
loading the BLK_SIZE tile to shared (`dev_read_pixels` for hex), building the 8
candidate predictors (zero/global/own/left/up/median), running
`dev_expanding_search_1/2` + `dev_hex2_search_1` or the exhaustive path, keeping
the lowest-cost `CostResult` (`dev_reduce_result<…,CPU_EMU=true>` serial scan),
then writes the winning `short2` back to `vectors[blky*nBlkX+blkx]`, fences, and
sets `prog[blkx]=blky` for the ANALYZE_SYNC=1 dependency wait. Porting this is
the remaining large work and is strictly `RIG-VERIFY` (device-dependent for
validation, but deterministic under CPU_EMU=true).

## 8. What is captured vs. what still blocks a faithful `kl_search`

Captured faithfully (transliteratable): the pure helpers (§5), the per-block
cost/SAD/expanding-refine arithmetic, the predictor-setup ordering, and the
`CPU_EMU=true` deterministic reduce semantics.

### 8a. RESOLVED for `kl_calc_all_sad` (read out of MVKernel.cu)

Items 1 and 2 below were open because §2 guessed at the host assembly. They
are now settled *for this kernel* by reading the source, and
`kt_calc_all_sad` is ALG-VERIFIED accordingly
(`python/run_mv_calc_all_sad.py`):

- **`vectors` is plain row-major `short2`**, indexed `[bx + by*nBlkX]`
  (`SearchBatch`, MVKernel.cu:819; the kernel's own read at :1122). No
  sentinels, no `vectorsPitch`, no appended copy region — that machinery
  exists only for `kl_search`'s predictor slots, which dereference
  `REF_VECTOR_INDEX` (:299). Two ABI bugs in the port were found and fixed
  by this: it had declared `vectors` as `int2*` (8 bytes/entry instead of
  4) and `out` as OpenCL `int3*` (16-byte stride) where upstream's
  `VECTOR {int x,y,sad}` (common/KMV.h:6) is packed 12.
- **Reference origin**: `&pRef[offx + offy*nPitch]` passed through
  `dev_get_ref_block`, which the port's `kt_ref_block_offset` reproduces
  exactly; chroma uses base `(offx>>1, offy>>1)` with the MV halved
  (`vx>>1, vy>>1`), and `offx/offy = nPad + blk*(BLK_SIZE/2)`.
- **Reachable grid**: upstream instantiates BLK_SIZE {8,16,32} x NPEL
  {1,2} x CHROMA (:2631). So `BLK_SIZE == 4` — whose luma loop bound
  `BLK_SIZE/8` would be zero, silently producing a luma SAD of 0 — is
  unreachable, as is `NPEL == 4` for this kernel. The port stays generic;
  the runner covers the reachable grid plus NPEL 4 for the shared helper.
- **Thread split is not observable**: the CUDA kernel spreads the window
  over `BLK_SIZE*8` threads (`x = tid % BLK_SIZE`, `yy = y, y+8, ...`) and
  chroma over three `BLK_SIZE_UV` cases, then `dev_reduce`s. All of it is
  integer absolute-difference addition, so the scalar double loop is exact.

### 8b. STILL blocking `kl_search`

1. **Predictor slot layout.** The sentinel (`[-2]/[-1]`) and appended copy
   region (`+nBlkX*nBlkY`) convention, and `vectorsPitch`, still have to be
   pinned for the `REF_VECTOR_INDEX[0..5]` (= `data[4..9]`) reads that
   resolve `median(left, up, bottom-right)` and the own/left/up MVs. §8a
   settles only the *direct* `[bx + by*nBlkX]` access.
2. **Batch / work-stealing mapping to OpenCL.** CUDA launches one block per
   batch column (`blocks(batch, min(nBlkX,nBlkY))`) that work-steals columns
   via a shared `next` counter and spin-waits on `prog[]` for the
   `ANALYZE_SYNC=1` left-column dependency. Reproducing this needs an
   OpenCL host that either serialises columns in dependency order or
   emulates the spin/atomic handshake — a host-side decision, not a kernel
   transliteration.

Both remaining items are host/layout questions that a device bring-up
(`docs/HOST_CONTRACT.md`) will answer empirically.
