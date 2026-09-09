#!/usr/bin/env python3
"""Validate the four KFM KFMFilterBase.cu coefficient kernels
(kf_calc_combe, kf_merge_uvcoefs, kf_extend_coef2, kf_apply_uvcoefs_420) in
src/opencl/kfm/kernels/kfm_filterbase.cl against the CPU mirror
sim/kfm_filterbase_ref.cpp with an independent Python golden.

All four are integer-exact.  kf_calc_combe is verified over its interior rows
(y in [2,height-3]); its border rows read a host-padded plane (VPAD) and are
RIG-VERIFY (mirror/golden put a -1 sentinel there).  kf_extend_coef2 is the CUDA
kl_extend_coef2 device kernel (upstream's CPU fallback differs at rows 0 and
height-1; the OpenCL target is the device kernel).

Run:  python3 python/run_kfm_filterbase.py
"""
import os, random, struct, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "kfm_filterbase_ref")


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "kffb_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "kffb_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def calc_combe_val(a, b, c, d, e):
    v = a + c * 4 + e - (b + d) * 3
    return -v if v < 0 else v


def golden_calc_combe(width, height, pitch, src):
    out = []
    for yy in range(height):
        for xx in range(width):
            if 2 <= yy <= height - 3:
                off = xx + yy * pitch
                c = calc_combe_val(src[off - 2 * pitch], src[off - pitch],
                                   src[off], src[off + pitch],
                                   src[off + 2 * pitch]) >> 2
                out.append(max(0, min(255, c)))
            else:
                out.append(-1)  # border sentinel
    return out


def golden_merge(width, height, pitchY, pitchUV, lx, ly, fY, fU, fV):
    # returns logical width*height of the folded Y plane
    fY = list(fY)
    for yy in range(height):
        for xx in range(width):
            oY = xx + yy * pitchY
            oUV = (xx >> lx) + (yy >> ly) * pitchUV
            u, v = fU[oUV], fV[oUV]
            fY[oY] = max(fY[oY], max(u, v))
    return [fY[xx + yy * pitchY] for yy in range(height) for xx in range(width)]


def golden_extend(width, height, pitch, src):
    out = []
    for yy in range(height):
        for xx in range(width):
            y0 = max(yy - 1, 0)
            y2 = min(yy + 1, height - 1)
            m = max(src[xx + y0 * pitch], src[xx + yy * pitch],
                    src[xx + y2 * pitch])
            out.append(m)
    return out


def golden_apply420(widthUV, heightUV, pitchY, pitchUV, fY):
    uout, vout = [], []
    for yy in range(heightUV):
        for xx in range(widthUV):
            oY0 = (2 * xx) + (2 * yy) * pitchY
            v = (fY[oY0] + fY[oY0 + 1] + fY[oY0 + pitchY] +
                 fY[oY0 + pitchY + 1])
            avg = (v + 2) >> 2
            uout.append(avg)
            vout.append(avg)
    return uout + vout


def main():
    subprocess.run(["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "kfm_filterbase_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(19)
    ok = True
    total = 0

    # K: calc_combe (interior only)
    for _ in range(180):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 20) * 4
        height = rng.randint(7, 18)          # >= 5 so interior exists
        pitch = width + rng.choice([0, 2])
        n = pitch * height
        # interlaced-ish content to exercise combing (row pitch layout)
        src = [0] * n
        for yy in range(height):
            for xx in range(width):
                field = (yy % 2)
                base = rng.randint(0, maxv)
                v = base + (rng.randint(0, maxv // 6) if field else 0)
                src[yy * pitch + xx] = max(0, min(maxv, v))
        hdr = [ord('K'), width, height, pitch, n]
        got = run_mirror(hdr + src)
        exp = golden_calc_combe(width, height, pitch, src)
        total += 1
        # compare interior rows only
        interior_ok = True
        for i, (g, e) in enumerate(zip(got, exp)):
            if e != -1 and g != e:
                interior_ok = False
                if ok:
                    print("calc_combe MISMATCH", width, height, "px", i, g, e)
        if not interior_ok:
            ok = False
            if total >= 3: break

    # M: merge_uvcoefs (YV12 lx=ly=1)
    for _ in range(160):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 20) * 4
        height = rng.randint(1, 6) * 2        # even (YV12)
        lx, ly = 1, 1
        pitchY = width + rng.choice([0, 2])
        uvwidth = width >> lx
        uvheight = height >> ly
        pitchUV = uvwidth + rng.choice([0, 2])
        nY = pitchY * height
        nU = pitchUV * uvheight
        fY = [rng.randint(0, 128) for _ in range(nY)]
        fU = [rng.randint(0, 128) for _ in range(nU)]
        fV = [rng.randint(0, 128) for _ in range(nU)]
        hdr = [ord('M'), width, height, pitchY, pitchUV, lx, ly, nY, nU]
        got = run_mirror(hdr + fY + fU + fV)
        exp = golden_merge(width, height, pitchY, pitchUV, lx, ly, fY, fU, fV)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("merge_uvcoefs MISMATCH", width, height, "px", i, g, e)
                    break
            if total >= 3: break

    # E: extend_coef2
    for _ in range(180):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 20) * 4
        height = rng.randint(1, 14)
        pitch = width + rng.choice([0, 2])
        n = pitch * height
        src = [rng.randint(0, 128) for _ in range(n)]
        hdr = [ord('E'), width, height, pitch, n]
        got = run_mirror(hdr + src)
        exp = golden_extend(width, height, pitch, src)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("extend_coef2 MISMATCH", width, height, "px", i, g, e)
                    break
            if total >= 3: break

    # A: apply_uvcoefs_420
    for _ in range(160):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        uvwidth = rng.randint(1, 12) * 4
        uvheight = rng.randint(1, 10)
        width = uvwidth * 2
        pitchY = width + rng.choice([0, 2])
        pitchUV = uvwidth + rng.choice([0, 2])
        nY = pitchY * (uvheight * 2)
        nUV = pitchUV * uvheight
        fY = [rng.randint(0, 128) for _ in range(nY)]
        fU = [rng.randint(0, 128) for _ in range(nUV)]
        fV = [rng.randint(0, 128) for _ in range(nUV)]
        hdr = [ord('A'), uvwidth, uvheight, pitchY, pitchUV, nY, nUV]
        got = run_mirror(hdr + fY + fU + fV)
        exp = golden_apply420(uvwidth, uvheight, pitchY, pitchUV, fY)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("apply_uvcoefs_420 MISMATCH", uvwidth, uvheight,
                          "px", i, g, e)
                    break
            if total >= 3: break

    print(f"KFM FilterBase (calc_combe/merge_uvcoefs/extend_coef2/"
          f"apply_uvcoefs_420): {'PASS' if ok else 'FAIL'} ({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
