#!/usr/bin/env python3
"""Validate the seven KFM KFMFilterBase.cu kernels
(kf_calc_combe, kf_merge_uvcoefs, kf_extend_coef2, kf_apply_uvcoefs_420,
kf_padv, kf_padh, kf_merge_block) in src/opencl/kfm/kernels/kfm_filterbase.cl
against the CPU mirror sim/kfm_filterbase_ref.cpp with an independent Python
golden.

All seven are integer-exact.  kf_calc_combe is verified over its interior rows
(y in [2,height-3]); its border rows read a host-padded plane (VPAD) and are
RIG-VERIFY (mirror/golden put a -1 sentinel there).  kf_extend_coef2 is the CUDA
kl_extend_coef2 device kernel (upstream's CPU fallback differs at rows 0 and
height-1; the OpenCL target is the device kernel).  kf_padv/kf_padh are the
in-place mirror pads (verified solo plus the composed padv-then-padh 2D pad in
upstream Deblock order).  kf_merge_block is the MergeBlock masked blender
(flag is uchar at both bit depths; full 0..255 flag sweep pins the
negative-invcombe wrap path too).

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


def golden_padv(buf, width, height, pitch, vpad, org):
    # out-of-place w.r.t. the pristine input: reads never observe writes,
    # which holds iff the pad is race-free (reads interior, writes pad rows)
    b = list(buf)
    for yy in range(vpad):
        top_dst = org + (-yy - 1) * pitch
        top_src = org + yy * pitch
        bot_dst = org + (height + yy) * pitch
        bot_src = org + (height - yy - 1) * pitch
        for xx in range(width):
            b[top_dst + xx] = buf[top_src + xx]
            b[bot_dst + xx] = buf[bot_src + xx]
    return b


def golden_padh(buf, width, height, pitch, hpad, org):
    b = list(buf)
    for yy in range(height):
        row = org + yy * pitch
        for xx in range(hpad):
            b[row + (-xx - 1)] = buf[row + xx]
            b[row + (width + xx)] = buf[row + (width - xx - 1)]
    return b


def golden_merge_block(width, height, pitch, fpitch, bits, s24, s60, flag):
    mask = 0xFF if bits == 8 else 0xFFFF
    out = []
    for yy in range(height):
        for xx in range(width):
            combe = flag[xx + yy * fpitch]
            t = (combe * s60[xx + yy * pitch] +
                 (128 - combe) * s24[xx + yy * pitch] + 64) >> 7
            out.append(t & mask)  # the (PX) cast wrap
    return out


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

    # V: padv (in-place vertical mirror pad)
    for _ in range(150):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 40)
        height = rng.randint(2, 24)
        vpad = min(rng.choice([1, 2, 4, 8]), height)
        pitch = width + rng.choice([0, 1, 3])
        n = pitch * (height + 2 * vpad)
        buf = [rng.randint(0, maxv) for _ in range(n)]
        org = vpad * pitch
        hdr = [ord('V'), width, height, pitch, vpad, n]
        got = run_mirror(hdr + buf)
        exp = golden_padv(buf, width, height, pitch, vpad, org)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("padv MISMATCH", width, height, pitch, vpad,
                          "px", i, g, e)
                    break
            if total >= 3: break

    # H: padh (in-place horizontal mirror pad)
    for _ in range(150):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(2, 40)
        height = rng.randint(1, 24)
        hpad = min(rng.choice([1, 2, 4, 8]), width)
        pitch = width + 2 * hpad + rng.choice([0, 1, 3])
        n = pitch * height
        buf = [rng.randint(0, maxv) for _ in range(n)]
        org = hpad
        hdr = [ord('H'), width, height, pitch, hpad, n]
        got = run_mirror(hdr + buf)
        exp = golden_padh(buf, width, height, pitch, hpad, org)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("padh MISMATCH", width, height, pitch, hpad,
                          "px", i, g, e)
                    break
            if total >= 3: break

    # B: padv then padh (composed 2D pad, upstream Deblock order)
    for _ in range(150):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(2, 32)
        height = rng.randint(2, 20)
        vpad = min(rng.choice([1, 2, 4, 8]), height)
        hpad = min(rng.choice([1, 2, 4, 8]), width)
        pitch = width + 2 * hpad + rng.choice([0, 1, 3])
        n = pitch * (height + 2 * vpad)
        buf = [rng.randint(0, maxv) for _ in range(n)]
        org = hpad + vpad * pitch
        hdr = [ord('B'), width, height, pitch, vpad, hpad, n]
        got = run_mirror(hdr + buf)
        step1 = golden_padv(buf, width, height, pitch, vpad, org)
        exp = golden_padh(step1, width, height + 2 * vpad, pitch, hpad,
                          org - vpad * pitch)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("pad-both MISMATCH", width, height, pitch,
                          vpad, hpad, "px", i, g, e)
                    break
            if total >= 3: break

    # G: merge_block (MergeBlock masked blender; flag uchar at both depths)
    for _ in range(200):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 16) * 4      # mult of 4: exact CUDA coverage
        height = rng.randint(1, 24)
        pitch = width + rng.choice([0, 2])
        fpitch = width + rng.choice([0, 2])
        nP = pitch * height
        nF = fpitch * height
        s24 = [rng.randint(0, maxv) for _ in range(nP)]
        s60 = [rng.randint(0, maxv) for _ in range(nP)]
        # production domain [0,128] + full uchar sweep (wrap path)
        if rng.random() < 0.5:
            flag = [rng.randint(0, 128) for _ in range(nF)]
        else:
            flag = [rng.randint(0, 255) for _ in range(nF)]
        hdr = [ord('G'), width, height, pitch, fpitch, bits, nP, nF]
        got = run_mirror(hdr + s24 + s60 + flag)
        exp = golden_merge_block(width, height, pitch, fpitch, bits,
                                 s24, s60, flag)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("merge_block MISMATCH", width, height, bits,
                          "px", i, g, e)
                    break
            if total >= 3: break

    print(f"KFM FilterBase (calc_combe/merge_uvcoefs/extend_coef2/"
          f"apply_uvcoefs_420/padv/padh/merge_block): {'PASS' if ok else 'FAIL'} "
          f"({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
