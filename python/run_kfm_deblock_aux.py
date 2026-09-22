"""Independent golden vs sim/kfm_deblock_aux_ref.cpp (graduated KDeblock helpers).

Modes: S scale_qp, C sharpen_coeff, H max_h, V max_v, B max_vh.  The B golden
is the separable composition max_h o max_v (box-max separability), which
cross-checks the direct box form in the mirror per the handoff recipe.
"""
import os
import random
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "kfm_deblock_aux_ref")

SHARPEN_COEFF = [
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 10,
    50, 90, 120, 150, 160,
    170, 180, 190, 200, 210,
    220, 230, 240, 245, 250,
    255, 255, 255, 255, 255,
]


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "kdx_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "kdx_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)) + "\n")
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def norm_qscale(q, t):
    if t == 0:
        return q << 2
    if t == 1:
        return q << 1
    if t == 2:
        return q
    if t == 3:
        return 63 - q + 2
    return q


def golden_scale_qp(width, height, src_pitch, stype, src):
    return [norm_qscale(src[x + y * src_pitch], stype) % 256
            for y in range(height) for x in range(width)]


def golden_sharpen_coeff(width, height, qp_pitch, qp):
    out = []
    for y in range(height):
        for x in range(width):
            q = qp[x + y * qp_pitch] >> 3
            out.append(255 if q >= 25 else SHARPEN_COEFF[q])
    return out


def golden_max_h(width, height, pitch, radius, src, org):
    return [max([0] + [src[org + (x + d) + y * pitch]
                       for d in range(-radius, radius + 1)])
            for y in range(height) for x in range(width)]


def golden_max_v(width, height, pitch, radius, src, org):
    return [max([0] + [src[org + x + (y + d) * pitch]
                       for d in range(-radius, radius + 1)])
            for y in range(height) for x in range(width)]


def golden_max_vh_sep(width, height, pitch, radius, src, org):
    # separable composition: horizontal pass over interior columns and the
    # radius-halo rows (the only cells the vertical pass consumes), then a
    # vertical pass.  All reads stay inside the 8px margin (radius <= 8).
    H2 = height + 2 * radius
    hpass = [0] * (width * H2)
    for yy in range(-radius, height + radius):
        for xx in range(width):
            hpass[xx + (yy + radius) * width] = max(
                [0] + [src[org + (xx + d) + yy * pitch]
                       for d in range(-radius, radius + 1)])
    return [max([0] + [hpass[x + (y + j + radius) * width]
                       for j in range(-radius, radius + 1)])
            for y in range(height) for x in range(width)]


def main():
    subprocess.run(["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "kfm_deblock_aux_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(131)
    ok = True
    total = 0

    def check(name, got, exp, info):
        nonlocal total, ok
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print(name, "MISMATCH", info, "px", i, g, e)
                    break
            return False
        return True

    # S: scale_qp (wrap pins: type0 q=64->0, q=255->252; OOR types 4/7)
    for t in range(200):
        width = rng.choice([1, 2, 3, 4, 5, 8, 13, 16, 31, 32, 40])
        height = rng.choice([1, 2, 3, 5, 8, 16, 24, 33])
        src_pitch = width + rng.choice([0, 0, 1, 2])
        dst_pitch = width + rng.choice([0, 1, 3])
        stype = rng.choice([0, 1, 2, 3, 3, 4, 7]) if t % 4 else t % 8
        nS = src_pitch * height
        craft = t % 4
        if craft == 0:
            src = [rng.randint(0, 255) for _ in range(nS)]
        elif craft == 1:  # full-range sweep hits every wrap edge
            src = [(i * 37 + t) % 256 for i in range(nS)]
        elif craft == 2:  # wrap-boundary values
            src = [rng.choice([0, 1, 62, 63, 64, 65, 127, 128, 254, 255])
                   for _ in range(nS)]
        else:
            src = [rng.choice([0, 51, 255])] * nS
        nums = ["S", width, height, dst_pitch, src_pitch, stype, nS] + src
        check("S", run_mirror(nums),
              golden_scale_qp(width, height, src_pitch, stype, src),
              (width, height, stype))

    # C: sharpen_coeff (boundary emphasis qp 191..215 i.e. q 23/24/25)
    for t in range(200):
        width = rng.choice([1, 2, 3, 4, 5, 8, 12, 16, 24, 32])
        height = rng.choice([1, 2, 3, 4, 7, 8, 12, 16])
        qp_pitch = width + rng.choice([0, 0, 1, 2])
        pitch = width + rng.choice([0, 1])
        nQ = qp_pitch * height
        craft = t % 5
        if craft == 0:
            qp = [rng.randint(0, 65535) for _ in range(nQ)]
        elif craft == 1:
            qp = [rng.randint(191, 215) for _ in range(nQ)]
        elif craft == 2:  # every q in 0..29 reachable (qp = 8q+r)
            qp = [(rng.randint(0, 29) * 8 + rng.randint(0, 7))
                  for _ in range(nQ)]
        elif craft == 3:  # production QP range + saturation tail
            qp = [rng.choice([0, 1, 40, 51 * 4, 200, 400, 1000, 65535])
                  for _ in range(nQ)]
        else:
            qp = [(i * 257 + t * 13) % 65536 for i in range(nQ)]
        nums = ["C", width, height, pitch, qp_pitch, nQ] + qp
        check("C", run_mirror(nums),
              golden_sharpen_coeff(width, height, qp_pitch, qp),
              (width, height))

    # H/V/B: max trio on 8px/side-padded planes, radius 1..8 (5 = production)
    for t in range(450):
        mode = "HVB"[t % 3]
        width = rng.choice([1, 2, 3, 4, 5, 8, 13, 16, 31, 32, 40])
        height = rng.choice([1, 2, 3, 5, 8, 13, 16, 24, 32])
        pitch = width + 16 + rng.choice([0, 0, 1, 2])
        radius = rng.choice([1, 2, 3, 5, 5, 5, 6, 8])
        nP = pitch * (height + 16)
        org = 8 + 8 * pitch
        craft = (t // 3) % 5
        if craft == 0:
            src = [rng.randint(0, 255) for _ in range(nP)]
        elif craft == 1:  # single spike pins the window extent exactly
            src = [0] * nP
            sx = rng.randrange(width)
            sy = rng.randrange(height)
            src[org + sx + sy * pitch] = 255
        elif craft == 2:  # all zeros -> 0-seed rule
            src = [0] * nP
        elif craft == 3:  # spikes on the margin edge (radius reach)
            src = [0] * nP
            for _ in range(3):
                mx = rng.randrange(-8, width + 8)
                my = rng.randrange(-8, height + 8)
                src[org + mx + my * pitch] = rng.randint(1, 255)
        else:  # ramp
            src = [(i * 11) % 256 for i in range(nP)]
        nums = [mode, width, height, pitch, radius, nP] + src
        if mode == "H":
            exp = golden_max_h(width, height, pitch, radius, src, org)
        elif mode == "V":
            exp = golden_max_v(width, height, pitch, radius, src, org)
        else:
            exp = golden_max_vh_sep(width, height, pitch, radius, src, org)
        check(mode, run_mirror(nums), exp, (width, height, radius))

    print("KFM KDeblock aux (scale_qp/sharpen_coeff/max_h/max_v/max_vh): "
          "PASS (%d cases)" % total)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
