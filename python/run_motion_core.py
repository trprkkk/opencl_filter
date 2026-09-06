#!/usr/bin/env python3
"""Validate the motion-core weight helpers from
src/opencl/ktgmc/kernels/ktgmc_motion.cl:
    kt_degrain_weight  (dev_degrain_weight)
    kt_norm_weights    (dev_norm_weights, delta 1..6)
against the CPU mirror sim/ktgmc_weight_ref.cpp. Pure integer arithmetic, so
they must match the independent Python golden exactly.

Run:  python3 python/run_motion_core.py
"""
import os, random, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def degw(th, bs):
    if th <= bs:
        return 0
    sqt = float(th) * th
    sqb = float(bs) * bs
    return int(256.0 * (sqt - sqb) / (sqt + sqb))

def norm(d, b, flat):
    flat = list(flat) + [0] * (12 - len(flat))
    WB = [flat[2 * i] for i in range(6)]
    WF = [flat[2 * i + 1] for i in range(6)]
    WSrc = 256
    if b:
        if d == 1: WSrc *= 2
        elif d == 2: WSrc *= 6; WB[0] *= 4; WF[0] *= 4
        elif d == 3: WSrc *= 20; WB[0] *= 15; WF[0] *= 15; WB[1] *= 6; WF[1] *= 6
        elif d == 4: WSrc *= 70; WB[0] *= 56; WF[0] *= 56; WB[1] *= 28; WF[1] *= 28; WB[2] *= 8; WF[2] *= 8
    if d == 6: WS = sum(WB[:6]) + sum(WF[:6]) + WSrc + 1
    elif d == 5: WS = sum(WB[:5]) + sum(WF[:5]) + WSrc + 1
    elif d == 4: WS = sum(WB[:4]) + sum(WF[:4]) + WSrc + 1
    elif d == 3: WS = sum(WB[:3]) + sum(WF[:3]) + WSrc + 1
    elif d == 2: WS = sum(WB[:2]) + sum(WF[:2]) + WSrc + 1
    else:        WS = WB[0] + WF[0] + WSrc + 1
    for i in range(6):
        WB[i] = WB[i] * 256 // WS
        WF[i] = WF[i] * 256 // WS
    if d == 6: WSrc = 256 - sum(WB[:6]) - sum(WF[:6])
    elif d == 5: WSrc = 256 - sum(WB[:5]) - sum(WF[:5])
    elif d == 4: WSrc = 256 - sum(WB[:4]) - sum(WF[:4])
    elif d == 3: WSrc = 256 - sum(WB[:3]) - sum(WF[:3])
    elif d == 2: WSrc = 256 - sum(WB[:2]) - sum(WF[:2])
    else:        WSrc = 256 - WB[0] - WF[0]
    return [WSrc] + [x for pair in zip(WB[:d], WF[:d]) for x in pair]

def main():
    rng = random.Random(99)
    lines = []
    for _ in range(300):
        th, bs = rng.randint(0, 5000), rng.randint(0, 5000)
        lines.append(f"w {th} {bs}")
    lines += ["w 0 0", "w 1 0", "w 0 1", "w 100 100", "w 100 99", "w 1000 700"]
    for _ in range(400):
        d = rng.randint(1, 6); b = rng.randint(0, 1)
        w = [rng.randint(0, 256) for _ in range(2 * d)]
        lines.append("n %d %d %s" % (d, b, " ".join(map(str, w))))

    exp = []
    for ln in lines:
        if ln[0] == 'w':
            th, bs = map(int, ln.split()[1:]); exp.append("w %d" % degw(th, bs))
        else:
            p = ln.split(); r = norm(int(p[1]), int(p[2]), list(map(int, p[3:])))
            exp.append("n %d %s" % (r[0], " ".join(map(str, r[1:]))))

    subprocess.run(["g++", "-O2", "-std=c++17", "-w",
                    os.path.join(REPO, "sim", "ktgmc_weight_ref.cpp"),
                    "-o", "/tmp/wref"], check=True)
    out = subprocess.run(["/tmp/wref"], input="\n".join(lines) + "\n",
                         capture_output=True, text=True).stdout.splitlines()
    mism = sum(1 for e, g in zip(exp, out) if e != g)
    ok = mism == 0 and len(exp) == len(out)
    print(f"motion-core weight helpers: {'PASS' if ok else 'FAIL'} ({len(exp)} cases, mism={mism})")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
