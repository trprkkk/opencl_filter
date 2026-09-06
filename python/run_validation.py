#!/usr/bin/env python3
"""Cross-language validation of the KTGMC simple-kernel port.

Two independent implementations of the exact same algorithm (the CUDA kernels
in KTGMC/Kernel.cu):
  1. Python golden below (double precision, mirrors the CUDA host/kernel math).
  2. sim/ktgmc_cpu_ref.cpp (the CPU reference mirror used to build the port).

Both operate on identical raw planes; their integer outputs must match bit-for-
bit.  The OpenCL kernels in src/opencl/ktgmc/kernels/ktgmc_simple.cl are a
line-for-line transliteration of this same math (float accumulation there only
matches the original CUDA float path and is irrelevant for these integer
comparisons except resample, which is double here and float on the device).

Run:  python3 python/run_validation.py
Builds the C++ ref if needed.  No third-party python packages required.
"""
import os, random, struct, subprocess, sys, math

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(REPO, "build", "val")
os.makedirs(WORK, exist_ok=True)

W, PITCH = 16, 20
A_H = B_H = REF_H = 10
FIELD_H = 5

# ---------------------------------------------------------------- generation
def gen_plane(rng, h, maxval, blocky=True):
    n = PITCH * h
    arr = [0] * n
    for y in range(h):
        base = rng.randrange(0, maxval + 1)
        for x in range(W):
            v = base + rng.randrange(-6, 7)
            v = min(maxval, max(0, v))
            arr[y * PITCH + x] = v
    return arr

def make_inputs(bits):
    maxval = 255 if bits == 8 else 65535
    d = os.path.join(WORK, f"in_{bits}")
    os.makedirs(d, exist_ok=True)
    rng = random.Random(1234 + bits)
    planes = {}
    planes["a"]     = gen_plane(rng, A_H,     maxval)
    planes["b"]     = gen_plane(rng, B_H,     maxval)
    planes["ref"]   = gen_plane(rng, REF_H,   maxval)
    planes["field"] = gen_plane(rng, FIELD_H, maxval)
    fmt = "B" if bits == 8 else "H"
    for name in ("a", "b", "ref", "field"):
        with open(os.path.join(d, name + ".raw"), "wb") as f:
            f.write(struct.pack(fmt * len(planes[name]), *planes[name]))
    h = {"a": A_H, "b": B_H, "ref": REF_H, "field": FIELD_H}
    with open(os.path.join(d, "plane.info"), "w") as f:
        for name in ("a", "b", "ref", "field"):
            f.write(f"{name} {W} {h[name]} {PITCH}\n")
    return d

# ------------------------------------------------------- golden implementation
def clamp(v, lo, hi): return hi if v > hi else (lo if v < lo else v)

def build_resampling_program(source_size, crop_start, crop_size, target_size, b, c):
    def mitf(x, b, c):
        p0 = (6 - 2*b)/6; p2 = (-18 + 12*b + 6*c)/6; p3 = (12 - 9*b - 6*c)/6
        q0 = (8*b + 24*c)/6; q1 = (-12*b - 48*c)/6; q2 = (6*b + 30*c)/6; q3 = (-b - 6*c)/6
        x = abs(x)
        return (p0 + x*x*(p2 + x*p3)) if x < 1 else ((q0 + x*(q1 + x*(q2 + x*q3))) if x < 2 else 0.0)
    support = 2.0
    filter_scale = target_size / crop_size
    filter_step = min(filter_scale, 1.0)
    filter_support = support / filter_step
    fir = int(math.ceil(filter_support * 2))
    off, coef = [], []
    pos_step = crop_size / target_size
    pos = crop_start if fir == 1 else crop_start + ((crop_size - target_size) / (target_size * 2))
    for i in range(target_size):
        end_pos = int(pos + filter_support)
        if end_pos > source_size - 1: end_pos = source_size - 1
        start_pos = end_pos - fir + 1
        if start_pos < 0: start_pos = 0
        off.append(start_pos)
        ok_pos = clamp(pos, 0.0, float(source_size - 1))
        total = sum(mitf((start_pos + j - ok_pos) * filter_step, b, c) for j in range(fir))
        if total == 0.0: total = 1.0
        value = 0.0
        row = []
        for k in range(fir):
            nv = value + mitf((start_pos + k - ok_pos) * filter_step, b, c) / total
            row.append(nv - value); value = nv
        coef.append(row)
        pos += pos_step
    return fir, off, coef

def kernel_makediff(a, b, w, h, mode, range_half, maxval):
    out = [0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i = y*PITCH+x
            va, vb = a[i], b[i]
            v = (va - vb + range_half) if mode == 0 else (va + vb - range_half)
            out[i] = clamp(v, 0, maxval)
    return out

def kernel_rg_box3x3(src, w, h, mode, maxval):
    out = [0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i = y*PITCH+x
            s = src[i]; v = s
            if 1 <= x < w-1 and 1 <= y < h-1:
                g = lambda dx, dy: src[(y+dy)*PITCH + (x+dx)]
                ul,uc,ur = g(-1,-1),g(0,-1),g(1,-1)
                ml,     mr = g(-1,0),          g(1,0)
                ll,lc,lr = g(-1,1),g(0,1),g(1,1)
                if mode == 20:
                    vt = (ul+uc+ur + ml+s+mr + ll+lc+lr + 4)//9
                else:
                    h0=ul+2*uc+ur; h1=ml+2*s+mr; h2=ll+2*lc+lr
                    vt=(h0+2*h1+h2+8)>>4
                v = clamp(vt, 0, maxval)
            out[i] = v
    return out

def _sort8(a):
    def cas(i,j):
        if a[i]>a[j]: a[i],a[j]=a[j],a[i]
    cas(0,1);cas(2,3);cas(4,5);cas(6,7);cas(0,2);cas(1,3);cas(4,6);cas(5,7)
    cas(1,2);cas(5,6);cas(0,4);cas(1,5);cas(2,6);cas(3,7);cas(2,4);cas(3,5)
    cas(1,2);cas(3,4);cas(5,6)

def kernel_removegrain(src, w, h, n, maxval):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x; s=src[i]; v=s
            if 1<=x<w-1 and 1<=y<h-1:
                a=[src[(y-1)*PITCH+(x-1)],src[(y-1)*PITCH+x],src[(y-1)*PITCH+(x+1)],
                   src[y*PITCH+(x-1)],                src[y*PITCH+(x+1)],
                   src[(y+1)*PITCH+(x-1)],src[(y+1)*PITCH+x],src[(y+1)*PITCH+(x+1)]]
                _sort8(a); v=clamp(s,a[n-1],a[7-(n-1)])
            out[i]=v
    return out

def _sort9(a):
    def cas(i,j):
        if a[i]>a[j]: a[i],a[j]=a[j],a[i]
    cas(0,1);cas(3,4);cas(6,7);cas(1,2);cas(4,5);cas(7,8);cas(0,1);cas(3,4)
    cas(6,7);cas(0,3);cas(1,4);cas(2,5);cas(3,6);cas(4,7);cas(5,8);cas(0,3)
    cas(1,4);cas(2,5);cas(1,3);cas(5,7);cas(2,6);cas(4,6);cas(2,4);cas(2,3)
    cas(5,6)

def kernel_repair(src, ref, w, h, n, maxval):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x; s=src[i]; v=s
            if 1<=x<w-1 and 1<=y<h-1:
                a=[ref[(y-1)*PITCH+(x-1)],ref[(y-1)*PITCH+x],ref[(y-1)*PITCH+(x+1)],
                   ref[y*PITCH+(x-1)], s, ref[y*PITCH+(x+1)],
                   ref[(y+1)*PITCH+(x-1)],ref[(y+1)*PITCH+x],ref[(y+1)*PITCH+(x+1)]]
                _sort9(a); v=clamp(s,a[n-1],a[8-(n-1)])
            out[i]=v
    return out

def kernel_vclean(src, w, h):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x; b=src[i]; v=b
            if 1<=y<h-1:
                a=src[(y-1)*PITCH+x]; c=src[(y+1)*PITCH+x]
                v=min(max(min(a,b),c),max(a,b))
            out[i]=v
    return out

def kernel_tfr(src, w, h, is_uv, maxval):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x; s=float(src[i])
            d=(s-128.0)*(128.0/112.0)+128.0 if is_uv else (s-16.0)*(255.0/219.0)
            out[i]=int(clamp(d+0.5,0.0,float(maxval)))
    return out

def kernel_merge(a, b, w, h, weight, maxval):
    inv=32767-weight
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x
            v=(a[i]*inv + b[i]*weight + 16384) >> 15
            out[i]=clamp(v,0,maxval)
    return out

def kernel_resample_v(field, w, fir, off, coef, maxval, h_in, h_out):
    out=[0]*(PITCH*h_out)
    for y in range(h_out):
        begin=off[y]
        for x in range(w):
            acc=0.0
            for i in range(fir):
                acc += float(field[(begin+i)*PITCH+x])*coef[y][i]
            acc=clamp(acc,0.0,float(maxval))
            out[y*PITCH+x]=int(acc+0.5)
    return out

# ---------------------------------------------------------------- comparison
def read_plane(path, w, h, bits):
    with open(path,"rb") as f: raw=f.read()
    fmt="B" if bits==8 else "H"
    n=PITCH*h
    return list(struct.unpack(fmt*n, raw))

def run_bits(bits):
    maxval=255 if bits==8 else 65535
    in_dir=make_inputs(bits)
    out_dir=os.path.join(WORK, f"ref_{bits}")
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(["/tmp/ktgmc_ref", in_dir, out_dir, str(bits)], check=True)
    rng=random.Random(1234+bits)
    # reload inputs
    def ld(name,h):
        return read_plane(os.path.join(in_dir,name+".raw"), W, h, bits)
    a=ld("a",A_H); b=ld("b",B_H); ref=ld("ref",REF_H); field=ld("field",FIELD_H)

    expected={}
    if bits==8:
        expected["makediff"]=kernel_makediff(a,b,W,A_H,0,128,255)
        expected["adddiff"]=kernel_makediff(a,b,W,A_H,1,128,255)
    else:
        expected["makediff"]=kernel_makediff(a,b,W,A_H,0,32768,65535)
        expected["adddiff"]=kernel_makediff(a,b,W,A_H,1,32768,65535)
    expected["rg11"]=kernel_rg_box3x3(a,W,A_H,11,maxval)
    expected["rg20"]=kernel_rg_box3x3(a,W,A_H,20,maxval)
    for n in range(1,5):
        expected[f"rgclip{n}"]=kernel_removegrain(a,W,A_H,n,maxval)
        expected[f"repair{n}"]=kernel_repair(a,ref,W,A_H,n,maxval)
    expected["vclean"]=kernel_vclean(a,W,A_H)
    expected["tfr_y"]=kernel_tfr(a,W,A_H,0,maxval)
    expected["tfr_uv"]=kernel_tfr(a,W,A_H,1,maxval)
    expected["merge"]=kernel_merge(a,b,W,A_H,int(0.5*32767),maxval)

    fir,off,coef=build_resampling_program(5,0.25,5,10,0.0,0.5)
    expected["resample_v"]=kernel_resample_v(field,W,fir,off,coef,maxval,FIELD_H,10)

    ok=True
    for name,exp in expected.items():
        got=read_plane(os.path.join(out_dir,name+".raw"), W,
                       10 if name!="resample_v" else 10, bits)
        # note resample_v height 10 already; ensure correct
        if got!=exp:
            ok=False
            diffs=[i for i in range(len(exp)) if got[i]!=exp[i]][:5]
            print(f"  MISMATCH {name}: {len(diffs)}+ diffs, e.g. {diffs}")
    # compare resample program tables
    if bits==8:
        coff=[int(l) for l in open(os.path.join(out_dir,"prog_offset.txt"))]
        cf=[float(l) for l in open(os.path.join(out_dir,"prog_coef.txt"))]
        if coff!=off: ok=False; print("  MISMATCH prog_offset")
        cf2=[cf[y*fir+i] for y in range(10) for i in range(fir)]
        if cf2!=[c for row in coef for c in row]: ok=False; print("  MISMATCH prog_coef")
    print(f"[bitdepth {bits}] {'PASS' if ok else 'FAIL'}")
    return ok

if __name__=="__main__":
    # build C++ ref
    subprocess.run(["g++","-O2","-std=c++17","-w",
                    os.path.join(REPO,"sim","ktgmc_cpu_ref.cpp"),
                    "-o","/tmp/ktgmc_ref"], check=True)
    allok=True
    for b in (8,16):
        allok &= run_bits(b)
    sys.exit(0 if allok else 1)
