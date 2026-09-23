/* Emulation shim so .cl kernel files can be parsed (and structurally linted)
 * as plain C. This is a PARSE-ONLY aid for catching typos / unbalanced code;
 * it does not reproduce OpenCL semantics (vector math, address spaces, ...).
 * Usage: cp file.cl file.c && gcc -fsyntax-only -x c -DPX=... -include oc_shim.h file.c
 */
#include <stddef.h>
#include <stdbool.h>

/* scalar types */
typedef unsigned char  uchar;
typedef unsigned short ushort;
typedef unsigned int   uint;
typedef unsigned long  ulong;
typedef unsigned char  half;   /* nominal; never parsed arithmetically here */

/* address-space / qualifier keywords (parse-only) */
#define __global
#define __constant
#define __local
#define __private
#define __kernel
#define kernel
#define __restrict
#define __read_only
#define __write_only
#define __attribute__(x) /* drop */

/* 2/3/4-component vectors used by the kernels, as structs so .x/.y/.z work */
typedef struct { int x,y; }                 int2;
typedef struct { int x,y,z; }                int3;
typedef struct { int x,y,z,w; }              int4;
typedef struct { uint x,y,z,w; }             uint4;
typedef struct { uchar x,y,z,w; }            uchar4;
typedef struct { float x,y,z,w; }            float4;
typedef struct { short x,y; }                short2;

/* work-item / group queries */
static int _gi[3], _li[3], _gid[3], _ls[3], _ng[3];
#define get_global_id(i)  ((int)_gi[(i)])
#define get_local_id(i)   ((int)_li[(i)])
#define get_group_id(i)   ((int)_gid[(i)])
#define get_local_size(i) ((int)_ls[(i)])
#define get_num_groups(i) ((int)_ng[(i)])
#define barrier(x)
#define mem_fence(x)

/* OpenCL builtin functions -> plain C equivalents/macros (parse-only) */
#define abs(x)     ((x) < 0 ? -(x) : (x))
#define fabs(x)    ((x) < 0 ? -(x) : (x))
#define min(a,b)   ((a) < (b) ? (a) : (b))
#define max(a,b)   ((a) > (b) ? (a) : (b))
#define fmin(a,b)  min(a,b)
#define fmax(a,b)  max(a,b)
#define clamp(x,a,b) ((x) < (a) ? (a) : ((x) > (b) ? (b) : (x)))
#define floor(x)   (x)
#define round(x)   (x)
#define ceil(x)    (x)
#define dot(a,b)   (a)
#define cross(a,b) (a)
#define mad(a,b,c) ((a)*(b)+(c))
#define convert_int(x)    ((int)(x))
#define convert_uint(x)   ((uint)(x))
#define convert_float(x)  ((float)(x))
#define convert_uchar(x)  ((uchar)(x))
#define convert_ushort(x) ((ushort)(x))
#define convert_short(x)  ((short)(x))
#define convert_int_rte(x)    ((int)(x))
#define convert_int_rtz(x)    ((int)(x))
#define convert_float_rte(x)  ((float)(x))
#define convert_float_rtz(x)  ((float)(x))

#define ATOMIC_FUNC_IMPL(T) \
static T atomic_xchg(__global T* p, T v){ return __sync_lock_test_and_set(p,v);}
#define ATOMIC_FUNCS ATOMIC_FUNC_IMPL(int)
ATOMIC_FUNCS
/* macro (not overloaded fns — C has no overloading): parses for both int and
 * unsigned long args, matching the OpenCL atomic_add builtin's call shape. */
#define atomic_add(p, v) (__sync_fetch_and_add((p), (v)))
/* 64-bit base atomics (cl_khr_int64_base_atomics call shape; parse-only). */
#define atom_add(p, v) (__sync_fetch_and_add((p), (v)))
static int atomic_max(__global int* p, int v){ return __sync_fetch_and_max(p,v);}
/* Parse-only stubs for the float-atomic CAS loop in
 * avscuda_conditional_rig.cl (never executed; the real device code uses
 * true bit-reinterpretation and hardware CAS). */
#define as_float(x)  ((float)(x))
#define as_uint(x)   ((uint)(x))
static uint atomic_cmpxchg(__global uint* p, uint cmp, uint v){
    return __sync_val_compare_and_swap(p, cmp, v); }
