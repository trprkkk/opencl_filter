/* CPU mirror of the eight AvsCUDA resizer kernels in
 * src/opencl/avscuda/kernels/avscuda_resample.cl (kl_resize_* twins,
 * AvsCUDA/filters/resample.cu).  Float accumulation is unfused float32
 * (build with -ffp-contract=off); int outputs clamp via fminf/fmaxf
 * (NaN -> limit, matching CUDA) then truncate.  The mirror reads the
 * LOGICAL programs (per-output coeff rows), like the port.
 *
 * Usage: avscuda_resample_ref <in> <out>
 *   All ints on one line (floats as bit patterns).  Modes:
 *     A vpt    : A tw th dp sp  nS src(nS) nO offs(nO)
 *                -> tw*th row-selected pixels
 *     B vpt_f32: B tw th dp sp  nS srcbits(nS) nO offs(nO) -> tw*th bits
 *     C vplanar: C tw th dp sp fs limit  nS src(nS) nO offs(nO) nC coeff(nC)
 *                -> tw*th clamped ints
 *     D vplanf : D tw th dp sp fs  nS srcbits(nS) nO offs(nO) nC coeff(nC)
 *                -> tw*th float bits
 *     E hpt    : E twu th dp sp U  nS src(nS) nO offs(nO)
 *                -> twu*U*th selected bytes
 *     F hplan8 : F twu th dp sp U fs limit  nS src(nS) nO offs(nO) nC coeff(nC)
 *                -> twu*U*th clamped ints
 *     G hplan16: G twu th dp sp U fs limit  nS src(nS) nO offs(nO) nC coeff(nC)
 *                -> twu*U*th clamped ints
 *     H hplanf : H twu th dp sp U fs  nS srcbits(nS) nO offs(nO) nC coeff(nC)
 *                -> twu*U*th float bits
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace std;

static float bits2f(int b){ float f; memcpy(&f,&b,4); return f; }
static int f2bits(float f){ int b; memcpy(&b,&f,4); return b; }

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: avscuda_resample_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    char m=(char)rd();
    vector<int> out;

    auto read=[&](int n){ vector<int> a(n); for(int&i:a)i=rd(); return a; };

    if(m=='A'){
        int tw=rd(),th=rd(),dp=rd(),sp=rd();
        (void)dp;
        int nS=rd(); vector<int> src=read(nS);
        int nO=rd(); vector<int> off=read(nO);
        for(int yy=0;yy<th;yy++)for(int xx=0;xx<tw;xx++)
            out.push_back(src[(size_t)(xx+off[yy]*sp)]);
    } else if(m=='B'){
        int tw=rd(),th=rd(),dp=rd(),sp=rd();
        (void)dp;
        int nS=rd(); vector<int> src=read(nS);
        int nO=rd(); vector<int> off=read(nO);
        for(int yy=0;yy<th;yy++)for(int xx=0;xx<tw;xx++)
            out.push_back(src[(size_t)(xx+off[yy]*sp)]);
    } else if(m=='C'){
        int tw=rd(),th=rd(),dp=rd(),sp=rd(),fs=rd();
        float limit=(float)rd();
        (void)dp;
        int nS=rd(); vector<int> src=read(nS);
        int nO=rd(); vector<int> off=read(nO);
        int nC=rd(); vector<int> cb=read(nC);
        for(int yy=0;yy<th;yy++)for(int xx=0;xx<tw;xx++){
            float r=0.5f;
            for(int i=0;i<fs;i++)
                r+=bits2f(cb[(size_t)(yy*fs+i)])*(float)src[(size_t)(xx+(off[yy]+i)*sp)];
            out.push_back((int)fmaxf(0.0f,fminf(r,limit)));
        }
    } else if(m=='D'){
        int tw=rd(),th=rd(),dp=rd(),sp=rd(),fs=rd();
        (void)dp;
        int nS=rd(); vector<int> src=read(nS);
        int nO=rd(); vector<int> off=read(nO);
        int nC=rd(); vector<int> cb=read(nC);
        for(int yy=0;yy<th;yy++)for(int xx=0;xx<tw;xx++){
            float r=0.0f;
            for(int i=0;i<fs;i++)
                r+=bits2f(cb[(size_t)(yy*fs+i)])*bits2f(src[(size_t)(xx+(off[yy]+i)*sp)]);
            out.push_back(f2bits(r));
        }
    } else if(m=='E'){
        int twu=rd(),th=rd(),dp=rd(),sp=rd(),U=rd();
        int nS=rd(); vector<int> src=read(nS);
        int nO=rd(); vector<int> off=read(nO);
        for(int yy=0;yy<th;yy++)for(int e=0;e<twu*U;e++){
            int unit=e/U, lane=e%U;
            out.push_back(src[(size_t)(off[unit]*U+lane+yy*sp)]);
        }
    } else if(m=='F'||m=='G'){
        int twu=rd(),th=rd(),dp=rd(),sp=rd(),U=rd(),fs=rd();
        float limit=(float)rd();
        (void)dp;
        int nS=rd(); vector<int> src=read(nS);
        int nO=rd(); vector<int> off=read(nO);
        int nC=rd(); vector<int> cb=read(nC);
        for(int yy=0;yy<th;yy++)for(int e=0;e<twu*U;e++){
            int unit=e/U, lane=e%U;
            float r=0.5f;
            for(int i=0;i<fs;i++)
                r+=bits2f(cb[(size_t)(unit*fs+i)])*(float)src[(size_t)((off[unit]+i)*U+lane+yy*sp)];
            out.push_back((int)fmaxf(0.0f,fminf(r,limit)));
        }
    } else if(m=='H'){
        int twu=rd(),th=rd(),dp=rd(),sp=rd(),U=rd(),fs=rd();
        (void)dp;
        int nS=rd(); vector<int> src=read(nS);
        int nO=rd(); vector<int> off=read(nO);
        int nC=rd(); vector<int> cb=read(nC);
        for(int yy=0;yy<th;yy++)for(int e=0;e<twu*U;e++){
            int unit=e/U, lane=e%U;
            float r=0.0f;
            for(int i=0;i<fs;i++)
                r+=bits2f(cb[(size_t)(unit*fs+i)])*bits2f(src[(size_t)((off[unit]+i)*U+lane+yy*sp)]);
            out.push_back(f2bits(r));
        }
    } else { fprintf(stderr,"bad mode %c\n",m); return 2; }
    FILE* o=fopen(argv[2],"w");
    for(int v:out)fprintf(o,"%d\n",v);
    fclose(o);
    return 0;
}
