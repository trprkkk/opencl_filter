/* CPU mirror of the four AvsCUDA merge kernels in
 * src/opencl/avscuda/kernels/avscuda_merge.cl (kl_merge_plane /
 * kl_average_plane twins, AvsCUDA/filters/merge.cu).  In-place on src
 * upstream (dual pitch); the mirror reads src+other and emits the merged
 * plane, which is the same values for these elementwise ops.
 *
 * Usage: avscuda_merge_ref <in> <out>
 *   All ints on one line.  Modes:
 *     A merge      : A w h sp op wi iw maxv  nS src(nS) nO other(nO)
 *                    -> output w*h of ((a*iw+b*wi+16384)>>15) & maxv
 *                       (the >>15 SIMD/device scale; the scalar-C >>16 twin
 *                       uses a different weight scale and is NOT replicated)
 *     B merge_f32  : B w h sp op wf_bits iwf_bits  nS src(nS) nO other(nO)
 *                    (float bit patterns) -> output w*h bit patterns of
 *                    a*iwf+b*wf, unfused float32
 *     C average    : C w h sp op maxv  nS src(nS) nO other(nO)
 *                    -> output w*h of (a+b+1)>>1 (exact average_plane_c twin)
 *     D average_f32: D w h sp op  nS src(nS) nO other(nO)
 *                    (float bit patterns) -> output w*h bit patterns of
 *                    (a+b)*0.5f (device form; golden uses (a+b)/2.0f)
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace std;

static float bits2f(int b){ float f; memcpy(&f,&b,4); return f; }
static int f2bits(float f){ int b; memcpy(&b,&f,4); return b; }

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: avscuda_merge_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    char m=(char)rd();
    vector<int> out;

    auto read=[&](int n){ vector<int> a(n); for(int&i:a)i=rd(); return a; };

    if(m=='A'){
        int w=rd(),h=rd(),sp=rd(),op=rd(),wi=rd(),iw=rd(),maxv=rd();
        int nS=rd(); vector<int> src=read(nS);
        int nO=rd(); vector<int> oth=read(nO);
        for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++){
            int a=src[(size_t)(xx+yy*sp)], b=oth[(size_t)(xx+yy*op)];
            out.push_back(((a*iw + b*wi + 16384) >> 15) & maxv);
        }
    } else if(m=='B'){
        int w=rd(),h=rd(),sp=rd(),op=rd();
        float wf=bits2f(rd()), iwf=bits2f(rd());
        int nS=rd(); vector<int> src=read(nS);
        int nO=rd(); vector<int> oth=read(nO);
        for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++){
            float a=bits2f(src[(size_t)(xx+yy*sp)]);
            float b=bits2f(oth[(size_t)(xx+yy*op)]);
            out.push_back(f2bits(a*iwf + b*wf));
        }
    } else if(m=='C'){
        int w=rd(),h=rd(),sp=rd(),op=rd(),maxv=rd();
        (void)maxv;
        int nS=rd(); vector<int> src=read(nS);
        int nO=rd(); vector<int> oth=read(nO);
        for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++){
            int a=src[(size_t)(xx+yy*sp)], b=oth[(size_t)(xx+yy*op)];
            out.push_back((a + b + 1) >> 1);
        }
    } else if(m=='D'){
        int w=rd(),h=rd(),sp=rd(),op=rd();
        int nS=rd(); vector<int> src=read(nS);
        int nO=rd(); vector<int> oth=read(nO);
        for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++){
            float a=bits2f(src[(size_t)(xx+yy*sp)]);
            float b=bits2f(oth[(size_t)(xx+yy*op)]);
            out.push_back(f2bits((a + b) * 0.5f));
        }
    } else { fprintf(stderr,"bad mode %c\n",m); return 2; }
    FILE* o=fopen(argv[2],"w");
    for(int v:out)fprintf(o,"%d\n",v);
    fclose(o);
    return 0;
}
