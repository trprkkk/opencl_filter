/* CPU mirror of the fifteen AvsCUDA Conditional kernels in
 * src/opencl/avscuda/kernels/avscuda_conditional.cl (kl_init_sum /
 * kl_sum_of_pixels / kl_sad / kl_init_hist / kl_count_hist twins,
 * AvsCUDA/filters/ConditionalFunctions.cu).  The mirror accumulates
 * serially; integer sums are order-free so this equals any atomic arrival
 * order (u32 wraparound included).  u64 sums print as one decimal (%llu).
 * The float sum/SAD reductions are // RIG-VERIFY (no mirror: their cross-
 * block arrival order is undefined upstream too).
 *
 * Usage: avscuda_conditional_ref <in> <out>
 *   All ints on one line.  Modes (widths a multiple of 4, like production):
 *     A init_u32 : A len -> len zeros
 *     B init_u64 : B len -> len zeros
 *     C init_f32 : C len -> len zeros (0.0f bits)
 *     D sum8_32  : D w h pitch maxv  nP src(nP) -> one u32 sum (%u)
 *     E sum8_64  : E w h pitch maxv  nP src(nP) -> one u64 sum (%llu)
 *     F sum16_32 : F w h pitch maxv  nP src(nP) -> one u32 sum of min(v,maxv)
 *     G sum16_64 : G w h pitch maxv  nP src(nP) -> one u64 sum of min(v,maxv)
 *     H sad8_32  : H w h pitch maxv  nP s0(nP) s1(nP) -> one u32 |a-b| sum
 *     I sad8_64  : I w h pitch maxv  nP s0(nP) s1(nP) -> one u64 |a-b| sum
 *     J sad16_32 : J w h pitch maxv  nP s0(nP) s1(nP) -> one u32 clamped sum
 *     K sad16_64 : K w h pitch maxv  nP s0(nP) s1(nP) -> one u64 clamped sum
 *     L hist8    : L w h pitch maxv len  nP src(nP) -> len ints
 *     M hist16   : M w h pitch maxv len  nP src(nP) -> len ints
 *     N histf32  : N w h pitch maxv len  nP srcbits(nP) -> len ints
 *     O init_hist: O len -> len zeros
 */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace std;

static float bits2f(int b){ float f; memcpy(&f,&b,4); return f; }

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: avscuda_conditional_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    char m=(char)rd();

    auto read=[&](int n){ vector<int> a(n); for(int&i:a)i=rd(); return a; };
    FILE* o=fopen(argv[2],"w");

    if(m=='A'||m=='B'||m=='C'||m=='O'){
        int len=rd();
        for(int i=0;i<len;i++)fprintf(o,"0\n");
    } else if(m=='D'||m=='E'||m=='F'||m=='G'){
        int w=rd(),h=rd(),pitch=rd(),maxv=rd();
        int nP=rd(); vector<int> src=read(nP);
        if(m=='D'||m=='F'){
            uint32_t s=0;
            for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++){
                int v=src[(size_t)(xx+yy*pitch)];
                if(m=='F'&&v>maxv)v=maxv;
                s+=(uint32_t)v;
            }
            fprintf(o,"%u\n",s);
        } else {
            unsigned long long s=0;
            for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++){
                int v=src[(size_t)(xx+yy*pitch)];
                if(m=='G'&&v>maxv)v=maxv;
                s+=(unsigned long long)v;
            }
            fprintf(o,"%llu\n",s);
        }
    } else if(m=='H'||m=='I'||m=='J'||m=='K'){
        int w=rd(),h=rd(),pitch=rd(),maxv=rd();
        int nP=rd(); vector<int> s0=read(nP), s1=read(nP);
        bool wide=(m=='I'||m=='K'), cl=(m=='J'||m=='K');
        if(!wide){
            uint32_t s=0;
            for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++){
                int a=s0[(size_t)(xx+yy*pitch)], b=s1[(size_t)(xx+yy*pitch)];
                if(cl){ if(a>maxv)a=maxv; if(b>maxv)b=maxv; }
                int d=a-b; s+=(uint32_t)(d>=0?d:-d);
            }
            fprintf(o,"%u\n",s);
        } else {
            unsigned long long s=0;
            for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++){
                int a=s0[(size_t)(xx+yy*pitch)], b=s1[(size_t)(xx+yy*pitch)];
                if(cl){ if(a>maxv)a=maxv; if(b>maxv)b=maxv; }
                int d=a-b; s+=(unsigned long long)(d>=0?d:-d);
            }
            fprintf(o,"%llu\n",s);
        }
    } else if(m=='L'||m=='M'||m=='N'){
        int w=rd(),h=rd(),pitch=rd(),maxv=rd(),len=rd();
        int nP=rd(); vector<int> src=read(nP);
        vector<int> hist(len,0);
        for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++){
            int idx;
            if(m=='N'){
                float s=bits2f(src[(size_t)(xx+yy*pitch)]);
                float t=s*65535.0f+0.5f;
                float c=fmaxf(0.0f,fminf(t,65535.0f));
                idx=(int)c;
            } else {
                idx=src[(size_t)(xx+yy*pitch)];
                if(m=='M'&&idx>maxv)idx=maxv;
            }
            if(idx<len)hist[idx]++;
        }
        for(int v:hist)fprintf(o,"%d\n",v);
    } else { fprintf(stderr,"bad mode %c\n",m); return 2; }
    fclose(o);
    return 0;
}
