/* CPU mirror of the NNEDI3 predictor kernel in
 * src/opencl/nnedi3/kernels/nnedi3_compute.cl (kl_compute_nn twin,
 * NNEDI3/nnedi3/nnedi3_kernel.cu @01931aa).
 *
 * Emulates one 16x32 group serially over its work list.  Float arithmetic
 * is unfused float32 (build with -ffp-contract=off) in upstream's operation
 * order, including the shuffle-down butterfly reduction tree (steps
 * 8,4,2,1) whose ORDER is observable for float sums.  dev_expf is
 * upstream's bit-twiddling approximation, transcribed.
 *
 * Usage: nnedi3_compute_ref <in> <out>
 *   All ints on one line (floats as bit patterns).  Mode:
 *     N xdia ydia nn qual val_min val_max refpitch dstpitch xbase ybase
 *       nWork nb work(2*nWork: x,y pairs)
 *       nWs ws(nWs)  nWf wfbits(nWf)  wspitch wfpitch
 *       nRef ref(nRef)
 *   ref is a dense plane at stride refpitch, already offset by the host so
 *   that (x,y) indexes the tile origin.  Output: two ints per written pixel,
 *   in work-list order: the clamped pixel, then the BIT PATTERN of the
 *   pre-rounding float `result * (1/qual)`.  The float bits are what make
 *   the summation order and dev_expf observable — the integer pixel alone
 *   rounds ULP-level differences away (mutation-tested; see the runner).
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <vector>
using namespace std;

static float bits2f(int b){ float f; memcpy(&f,&b,4); return f; }

enum { NN_BLOCK_W = 16, NN_BLOCK_H = 32 };

static float dev_expf(float f){
    const float exp_lo = -80.0f;
    const float exp_hi = +80.0f;
    const float e0_mult = 12102203.161561486f;
    const float e0_bias = 1064866805.0f;
    union { int i; float f; } t;
    float c = f < exp_lo ? exp_lo : (f > exp_hi ? exp_hi : f);
    t.i = (int)(c * e0_mult + e0_bias);
    return t.f;
}

/* butterfly reduce, steps 8,4,2,1; lane 0 result is what upstream uses */
template <typename T>
static T tree_reduce(vector<T>& v){
    for(int off=8;off>0;off>>=1){
        vector<T> prev=v;
        for(int t=0;t<NN_BLOCK_W;t++)
            if(t+off<NN_BLOCK_W) v[t]=prev[t]+prev[t+off];
    }
    return v[0];
}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: nnedi3_compute_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    char m=(char)rd();
    if(m!='N'){fprintf(stderr,"unknown mode %c\n",m);return 2;}

    int xdia=rd(),ydia=rd(),nn=rd(),qual=rd(),val_min=rd(),val_max=rd();
    int rp=rd(),dp=rd(),xbase=rd(),ybase=rd();
    int nWork=rd(); int nb=rd();
    vector<int> work(2*nWork); for(int&v:work)v=rd();
    int nWs=rd(); vector<int> ws(nWs); for(int&v:ws)v=rd();
    int nWf=rd(); vector<float> wf(nWf); for(float&v:wf)v=bits2f(rd());
    int wspitch=rd(), wfpitch=rd();
    int nRef=rd(); vector<int> ref(nRef); for(int&v:ref)v=rd();
    (void)dp;

    int K=xdia*ydia;
    vector<int> out;
    vector<vector<int>> B(NN_BLOCK_H, vector<int>(K));
    vector<float> avg(NN_BLOCK_H), var(NN_BLOCK_H), invvar(NN_BLOCK_H);
    vector<int> outx(NN_BLOCK_H), outy(NN_BLOCK_H);
    vector<int> wrote_val(NN_BLOCK_H), wrote_bits(NN_BLOCK_H);

    for(int b=0;b<nb;b+=NN_BLOCK_H){
        for(int ty=0;ty<NN_BLOCK_H;ty++){
            int xx=xbase, yy=ybase;
            if(b+ty<nb){ xx+=work[(b+ty)*2+0]; yy+=work[(b+ty)*2+1]; }
            outx[ty]=xx; outy[ty]=yy;
            for(int k=0;k<K;k++)
                B[ty][k]=ref[(size_t)(xx+yy*rp) + (k%xdia) + (size_t)(k/xdia)*rp];
        }

        for(int ty=0;ty<NN_BLOCK_H;ty++){
            vector<int> vs(NN_BLOCK_W,0), vq(NN_BLOCK_W,0);
            for(int tx=0;tx<NN_BLOCK_W;tx++){
                int sum=0,sumsq=0;
                for(int i=0;i<K/NN_BLOCK_W;i++){
                    int v=B[ty][tx+i*NN_BLOCK_W];
                    sum+=v; sumsq+=v*v;
                }
                vs[tx]=sum; vq[tx]=sumsq;
            }
            int sum=tree_reduce(vs), sumsq=tree_reduce(vq);
            float scale=1.0f/(float)K;
            float avg_=sum*scale;
            float var_=sumsq*scale-avg_*avg_;
            float invvar_;
            if(var_<=FLT_EPSILON){ var_=0.0f; invvar_=0.0f; }
            else { var_=sqrtf(var_); invvar_=1.0f/var_; }
            avg[ty]=avg_; var[ty]=var_; invvar[ty]=invvar_;
        }

        for(int ty=0;ty<NN_BLOCK_H;ty++){
            float result=0.0f;
            for(int q=0;q<qual;q++){
                vector<float> vv(NN_BLOCK_W,0.0f), vw(NN_BLOCK_W,0.0f);
                for(int tx=0;tx<NN_BLOCK_W;tx++){
                    float vsum=0.0f,wsum=0.0f;
                    for(int i=0;i<nn/NN_BLOCK_W;i++){
                        int j=i*NN_BLOCK_W+tx;
                        int sx=0,sy=0;
                        for(int k=0;k<K;k++){
                            int v=B[ty][k];
                            size_t wi=(size_t)(j+k*nn+q*wspitch)*2;
                            sx+=v*ws[wi+0]; sy+=v*ws[wi+1];
                        }
                        size_t f1=(size_t)(j+q*wfpitch)*2;
                        size_t f2=(size_t)(j+nn+q*wfpitch)*2;
                        float res0=(float)sx*wf[f1+0]*invvar[ty]+wf[f2+0];
                        float res1=(float)sy*wf[f1+1]*invvar[ty]+wf[f2+1];
                        res0=dev_expf(res0);
                        vsum+=res0*(res1/(1.0f+fabsf(res1)));
                        wsum+=res0;
                    }
                    vv[tx]=vsum; vw[tx]=wsum;
                }
                float vsum=tree_reduce(vv), wsum=tree_reduce(vw);
                const float min_weight_sum=1e-10f;
                if(wsum>min_weight_sum) result+=((5.0f*vsum)/wsum)*var[ty]+avg[ty];
                else result+=avg[ty];
            }
            float scale=1.0f/(float)qual;
            float pre=result*scale;
            int v=(int)(pre+0.5f);
            if(v<val_min)v=val_min; if(v>val_max)v=val_max;
            wrote_val[ty]=v;
            int pb; memcpy(&pb,&pre,4); wrote_bits[ty]=pb;
        }

        for(int ty=0;ty<NN_BLOCK_H;ty++)
            if(b+ty<nb){ out.push_back(wrote_val[ty]); out.push_back(wrote_bits[ty]); }
    }

    FILE* o=fopen(argv[2],"w");
    for(int v:out)fprintf(o,"%d\n",v);
    fclose(o);
    return 0;
}
