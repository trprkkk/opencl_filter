/* CPU mirror of kl_reduce_banding / cpu_reduce_banding (KFM/KDeband.cu), the
 * KDeband debanding core.  Transliterated verbatim from the authoritative CPU
 * function cpu_reduce_banding in that file (this is the OpenCL kernel
 * kf_deband_reduce_banding in src/opencl/kfm/kernels/kfm_deband.cl).
 *
 * The random byte stream `rand` is NOT regenerated here — it is an input (the
 * authoritative XorShift generator lives in the Python golden so that the two
 * reduce-band implementations are independent of it).
 *
 * Input (space-separated ints on one file):
 *   width height pitch range thresh sample_mode blur_first maxv
 *   rand_len src_len
 *   rand[rand_len]   (bytes 0..255)     -- length = 2*width*height when pitch==width
 *   src[src_len]     -- height rows of `pitch`, each row padded to pitch
 * Output: width*height ints (row-major over x in [0,width), y in [0,height)),
 *   dst[x + y*pitch]  -- the logical debanded plane.
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace std;
static int random_range(int random, int range){ // range <= 127
    return ((((range << 1) + 1) * random) >> 8) - range;
}
int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: kfm_deband_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    int width=rd(),height=rd(),pitch=rd(),range=rd(),thresh=rd();
    int sample_mode=rd(),blur_first=rd(),maxv=rd();
    int rand_len=rd(),src_len=rd();
    vector<int> randbuf(rand_len),src(src_len);
    for(int&i:randbuf)i=rd(); for(int&i:src)i=rd();
    (void)maxv;

    int rand_step=width*height;
    vector<int> out((size_t)width*height);
    for(int y=0;y<height;y++){
        for(int xx=0;xx<width;xx++){
            int offset=y*pitch+xx;
            int rl=range;
            if(y<rl)rl=y; {int t=height-y-1; if(t<rl)rl=t;}
            if(xx<rl)rl=xx; {int t=width-xx-1; if(t<rl)rl=t;}
            int range_limited=rl;
            int refA=random_range(randbuf[offset+rand_step*0],range_limited);
            int refB=random_range(randbuf[offset+rand_step*1],range_limited);
            int src_val=src[offset];
            int avg,diff;
            if(sample_mode==0){
                int ref=refA*pitch+refB;
                avg=src[offset+ref];
                int d=src_val-avg; if(d<0)d=-d; diff=d;
            } else if(sample_mode==1){
                int ref=refA*pitch+refB;
                int ref_p=src[offset+ref];
                int ref_m=src[offset-ref];
                avg=(ref_p+ref_m)>>1;
                if(blur_first){ int d=src_val-avg; if(d<0)d=-d; diff=d; }
                else { int a=src_val-ref_p; if(a<0)a=-a; int b=src_val-ref_m; if(b<0)b=-b; diff=a>b?a:b; }
            } else {
                int ref_0=refA*pitch+refB;
                int ref_1=refA-refB*pitch;
                int r0p=src[offset+ref_0], r0m=src[offset-ref_0];
                int r1p=src[offset+ref_1], r1m=src[offset-ref_1];
                avg=(r0p+r0m+r1p+r1m)>>2;
                if(blur_first){ int d=src_val-avg; if(d<0)d=-d; diff=d; }
                else {
                    int a0=src_val-r0p; if(a0<0)a0=-a0;
                    int a1=src_val-r0m; if(a1<0)a1=-a1;
                    int b0=src_val-r1p; if(b0<0)b0=-b0;
                    int b1=src_val-r1m; if(b1<0)b1=-b1;
                    int m0=a0>a1?a0:a1; int m1=b0>b1?b0:b1; diff=m0>m1?m0:m1;
                }
            }
            out[y*width+xx]=(diff<=thresh)?avg:src_val;
        }
    }
    FILE* o=fopen(argv[2],"w");
    for(int v:out) fprintf(o,"%d\n",v);
    if(o)fclose(o);
    return 0;
}
