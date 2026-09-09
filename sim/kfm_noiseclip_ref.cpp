/* CPU mirror of the KFM KNoiseClip kernel in src/opencl/kfm/kernels/
 * kfm_noiseclip.cl, faithful to cpu_noise_clip / dev_limitter in
 * KFM/DecombeUCF.cu.  8-bit only, integer-exact.
 *
 * Usage: kfm_noiseclip_ref <in> <out>
 *   in:  N width height pitch nmin range n  src(...) noise(...)
 *   out: width*height ints (row-major over width).
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace std;

static int limitter(int s, int nmin, int range){
    if (s == 128) return 128;
    if (s < 128) return (((127 - range) < s) && (s < (128 - nmin))) ? 0 : 56;
    return (((128 + nmin) < s) && (s < (129 + range))) ? 255 : 199;
}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: kfm_noiseclip_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    int mode=rd(); if(mode!=(int)'N'){fprintf(stderr,"bad mode\n");return 2;}
    int width=rd(),height=rd(),pitch=rd(),nmin=rd(),range=rd(),n=rd();
    vector<int> src(n),noise(n);
    for(int&i:src)i=rd(); for(int&i:noise)i=rd();
    vector<int> out((size_t)width*height);
    for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++){
        int off=xx+yy*pitch;
        int s=(src[off]-noise[off]+256)>>1;
        out[(size_t)yy*width+xx]=limitter(s,nmin,range);
    }
    FILE* o=fopen(argv[2],"w");
    for(int v:out)fprintf(o,"%d\n",v);
    fclose(o);
    return 0;
}
