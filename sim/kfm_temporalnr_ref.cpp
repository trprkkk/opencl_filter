/* CPU mirror of the KFM KTemporalNR kernel in src/opencl/kfm/kernels/
 * kfm_temporalnr.cl, faithful to the authoritative CPU twin `cpu_temporal_nr`
 * in KFM/KDeband.cu.  The kernel takes the nframes planes packed contiguously
 * (plane i at offset i*frame_stride) instead of CUDA's per-plane pointer array
 * (a host-layout seam only — the per-pixel arithmetic is identical).
 *
 * Float: avg = (float)sum / count + 0.5f in float32 with no FMA contraction;
 * compile with -ffp-contract=off.  The python golden emulates the same float32.
 *
 * Usage: kfm_temporalnr_ref <in> <out>
 *   in:  T width height pitch frame_stride nframes mid thresh
 *        NframesData frames[0..NframesData)      (>= nframes*frame_stride)
 *   out: width*height ints (logical region, row-major over width).
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace std;

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: kfm_temporalnr_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    int mode=rd(); if(mode!=(int)'T'){fprintf(stderr,"bad mode\n");return 2;}
    int width=rd(),height=rd(),pitch=rd(),frame_stride=rd();
    int nframes=rd(),mid=rd(),thresh=rd();
    int Ndata=rd();
    vector<int> frames(Ndata); for(int&i:frames)i=rd();

    vector<int> dst((size_t)width*height);
    for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++){
        size_t idx=(size_t)xx+(size_t)yy*(size_t)pitch;
        int center=frames[(size_t)mid*frame_stride+idx];
        int count=0,sum=0;
        for(int i=0;i<nframes;i++){
            int ref=frames[(size_t)i*frame_stride+idx];
            int diff=ref-center; if(diff<0)diff=-diff;
            if(diff<=thresh){count++;sum+=ref;}
        }
        float avg=(float)sum/(float)count+0.5f;
        dst[(size_t)xx+(size_t)yy*width]=(int)avg;
    }
    FILE* o=fopen(argv[2],"w");
    for(int v:dst)fprintf(o,"%d\n",v);
    fclose(o);
    return 0;
}
