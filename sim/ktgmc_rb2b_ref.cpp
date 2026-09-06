/* CPU mirror of kl_RB2B_bilinear_filtered (kt_rb2b_bilinear_filtered in
 * src/opencl/ktgmc/kernels/ktgmc_motion.cl), translated from the authoritative
 * CPU reference RB2BilinearFilteredVertical + RB2BilinearFilteredHorizontalInplace
 * in KTGMC/MV.cpp (Fizick (1,3,3,1)/8 half-band 1:2 downsample). This is the
 * two-phase separable form with an explicit intermediate plane.
 *
 * Usage: ktgmc_rb2b_ref <infile> <outfile>
 * Input (space-separated ints):
 *   nWidth nHeight srcPitch dstPitch
 *   then (srcPitch * 2*nHeight) source samples (rows 0..2*nHeight-1)
 * Output: for each of (nWidth*nHeight) dst pixels, one value (row-major).
 * Pixels are ints in [0,maxval]; the .cl kernel operates on PX (uchar/ushort).
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace std;

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: rb2b <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> n; int x;
    while(f&&fscanf(f,"%d",&x)==1) n.push_back(x);
    if(f)fclose(f);
    int p=0;
    int nWidth=n[p++],nHeight=n[p++],srcPitch=n[p++],dstPitch=n[p++];
    int srcH=2*nHeight;
    vector<int> src(srcPitch*(long)srcH);
    for(size_t i=0;i<src.size();++i) src[i]=n[p++];

    // ---- vertical phase -> intermediate [nHeight][2*nWidth] ----
    vector<int> iv((size_t)nHeight*2*nWidth);
    auto IV=[&](int y,int c)->int&{return iv[(size_t)y*2*nWidth+c];};
    for(int y=0;y<nHeight;y++){
        for(int c=0;c<2*nWidth;c++){
            int v;
            if(y==0){
                v=(src[c]+src[c+srcPitch]+1)>>1;
            } else if(y<nHeight-1){
                v=(src[c+(2*y-1)*srcPitch]
                  + src[c+(2*y)*srcPitch]*3
                  + src[c+(2*y+1)*srcPitch]*3
                  + src[c+(2*y+2)*srcPitch]+4)/8;
            } else {
                v=(src[c+(2*y)*srcPitch]+src[c+(2*y+1)*srcPitch]+1)>>1;
            }
            IV(y,c)=v;
        }
    }

    // ---- horizontal phase -> dst ----
    vector<int> dst((size_t)nWidth*nHeight);
    for(int y=0;y<nHeight;y++){
        for(int x=0;x<nWidth;x++){
            int v;
            if(x==0||x==nWidth-1){
                v=(IV(y,2*x)+IV(y,2*x+1)+1)>>1;
            } else {
                v=(IV(y,2*x-1)+IV(y,2*x)*3+IV(y,2*x+1)*3+IV(y,2*x+2)+4)/8;
            }
            dst[x+y*nWidth]=v;
        }
    }

    FILE* o=fopen(argv[2],"w");
    for(int v: dst) fprintf(o,"%d\n",v);
    if(o)fclose(o);
    return 0;
}
