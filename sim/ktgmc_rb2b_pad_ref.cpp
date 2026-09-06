/* CPU mirror of kl_RB2B_bilinear_filtered_with_pad (kt_rb2b_bilinear_filtered_with_pad
 * in src/opencl/ktgmc/kernels/ktgmc_motion.cl), transliterated verbatim from the
 * CUDA kernel of the same name in KTGMC/MVKernel.cu (used by MV.cpp ReduceToPad).
 *
 * This is the CUDA-only FUSED 4x4-tap anti-aliased 1:2 downsample that fills the
 * destination plane's hpad/vpad border in one pass with a single +32/64 rounding.
 * It is numerically DISTINCT from the two-phase separable RB2BilinearFiltered (the
 * CPU ReduceToPad twin runs that + a separate Pad()); there is therefore no
 * separable host routine to cross-check against — this mirror and the Python
 * golden independently re-implement the same fused CUDA algorithm so that the .cl
 * transliteration is bit-for-bit cross-checked for transcription errors.
 *
 * Usage: ktgmc_rb2b_pad_ref <infile> <outfile>
 * Input (space-separated ints):
 *   nWidth nHeight srcPitch dstPitch hpad vpad
 *   then (srcPitch * 2*nHeight) source samples, row 0..2*nHeight-1.
 * Output: row-major over the full padded dst rect
 *   (nWidth+2*hpad) x (nHeight+2*vpad), where output[(dsty+vpad)*W + (dstx+hpad)]
 *   = dst[dstx + dsty*dstPitch] for dstx in [-hpad, nWidth+hpad-1].
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace std;

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: rb2bpad <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> n; int x;
    while(f&&fscanf(f,"%d",&x)==1) n.push_back(x);
    if(f)fclose(f);
    int p=0;
    int nWidth=n[p++],nHeight=n[p++],srcPitch=n[p++],dstPitch=n[p++];
    int hpad=n[p++],vpad=n[p++];
    int srcH=2*nHeight;
    vector<int> src((size_t)srcPitch*srcH);
    for(size_t i=0;i<src.size();++i) src[i]=n[p++];

    int W=nWidth+2*hpad, H=nHeight+2*vpad;
    vector<int> out((size_t)W*H);
    for(int dsty=-vpad; dsty<nHeight+vpad; dsty++){
        int ymul0=0,ymul1=4,srcy;
        if(dsty<=0)                 srcy=0;
        else if(dsty>=nHeight-1)    srcy=(nHeight-1)*2;
        else{ srcy=dsty*2; ymul0=1; ymul1=3; }
        for(int dstx=-hpad; dstx<nWidth+hpad; dstx++){
            int xmul0=0,xmul1=4,srcx;
            if(dstx<=0)              srcx=0;
            else if(dstx>=nWidth-1)  srcx=(nWidth-1)*2;
            else{ srcx=dstx*2; xmul0=1; xmul1=3; }
            int sum=0;
            for(int j=-1;j<=2;j++){
                int ymul=(j==0||j==1)?ymul1:ymul0;
                if(ymul>0){
                    for(int i=-1;i<=2;i++){
                        int xmul=(i==0||i==1)?xmul1:xmul0;
                        if(xmul>0){
                            int pix=src[(size_t)(srcx+i)+(size_t)(srcy+j)*srcPitch];
                            sum+=pix*ymul*xmul;
                        }
                    }
                }
            }
            sum=(sum+32)/64;
            out[(size_t)(dsty+vpad)*W+(dstx+hpad)]=sum;
        }
    }

    FILE* o=fopen(argv[2],"w");
    for(int v: out) fprintf(o,"%d\n",v);
    if(o)fclose(o);
    return 0;
}
