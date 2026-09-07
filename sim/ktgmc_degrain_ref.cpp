/* CPU mirror of the KMDegrain/KMCompensate OVERLAP pixel combiner, faithful to
 * the MV.cpp CPU path (KMDegrainCore::Proc overlap branch: per block run the
 * Degrain1to6_C weighted denoise into a tmpBlock, then Overlaps_C feathered
 * window accumulation into a global tmp plane, then Short2Bytes).  It checks
 * the two .cl kernels (kt_degrain_patch + kt_overlap_out in
 * src/opencl/ktgmc/kernels/ktgmc_motion.cl), which compute the SAME result
 * per output pixel by summing the covering blocks' window terms.
 *
 * Input (space-separated ints on one file):
 *   nBlkX nBlkY nBlkSize stepX stepY overlapX overlapY delta
 *   width height maxv shift
 *   src_pitch refF_pitch refB_pitch dst_pitch win_size     (win_size=nBlkSize^2)
 *   src_len refF_len refB_len
 *   WSrcArr[nBlk]  WF[delta*nBlk]  WB[delta*nBlk]
 *   refBaseF[delta*nBlk]  refBaseB[delta*nBlk]
 *   winBase[9*win_size]  refFPlane[refF_len]  refBPlane[refB_len]  src[src_len]
 * Output: the full dst plane (width*height ints, row-major).
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace std;
int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: ktgmc_degrain_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    int nBlkX=rd(),nBlkY=rd(),nBlkSize=rd(),stepX=rd(),stepY=rd();
    int overlapX=rd(),overlapY=rd(),delta=rd();
    int width=rd(),height=rd(),maxv=rd(),shift=rd();
    int srcPitch=rd(),refFPitch=rd(),refBPitch=rd(),dstPitch=rd(),winSize=rd();
    int srcLen=rd(),refFLen=rd(),refBLen=rd();
    int nBlk=nBlkX*nBlkY;
    vector<int> WSrc(nBlk),WF(delta*nBlk),WB(delta*nBlk);
    for(int&i:WSrc)i=rd(); for(int&i:WF)i=rd(); for(int&i:WB)i=rd();
    vector<int> baseF(delta*nBlk),baseB(delta*nBlk);
    for(int&i:baseF)i=rd(); for(int&i:baseB)i=rd();
    vector<int> win(9*winSize); for(int&i:win)i=rd();
    vector<int> refF(refFLen),refB(refBLen),src(srcLen);
    for(int&i:refF)i=rd(); for(int&i:refB)i=rd(); for(int&i:src)i=rd();

    int W_B=nBlkX*stepX+overlapX, H_B=nBlkY*stepY+overlapY;
    // global tmp accumulation plane (pitch = width), matching CPU tmpDst
    vector<int> tmp((size_t)width*height,0);
    auto deg = [&](int blk,int u,int v)->int{
        int bx=blk%nBlkX, by=blk/nBlkX;
        int val = src[(by*stepY+v)*srcPitch + (bx*stepX+u)] * WSrc[blk];
        for(int k=0;k<delta;k++){
            int o=k*nBlk+blk;
            val += refB[baseB[o]+u+v*refBPitch] * WB[o];
            val += refF[baseF[o]+u+v*refFPitch] * WF[o];
        }
        return (val + (shift==11?0:128)) >> 8;   // 16-bit: no round
    };
    for(int by=0;by<nBlkY;by++){
        int wby=3*((by+nBlkY-3)/(nBlkY-2));
        for(int bx=0;bx<nBlkX;bx++){
            int blk=bx+by*nBlkX;
            int wbx=(bx+nBlkX-3)/(nBlkX-2);
            const int* W=&win[(wby+wbx)*winSize];
            int dstCol=bx*stepX;
            for(int v=0;v<nBlkSize;v++){
                for(int u=0;u<nBlkSize;u++){
                    int dg=deg(blk,u,v);
                    int term = (shift==11) ? dg*W[v*nBlkSize+u]
                                           : (dg*W[v*nBlkSize+u]+256)>>6;
                    tmp[(by*stepY+v)*width + (dstCol+u)] += term;
                }
            }
        }
    }
    // Short2Bytes over the block-covered region + source-copy edges
    vector<int> out((size_t)width*height);
    for(int y=0;y<height;y++){
        for(int x=0;x<width;x++){
            int val;
            if(x<W_B && y<H_B){
                int a=tmp[y*width+x]>>shift; val=a>maxv?maxv:a; if(val<0)val=0;
            } else {
                val=src[y*srcPitch+x];
            }
            out[y*width+x]=val;
        }
    }
    FILE* o=fopen(argv[2],"w");
    for(int v:out) fprintf(o,"%d\n",v);
    if(o)fclose(o);
    return 0;
}
