/* CPU mirror of kl_prepare_search (kt_prepare_search in
 * src/opencl/ktgmc/kernels/ktgmc_motion.cl). Independent re-implementation
 * for cross-checking against the Python golden. ANALYZE_SYNC == 1.
 *
 * Usage: ktgmc_searchprep_ref <infile> <outfile>
 * Input file (space-separated ints):
 *   nBlkX nBlkY nBlkSize nLogScale nLambdaLevel lsad
 *   penaltyZero penaltyGlobal penaltyNew
 *   nPel nPad nBlkSizeOvr nExtendedWidth nExtendedHeight
 *   then (nBlkX*nBlkY) triplets "vx vy sad"  (row-major per-block vector+sad)
 * Output file: for each block (row-major): "d0..d11 f0..f4" (17 ints).
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace std;

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: searchprep <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> n; int x;
    while(f&&fscanf(f,"%d",&x)==1) n.push_back(x);
    if(f)fclose(f);
    int p=0;
    int nBlkX=n[p++],nBlkY=n[p++],nBlkSize=n[p++],nLogScale=n[p++],
        nLambdaLevel=n[p++],lsad=n[p++],penaltyZero=n[p++],
        penaltyGlobal=n[p++],penaltyNew=n[p++],nPel=n[p++],nPad=n[p++],
        nBlkSizeOvr=n[p++],nExtendedWidth=n[p++],nExtendedHeight=n[p++];
    int nBlk=nBlkX*nBlkY;
    vector<int> vx(nBlk),vy(nBlk),sad(nBlk);
    for(int i=0;i<nBlk;i++){vx[i]=n[p++];vy[i]=n[p++];sad[i]=n[p++];}
    FILE* o=fopen(argv[2],"w");
    int nPaddingScaled=nPad>>nLogScale;
    for(int by=0;by<nBlkY;by++)for(int bx=0;bx<nBlkX;bx++){
        int blkIdx=bx+by*nBlkX;
        int X=nPad+nBlkSizeOvr*bx, Y=nPad+nBlkSizeOvr*by;
        int nDxMax=nPel*(nExtendedWidth -X-nBlkSize-nPad+nPaddingScaled)-1;
        int nDyMax=nPel*(nExtendedHeight-Y-nBlkSize-nPad+nPaddingScaled)-1;
        int nDxMin=-nPel*(X-nPad+nPaddingScaled);
        int nDyMin=-nPel*(Y-nPad+nPaddingScaled);
        int p1=-2; if(bx>0) p1=blkIdx-1+nBlkX*nBlkY;
        int pp2=-2; if(by>0) pp2=blkIdx-nBlkX; else pp2=p1;
        int p3=-2; if((by<nBlkY-1)&&(bx<nBlkX-1)) p3=blkIdx+nBlkX+1+nBlkX*nBlkY;
        int d[12]; d[0]=nDxMax;d[1]=nDyMax;d[2]=nDxMin;d[3]=nDyMin;
        d[4]=-2;d[5]=-1;d[6]=blkIdx;d[7]=p1;d[8]=pp2;d[9]=p3;
        d[10]=vx[blkIdx];d[11]=vy[blkIdx];
        int f[5]; f[0]=penaltyZero;f[1]=penaltyGlobal;f[2]=0;f[3]=penaltyNew;
        int lambda=nLambdaLevel*lsad/(lsad+(sad[blkIdx]>>1))
                  *lsad/(lsad+(sad[blkIdx]>>1));
        if(by==0) lambda=0;
        f[4]=lambda;
        fprintf(o,"%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
            d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],
            f[0],f[1],f[2],f[3],f[4]);
    }
    if(o)fclose(o);
    return 0;
}
