/* CPU mirror of kt_calc_all_sad in
 * src/opencl/ktgmc/kernels/ktgmc_motion.cl (kl_calc_all_sad twin,
 * KTGMC/MVKernel.cu:1122).
 *
 * Pure integer absolute-difference sums, so the CUDA per-thread split and
 * its dev_reduce are order-independent and a scalar loop is exact.  The
 * layout follows the upstream host structs verbatim: `vectors` is a plain
 * row-major short2 array (2 shorts per block), `out` is the packed 12-byte
 * VECTOR {x,y,sad} (3 ints per block).
 *
 * Usage: mv_calc_all_sad_ref <in> <out>
 *   All ints on one line.  Mode:
 *     S nBlkX nBlkY nPad BLK_SIZE NPEL chroma nPitchY nPitchUV
 *       nImgPitchY nImgPitchUV
 *       nV vectors(nV)            (2 shorts per block, as ints)
 *       nY srcY(nY) refY(nY)
 *       nUV srcU(nUV) srcV(nUV) refU(nUV) refV(nUV)   (0 when !chroma)
 *   Output: per block, dst_sad then the 3 VECTOR ints (4 ints per block).
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace std;

static int ref_block_offset(int vx,int vy,int nPitch,int nImgPitch,int NPEL){
    if(NPEL==1) return vx + vy*nPitch;
    if(NPEL==2){
        int sx=vx&1, sy=vy&1, si=sx+sy*2;
        return (vx>>1) + (vy>>1)*nPitch + si*nImgPitch;
    }
    int sx=vx&3, sy=vy&3, si=sx+sy*4;
    return (vx>>2) + (vy>>2)*nPitch + si*nImgPitch;
}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: mv_calc_all_sad_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    char m=(char)rd();
    if(m!='S'){fprintf(stderr,"unknown mode %c\n",m);return 2;}

    int nBlkX=rd(),nBlkY=rd(),nPad=rd(),BLK=rd(),NPEL=rd(),chroma=rd();
    int nPitchY=rd(),nPitchUV=rd(),nImgPitchY=rd(),nImgPitchUV=rd();
    int baseY=rd(),baseUV=rd();
    auto read=[&](int n){ vector<int> a(n); for(int&i:a)i=rd(); return a; };
    int nV=rd(); vector<int> vec=read(nV);
    int nY=rd(); vector<int> srcY=read(nY), refY=read(nY);
    int nUV=rd();
    vector<int> srcU,srcV,refU,refV;
    if(nUV){ srcU=read(nUV); srcV=read(nUV); refU=read(nUV); refV=read(nUV); }

    vector<int> out;
    int blkStep=BLK>>1;
    for(int by=0;by<nBlkY;by++)for(int bx=0;bx<nBlkX;bx++){
        int blk=bx+by*nBlkX;
        int offx=nPad+bx*blkStep, offy=nPad+by*blkStep;
        int vx=vec[blk*2+0], vy=vec[blk*2+1];
        int sad=0;
        int roff=ref_block_offset(vx,vy,nPitchY,nImgPitchY,NPEL);
        for(int jy=0;jy<BLK;jy++)for(int jx=0;jx<BLK;jx++){
            int a=srcY[(size_t)(baseY+(offx+jx)+(offy+jy)*nPitchY)];
            int b=refY[(size_t)(baseY+(offx+jx)+(offy+jy)*nPitchY+roff)];
            int d=a-b; if(d<0)d=-d;
            sad+=d;
        }
        if(chroma){
            int bs2=BLK>>1, bux=offx>>1, buy=offy>>1;
            int roffUV=ref_block_offset(vx>>1,vy>>1,nPitchUV,nImgPitchUV,NPEL);
            for(int jy=0;jy<bs2;jy++)for(int jx=0;jx<bs2;jx++){
                int a=srcU[(size_t)(baseUV+(bux+jx)+(buy+jy)*nPitchUV)];
                int b=refU[(size_t)(baseUV+(bux+jx)+(buy+jy)*nPitchUV+roffUV)];
                int d=a-b; if(d<0)d=-d;
                sad+=d;
            }
            for(int jy=0;jy<bs2;jy++)for(int jx=0;jx<bs2;jx++){
                int a=srcV[(size_t)(baseUV+(bux+jx)+(buy+jy)*nPitchUV)];
                int b=refV[(size_t)(baseUV+(bux+jx)+(buy+jy)*nPitchUV+roffUV)];
                int d=a-b; if(d<0)d=-d;
                sad+=d;
            }
        }
        out.push_back(sad);
        out.push_back(vx); out.push_back(vy); out.push_back(sad);
    }

    FILE* o=fopen(argv[2],"w");
    for(int v:out)fprintf(o,"%d\n",v);
    fclose(o);
    return 0;
}
