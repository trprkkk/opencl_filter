/* CPU mirror of the KTGMC block-search E2E (kt_search_column in
 * src/opencl/ktgmc/kernels/ktgmc_search.cl + the host-side ordered dispatch
 * of SEARCH_HOST_CONTRACT.md §4).  Adapted from the CORRECTED scalar
 * reference model in cuda_search/cuda_search_host.cu (scalar_search /
 * refine), which the CUDA twin validated bit-exact on 9 configs
 * (cuda_search/run_s2.txt: order0/order1 mism=0 everywhere).
 *
 * The mirror implements the full host assembly of one Search() call:
 *   1. kl_prepare_search host-equivalent (ANALYZE_SYNC=1: left predictor =
 *      blkIdx-1, OWN region, read fresh; up = blkIdx-nBlkX own region;
 *      bottom-right = blkIdx+nBlkX+1+nBlkCount COPY region; sentinels
 *      slot -2 = (0,0), slot -1 = globalMV*nPel as short2);
 *   2. the copy-region snapshot (vectors_copy[blkIdx] = vectors[blkIdx]);
 *   3. the ordered search, column-major ascending (order0: bx outer, by
 *      inner per column), which is the dispatch order the OpenCL port uses;
 *   4. kl_calc_all_sad (scalar SAD of the final MV).
 *
 * Conditional-refine / shared-centre semantics are the stage-1 corrected
 * ones (SEARCH_MEASURE_REPORT.md §3.6): refine updates are conditional
 * (entry best survives unless strictly beaten), EXHAUSTIVE expand1/expand2
 * share the pre-expand1 centre, HEX chains hex2 -> expand1.
 *
 * Usage: ktgmc_search_ref <in> <out>  (all ints, whitespace separated)
 * Input:
 *   S nBlkX nBlkY nBlkSize SEARCH NPEL chroma nPad nBlkSizeOvr nLogScale
 *     nExtendedWidth nExtendedHeight nLambdaLevel lsad
 *     penaltyZero penaltyGlobal penaltyNew
 *   gx gy                      (global MV, pre-scaling)
 *   batch
 *   per batch: own[2*n] copy[2*n] sads[n]        (x,y pairs as ints)
 *   nSrcY srcY... nRefY refY...
 *   nUV [srcU srcV refU refV...]                (nUV==0 when !chroma)
 *   nPitchY nPitchUV nImgPitchY nImgPitchUV
 * Output (per batch, per block, row-major): x y sad   (3 ints per block)
 */
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <vector>
#include <algorithm>
using namespace std;

struct CR { int cost; int x, y; };

static int ref_block_offset(int vx,int vy,int nPitch,int nImgPitch,int NPEL){
    if(NPEL==1) return vx + vy*nPitch;
    if(NPEL==2){
        int sx=vx&1, sy=vy&1, si=sx+sy*2;
        return (vx>>1) + (vy>>1)*nPitch + si*nImgPitch;
    }
    int sx=vx&3, sy=vy&3, si=sx+sy*4;
    return (vx>>2) + (vy>>2)*nPitch + si*nImgPitch;
}
static int sad_of(const vector<int>& srcY,const vector<int>& refY,
                  const vector<int>* srcU,const vector<int>* refU,
                  const vector<int>* srcV,const vector<int>* refV,
                  int offx,int offy,int BS,int chroma,int nPitchY,int nPitchUV,
                  int nImgPitchY,int nImgPitchUV,int NPEL,int vx,int vy){
    int roff = ref_block_offset(vx,vy,nPitchY,nImgPitchY,NPEL);
    int sad = 0;
    for(int jy=0;jy<BS;jy++)for(int jx=0;jx<BS;jx++){
        int a = srcY[(size_t)(offx+jx)+(offy+jy)*nPitchY];
        int b = refY [(size_t)(offx+jx)+(offy+jy)*nPitchY+roff];
        int d = a-b; if(d<0)d=-d; sad+=d;
    }
    if(chroma){
        int bs2=BS/2;
        int bux=offx>>1, buy=offy>>1;
        int roffUV = ref_block_offset(vx>>1,vy>>1,nPitchUV,nImgPitchUV,NPEL);
        for(int jy=0;jy<bs2;jy++)for(int jx=0;jx<bs2;jx++){
            int a = (*srcU)[(size_t)(bux+jx)+(buy+jy)*nPitchUV];
            int b = (*refU)[(size_t)(bux+jx)+(buy+jy)*nPitchUV+roffUV];
            int d = a-b; if(d<0)d=-d; sad+=d;
            a = (*srcV)[(size_t)(bux+jx)+(buy+jy)*nPitchUV];
            b = (*refV)[(size_t)(bux+jx)+(buy+jy)*nPitchUV+roffUV];
            d = a-b; if(d<0)d=-d; sad+=d;
        }
    }
    return sad;
}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: ktgmc_search_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; long long x;
    while(f&&fscanf(f,"%lld",&x)==1) t.push_back((int)x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};

    if((char)rd()!='S'){fprintf(stderr,"bad mode\n");return 2;}
    int nBlkX=rd(),nBlkY=rd(),BS=rd(),SEARCH=rd(),NPEL=rd(),chroma=rd();
    int nPad=rd(),nBlkSizeOvr=rd(),nLogScale=rd();
    int nExtW=rd(),nExtH=rd(),nLambdaLevel=rd(),lsad=rd();
    int pZero=rd(),pGlobal=rd(),pNew=rd();
    int gx=rd(),gy=rd();
    int batch=rd();
    int n=nBlkX*nBlkY;

    vector<int> own((size_t)batch*2*n), copy((size_t)batch*2*n), sads((size_t)batch*n);
    for(size_t i=0;i<own.size();i++) own[i]=rd();
    for(size_t i=0;i<copy.size();i++) copy[i]=rd();
    for(size_t i=0;i<sads.size();i++) sads[i]=rd();

    int nSrcY=rd(); vector<int> srcY(nSrcY); for(int&i:srcY)i=rd();
    int nRefY=rd(); vector<int> refY(nRefY); for(int&i:refY)i=rd();
    int nUV=rd();
    vector<int> srcU,srcV,refU,refV;
    if(nUV){
        srcU.resize(nUV); for(int&i:srcU)i=rd();
        srcV.resize(nUV); for(int&i:srcV)i=rd();
        refU.resize(nUV); for(int&i:refU)i=rd();
        refV.resize(nUV); for(int&i:refV)i=rd();
    }
    int nPitchY=rd(),nPitchUV=rd(),nImgPitchY=rd(),nImgPitchUV=rd();

    /* ---- sentinels (kl_init_const_vec; global sentinel as short2) ---- */
    int zeroX=0, zeroY=0;
    short gsx=(short)(gx*NPEL), gsy=(short)(gy*NPEL);

    /* ---- prepare host-equivalent (ANALYZE_SYNC=1) ---- */
    vector<int> data((size_t)batch*n*12), dataf((size_t)batch*n*5);
    int nPaddingScaled = nPad >> nLogScale;
    for(int b=0;b<batch;b++)for(int by=0;by<nBlkY;by++)for(int bx=0;bx<nBlkX;bx++){
        int blkIdx=bx+by*nBlkX; size_t o=((size_t)b*n+blkIdx)*12, fo=o/12*5;
        int X=nPad+nBlkSizeOvr*bx, Y=nPad+nBlkSizeOvr*by;
        data[o+0]=NPEL*(nExtW -X-BS-nPad+nPaddingScaled)-1;
        data[o+1]=NPEL*(nExtH-Y-BS-nPad+nPaddingScaled)-1;
        data[o+2]=-NPEL*(X-nPad+nPaddingScaled);
        data[o+3]=-NPEL*(Y-nPad+nPaddingScaled);
        int p1=-2;
        if(bx>0) p1=blkIdx-1;                 /* ANALYZE_SYNC=1: own, FRESH */
        int p2=(by>0)? blkIdx-nBlkX : p1;
        int p3=-2;
        if((by<nBlkY-1)&&(bx<nBlkX-1)) p3=blkIdx+nBlkX+1+nBlkX*nBlkY;
        data[o+4]=-2; data[o+5]=-1; data[o+6]=blkIdx;
        data[o+7]=p1; data[o+8]=p2; data[o+9]=p3;
        data[o+10]=own[(size_t)b*2*n+2*blkIdx+0];   /* PRED_X/Y = incoming */
        data[o+11]=own[(size_t)b*2*n+2*blkIdx+1];
        dataf[fo+0]=pZero; dataf[fo+1]=pGlobal; dataf[fo+2]=0; dataf[fo+3]=pNew;
        int sad=sads[(size_t)b*n+blkIdx];
        int lambda=nLambdaLevel*lsad/(lsad+(sad>>1))*lsad/(lsad+(sad>>1));
        if(by==0) lambda=0;
        dataf[fo+4]=lambda;
    }

    /* ---- ordered search (order0: column-major ascending) ---- */
    vector<int> out;
    for(int b=0;b<batch;b++){
        /* flat working vectors: [0..n) own (fresh), [n..2n) copy (stale);
         * the search reads are resolved against this array plus the
         * sentinels, exactly like the kernel's vrow[] addressing */
        vector<int> vx(2*n), vy(2*n);
        for(int i=0;i<2*n;i++){ vx[i]=own[(size_t)b*2*n+2*i]; vy[i]=own[(size_t)b*2*n+2*i+1]; }
        for(int i=0;i<n;i++){ vx[n+i]=copy[(size_t)b*2*n+2*i]; vy[n+i]=copy[(size_t)b*2*n+2*i+1]; }

        for(int blkx=0;blkx<nBlkX;blkx++)for(int blky=0;blky<nBlkY;blky++){
            int blkIdx=blky*nBlkX+blkx;
            const int* D=&data[((size_t)b*n+blkIdx)*12];
            const int* F=&dataf[((size_t)b*n+blkIdx)*5];
            int offx=nPad+blkx*(BS/2), offy=nPad+blky*(BS/2);

            auto resolve=[&](int idx,int&x,int&y){
                if(idx==-2){x=zeroX;y=zeroY;}
                else if(idx==-1){x=gsx;y=gsy;}
                else {x=vx[idx];y=vy[idx];}
            };
            auto clip=[&](int&x,int&y){
                x = (x>D[0])?D[0] : ((x<D[2])?D[2]:x);
                y = (y>D[1])?D[1] : ((y<D[3])?D[3]:y);
            };

            /* 7 predictors (CPU_EMU placement) */
            int px[7],py[7];
            for(int k=0;k<6;k++){
                int tx,ty; resolve(D[4+k],tx,ty); clip(tx,ty);
                int dst=(k<3)?k:k+1;
                px[dst]=tx; py[dst]=ty;
            }
            px[3]=min(max(min(px[4],px[5]),px[6]),max(px[4],px[5]));
            py[3]=min(max(min(py[4],py[5]),py[6]),max(py[4],py[5]));

            CR r[7];
            for(int i=0;i<7;i++){
                r[i].x=px[i]; r[i].y=py[i];
                r[i].cost=(F[4]*((px[i]-D[10])*(px[i]-D[10])
                                +(py[i]-D[11])*(py[i]-D[11])))>>8;
            }
            for(int i=0;i<7;i++){
                int sad=sad_of(srcY,refY,
                               chroma?&srcU:nullptr,chroma?&refU:nullptr,
                               chroma?&srcV:nullptr,chroma?&refV:nullptr,
                               offx,offy,BS,chroma,nPitchY,nPitchUV,
                               nImgPitchY,nImgPitchUV,NPEL,r[i].x,r[i].y);
                if(i<3) r[i].cost=sad+((sad*F[i])>>8);
                else    r[i].cost+=sad;
            }
            for(int i=1;i<7;i++) if(r[0].cost>r[i].cost) r[0]=r[i];
            CR best=r[0];

            /* refinement patterns (verbatim_search.inc constant order) */
            static const int e1x[8]={0,0,-1,1,-1,-1,1,1};
            static const int e1y[8]={-1,1,0,0,-1,1,-1,1};
            static const int e2x[16]={-1,-1,0,0,1,1,-2,2,-2,2,-2,2,-2,-2,2,2};
            static const int e2y[16]={-2,2,-2,2,-2,2,-1,-1,0,0,1,1,-2,2,-2,2};
            static const int hx[6]={-2,-1,1,2,1,-1};
            static const int hy[6]={0,2,2,0,-2,-2};

            /* one refine stage: prune vs entry best; survivors get
             * sad + (sad*penaltyNew)>>8; conditional final update */
            auto refine=[&](int cx,int cy,const int* adx,const int* ady,int nC){
                vector<CR> rr(nC);
                for(int i=0;i<nC;i++){
                    int X=cx+adx[i],Y=cy+ady[i];
                    bool ok=(X<=D[0])&&(Y<=D[1])&&(X>=D[2])&&(Y>=D[3]);
                    int c=(F[4]*((X-D[10])*(X-D[10])+(Y-D[11])*(Y-D[11])))>>8;
                    if(c>=best.cost) ok=false;
                    rr[i].x=X; rr[i].y=Y;
                    rr[i].cost=ok?c:INT_MAX;
                }
                for(int i=0;i<nC;i++){
                    if(rr[i].cost==INT_MAX) continue;
                    int sad=sad_of(srcY,refY,
                                   chroma?&srcU:nullptr,chroma?&refU:nullptr,
                                   chroma?&srcV:nullptr,chroma?&refV:nullptr,
                                   offx,offy,BS,chroma,nPitchY,nPitchUV,
                                   nImgPitchY,nImgPitchUV,NPEL,rr[i].x,rr[i].y);
                    rr[i].cost+=sad+((sad*F[3])>>8);
                }
                for(int i=1;i<nC;i++) if(rr[0].cost>rr[i].cost) rr[0]=rr[i];
                if(rr[0].cost<best.cost) best=rr[0];
            };

            if(SEARCH==1){
                int bmx=best.x,bmy=best.y;         /* snapshot (lines 814-815) */
                refine(bmx,bmy,e1x,e1y,8);
                refine(bmx,bmy,e2x,e2y,16);        /* SAME pre-expand1 centre */
            }else{
                refine(best.x,best.y,hx,hy,6);
                refine(best.x,best.y,e1x,e1y,8);   /* chained off hex2 result */
            }
            vx[blkIdx]=best.x; vy[blkIdx]=best.y;
        }

        /* kl_calc_all_sad: SAD of the final MV */
        for(int i=0;i<n;i++){
            int bx=i%nBlkX,by=i/nBlkX;
            int offx=nPad+bx*(BS/2),offy=nPad+by*(BS/2);
            int sad=sad_of(srcY,refY,
                           chroma?&srcU:nullptr,chroma?&refU:nullptr,
                           chroma?&srcV:nullptr,chroma?&refV:nullptr,
                           offx,offy,BS,chroma,nPitchY,nPitchUV,
                           nImgPitchY,nImgPitchUV,NPEL,vx[i],vy[i]);
            out.push_back(vx[i]); out.push_back(vy[i]); out.push_back(sad);
        }
    }

    FILE* o=fopen(argv[2],"w");
    for(int v:out)fprintf(o,"%d\n",v);
    fclose(o);
    return 0;
}
