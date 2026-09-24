/* CPU mirror of the two provisional MV "prepare" kernels in
 * src/opencl/ktgmc/kernels/ktgmc_degrain_rig.cl
 * (kl_prepare_degrain :1884 / kl_prepare_compensate :2137 twins).
 *
 * These kernels are // RIG-VERIFY: the surrounding host block-geometry model
 * is unsettled (docs/MV_PORT_SPEC.md §6.1).  This mirror + the Python golden
 * pin their per-block ARITHMETIC so it cannot drift while that question is
 * open — the same treatment kf_sharpen gets.
 *
 * Usage: mv_prepare_ref <in> <out>
 *   Modes (all ints on one line):
 *     D nBlkX nBlkY nPad nBlkSize nTh2 thSAD delta binomial NPEL SHIFT
 *       nPitch nPitchSuper nImgPitch
 *       sceneChangeB(delta) sceneChangeF(delta)
 *       isUsableB(delta) isUsableF(delta)
 *       mvB(delta*nBlk*3) mvF(delta*nBlk*3)
 *       -> win_slot(nBlk), src_base(nBlk), WSrc(nBlk),
 *          WB(delta*nBlk), WF(delta*nBlk),
 *          refBaseB(delta*nBlk), refBaseF(delta*nBlk)
 *     C nBlkX nBlkY nPad nBlkSize nTh2 time256 thSAD NPEL SHIFT
 *       nPitchSuper nImgPitch sceneChange mv(nBlk*3)
 *       -> win_slot(nBlk), ref_base(nBlk), ref_sel(nBlk)
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace std;

#define MV_MAX_DELTA 6

static int ref_block_offset(int vx,int vy,int nPitch,int nImgPitch,int NPEL){
    if(NPEL==1) return vx + vy*nPitch;
    if(NPEL==2){
        int sx=vx&1, sy=vy&1;
        return (vx>>1) + (vy>>1)*nPitch + (sx+sy*2)*nImgPitch;
    }
    int sx=vx&3, sy=vy&3;
    return (vx>>2) + (vy>>2)*nPitch + (sx+sy*4)*nImgPitch;
}

static int degrain_weight(int thSAD,int blockSAD){
    if(thSAD<=blockSAD) return 0;
    float a=(float)thSAD*(float)thSAD;
    float b=(float)blockSAD*(float)blockSAD;
    return (int)(256.0f*(a-b)/(a+b));
}

static int norm_weights(int delta,int binomial,int*WRefB,int*WRefF){
    int WSrc=256;
    if(binomial){
        if(delta==1){ WSrc*=2; }
        else if(delta==2){ WSrc*=6; WRefB[0]*=4; WRefF[0]*=4; }
        else if(delta==3){ WSrc*=20; WRefB[0]*=15; WRefF[0]*=15; WRefB[1]*=6; WRefF[1]*=6; }
        else if(delta==4){ WSrc*=70; WRefB[0]*=56; WRefF[0]*=56; WRefB[1]*=28; WRefF[1]*=28; WRefB[2]*=8; WRefF[2]*=8; }
    }
    int WSum=WSrc+1;
    for(int i=0;i<delta;i++) WSum+=WRefB[i]+WRefF[i];
    for(int i=0;i<delta;i++){
        WRefB[i]=WRefB[i]*256/WSum;
        WRefF[i]=WRefF[i]*256/WSum;
    }
    WSrc=256;
    for(int i=0;i<delta;i++) WSrc-=WRefB[i]+WRefF[i];
    return WSrc;
}

static int win_slot(int blkx,int blky,int nBlkX,int nBlkY){
    int wby=((blky+nBlkY-3)/(nBlkY-2))*3;
    int wbx= (blkx+nBlkX-3)/(nBlkX-2);
    return wby+wbx;
}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: mv_prepare_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    auto read=[&](int n){ vector<int> a(n); for(int&i:a)i=rd(); return a; };
    char m=(char)rd();
    vector<int> out;

    if(m=='D'){
        int nBlkX=rd(),nBlkY=rd(),nPad=rd(),nBlkSize=rd();
        int nTh2=rd(),thSAD=rd(),delta=rd(),binomial=rd(),NPEL=rd(),SHIFT=rd();
        int nPitch=rd(),nPitchSuper=rd(),nImgPitch=rd();
        vector<int> scB=read(delta), scF=read(delta);
        vector<int> usB=read(delta), usF=read(delta);
        int nBlk=nBlkX*nBlkY;
        vector<int> mvB=read(delta*nBlk*3), mvF=read(delta*nBlk*3);

        vector<int> ws(nBlk),sb(nBlk),wsrc(nBlk);
        vector<int> wb(delta*nBlk),wf(delta*nBlk),rb(delta*nBlk),rf(delta*nBlk);

        for(int blky=0;blky<nBlkY;blky++)for(int blkx=0;blkx<nBlkX;blkx++){
            int idx=blkx+blky*nBlkX;
            ws[idx]=win_slot(blkx,blky,nBlkX,nBlkY);
            int blkStep=nBlkSize/2;
            int offx=blkx*blkStep, offy=blky*blkStep;
            int offsetS=(nPad+offx)+(nPad+offy)*nPitchSuper;
            sb[idx]=offx+offy*nPitch;

            int WRefB[MV_MAX_DELTA]={0},WRefF[MV_MAX_DELTA]={0};
            for(int i=0;i<delta;i++){
                int k=(i*nBlk+idx)*3;
                int uB = usB[i] && !(scB[i]>nTh2);
                if(uB){
                    rb[i*nBlk+idx]=offsetS+ref_block_offset(mvB[k+0]>>SHIFT,mvB[k+1]>>SHIFT,nPitchSuper,nImgPitch,NPEL);
                    WRefB[i]=degrain_weight(thSAD,mvB[k+2]);
                } else { rb[i*nBlk+idx]=0; WRefB[i]=0; }
                int uF = usF[i] && !(scF[i]>nTh2);
                if(uF){
                    rf[i*nBlk+idx]=offsetS+ref_block_offset(mvF[k+0]>>SHIFT,mvF[k+1]>>SHIFT,nPitchSuper,nImgPitch,NPEL);
                    WRefF[i]=degrain_weight(thSAD,mvF[k+2]);
                } else { rf[i*nBlk+idx]=0; WRefF[i]=0; }
            }
            wsrc[idx]=norm_weights(delta,binomial,WRefB,WRefF);
            for(int i=0;i<delta;i++){ wb[i*nBlk+idx]=WRefB[i]; wf[i*nBlk+idx]=WRefF[i]; }
        }
        for(int v:ws)out.push_back(v);
        for(int v:sb)out.push_back(v);
        for(int v:wsrc)out.push_back(v);
        for(int v:wb)out.push_back(v);
        for(int v:wf)out.push_back(v);
        for(int v:rb)out.push_back(v);
        for(int v:rf)out.push_back(v);
    } else if(m=='C'){
        int nBlkX=rd(),nBlkY=rd(),nPad=rd(),nBlkSize=rd();
        int nTh2=rd(),time256=rd(),thSAD=rd(),NPEL=rd(),SHIFT=rd();
        int nPitchSuper=rd(),nImgPitch=rd();
        int sc=rd();
        int nBlk=nBlkX*nBlkY;
        vector<int> mv=read(nBlk*3);
        vector<int> ws(nBlk),rbase(nBlk),rsel(nBlk);

        for(int blky=0;blky<nBlkY;blky++)for(int blkx=0;blkx<nBlkX;blkx++){
            int idx=blkx+blky*nBlkX;
            if(sc>nTh2){ ws[idx]=-1; rbase[idx]=-1; rsel[idx]=-1; continue; }
            ws[idx]=win_slot(blkx,blky,nBlkX,nBlkY);
            int blkStep=nBlkSize/2;
            int offsetS=(nPad+blkx*blkStep)+(nPad+blky*blkStep)*nPitchSuper;
            int k=idx*3;
            if(mv[k+2]<thSAD){
                int mx=(mv[k+0]*time256/256)>>SHIFT;
                int my=(mv[k+1]*time256/256)>>SHIFT;
                rbase[idx]=offsetS+ref_block_offset(mx,my,nPitchSuper,nImgPitch,NPEL);
                rsel[idx]=1;
            } else {
                rbase[idx]=offsetS+ref_block_offset(0,0,nPitchSuper,nImgPitch,NPEL);
                rsel[idx]=0;
            }
        }
        for(int v:ws)out.push_back(v);
        for(int v:rbase)out.push_back(v);
        for(int v:rsel)out.push_back(v);
    } else { fprintf(stderr,"unknown mode %c\n",m); return 2; }

    FILE* o=fopen(argv[2],"w");
    for(int v:out)fprintf(o,"%d\n",v);
    fclose(o);
    return 0;
}
