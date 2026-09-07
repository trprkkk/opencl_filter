/* CPU mirror of the KFM KEdgeLevel kernels in src/opencl/kfm/kernels/
 * kfm_edgelevel.cl, faithful to the authoritative CPU twins in KFM/KDeband.cu
 * (cpu_edgelevel / cpu_edgelevel_repair / cpu_el_to444 / cpu_el_from444).  For
 * el_to444 the CUDA kernel kl_el_to444 (which this .cl transliterates) differs
 * from cpu_el_to444 only at borders via a border-clamp; we replicate the CUDA
 * kernel (i.e. the .cl) so the mirror validates the transliteration, not the
 * divergent CPU-upconvert routine.
 *
 * Float arithmetic uses float (float32) with no FMA contraction; compile with
 * -ffp-contract=off.  The python golden emulates the same float32 rounding.
 *
 * Usage: kfm_edgelevel_ref <in> <out>
 * Input (space separated ints on one file) starts with a "mode":
 *   mode E  (edgelevel, Y-only or Y+UV):
 *       E width height pitch maxv str strUV thrs check selective uv
 *         nSrc src[...]  nU u[...] nV v[...]     (all planes length = pitch*height)
 *     Output: full dstY plane then (if uv) dstU,dstV (each pitch*height ints).
 *   mode R  (edgelevel_repair):  R width height pitch N el[...] src[...]
 *     Output: full dst plane (interior meaningful; borders read src window OOB).
 *   mode U  (el_to444): U width height srcPitch dstPitch logUVx logUVy src[...]
 *     Output: (BW*width)*(BH*height) dst (row-major, dstPitch used as stride).
 *   mode D  (el_from444): D width height srcPitch dstPitch logUVx logUVy src[...]
 *     Output: width*height dst.
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace std;
static inline int clampI(int v,int a,int b){ return v<a?a:(v>b?b:v); }
static float fclampf(float v,float a,float b){ return v<a?a:(v>b?b:v); }
static int scale(int c,int maxv){ return (int)((((float)c)/255.0f)* (float)maxv); }

static void sort8(int a[8]){
    int t; auto cs=[&](int p,int q){ if(a[p]>a[q]){t=a[p];a[p]=a[q];a[q]=t;} };
    cs(0,1);cs(2,3);cs(4,5);cs(6,7);
    cs(0,2);cs(1,3);cs(4,6);cs(5,7);
    cs(1,2);cs(5,6);
    cs(0,4);cs(1,5);cs(2,6);cs(3,7);
    cs(2,4);cs(3,5);
    cs(1,2);cs(3,4);cs(5,6);
}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: kfm_edgelevel_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    char mode; { int m=rd(); mode=(char)m; }
    if(mode=='E'){
        int width=rd(),height=rd(),pitch=rd(),maxv=rd(),str=rd(),strUV=rd(),thrs=rd();
        int check=rd(),selective=rd(),uv=rd();
        int nSrc=rd(),nU=rd(),nV=rd();
        vector<int> src(nSrc),u(nU),v(nV);
        for(int&i:src)i=rd(); for(int&i:u)i=rd(); for(int&i:v)i=rd();
        vector<int> dY((size_t)pitch*height),dU,dV;
        if(uv){ dU.assign((size_t)pitch*height,0); dV.assign((size_t)pitch*height,0); }
        int S=selective?1:0;
        for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++){
            int off=yy*pitch+xx;
            if(yy<=(1+S)||yy>=height-(2+S)||xx<=(1+S)||xx>=width-(2+S)){
                dY[off]= check? scale(16,maxv): src[off];
                if(uv){ dU[off]=u[off]; dV[off]=v[off]; }
                continue;
            }
            int hmax,hmin,vmax,vmin,hdiffmax=0,vdiffmax=0;
            int hprev=hmax=hmin=src[off-(2+S)];
            int vprev=vmax=vmin=src[off-(2+S)*pitch];
            for(int i=-(1+S);i<(3+S);i++){
                int hc=src[off+i], vc=src[off+i*pitch];
                if(hc>hmax)hmax=hc; if(hc<hmin)hmin=hc;
                if(vc>vmax)vmax=vc; if(vc<vmin)vmin=vc;
                if(selective){ int dh=hc-hprev; if(dh<0)dh=-dh; int dv=vc-vprev; if(dv<0)dv=-dv; if(dh>hdiffmax)hdiffmax=dh; if(dv>vdiffmax)vdiffmax=dv; }
                hprev=hc; vprev=vc;
            }
            if(hmax-hmin<vmax-vmin){hmax=vmax;hmin=vmin;}
            if(vdiffmax>hdiffmax)hdiffmax=vdiffmax;
            float factor=1.0f;
            if(selective){
                float rdiff=(float)hdiffmax/(float)(hmax-hmin);
                float a=fclampf((0.55f-rdiff)*10.0f,0.0f,1.0f);
                float b=fclampf((0.35f-rdiff)*10.0f,0.0f,1.0f);
                factor=a-b;
            }
            int srcvY=src[off];
            int dstvY,dstvU=0,dstvV=0;
            if(check){
                if(hmax-hmin>thrs && factor>0.0f){
                    int avgY=(hmax+hmin)>>1;
                    if(srcvY>avgY) dstvY=(factor==1.0f)?scale(50,maxv):scale(120,maxv);
                    else dstvY=(factor==1.0f)?scale(240,maxv):scale(180,maxv);
                } else dstvY=scale(16,maxv);
                if(uv){dstvU=u[off];dstvV=v[off];}
            } else {
                if(hmax-hmin>thrs && factor>0.0f){
                    float factorY=((float)str*factor)*0.0625f;
                    int avgY=(hmax+hmin)>>1;
                    int yv=srcvY+(int)(((float)(srcvY-avgY))*factorY);
                    yv=clampI(yv,hmin,hmax); yv=clampI(yv,0,maxv);
                    dstvY=yv;
                    if(uv){
                        float factorUV=(float)strUV*0.0625f;
                        // U
                        int Uhmax,Uhmin,Uvmax,Uvmin;
                        Uhmax=Uhmin=u[off-(2+S)]; Uvmax=Uvmin=u[off-(2+S)*pitch];
                        for(int i=-(1+S);i<(3+S);i++){
                            int hc=u[off+i],vc=u[off+i*pitch];
                            if(hc>Uhmax)Uhmax=hc;if(hc<Uhmin)Uhmin=hc;
                            if(vc>Uvmax)Uvmax=vc;if(vc<Uvmin)Uvmin=vc;
                        }
                        if(Uhmax-Uhmin<Uvmax-Uvmin){Uhmax=Uvmax;Uhmin=Uvmin;}
                        int avgU=(Uhmax+Uhmin)>>1, svU=u[off];
                        int uout=svU+(int)(((float)(svU-avgU))*factorUV);
                        uout=clampI(uout,Uhmin,Uhmax); uout=clampI(uout,0,maxv);
                        dstvU=uout;
                        int Vhmax,Vhmin,Vvmax,Vvmin;
                        Vhmax=Vhmin=v[off-(2+S)]; Vvmax=Vvmin=v[off-(2+S)*pitch];
                        for(int i=-(1+S);i<(3+S);i++){
                            int hc=v[off+i],vc=v[off+i*pitch];
                            if(hc>Vhmax)Vhmax=hc;if(hc<Vhmin)Vhmin=hc;
                            if(vc>Vvmax)Vvmax=vc;if(vc<Vvmin)Vvmin=vc;
                        }
                        if(Vhmax-Vhmin<Vvmax-Vvmin){Vhmax=Vvmax;Vhmin=Vvmin;}
                        int avgV=(Vhmax+Vhmin)>>1, svV=v[off];
                        int vout=svV+(int)(((float)(svV-avgV))*factorUV);
                        vout=clampI(vout,Vhmin,Vhmax); vout=clampI(vout,0,maxv);
                        dstvV=vout;
                    }
                } else { dstvY=srcvY; if(uv){dstvU=u[off];dstvV=v[off];} }
            }
            dY[off]=dstvY;
            if(uv){dU[off]=dstvU; dV[off]=dstvV;}
        }
        FILE* o=fopen(argv[2],"w");
        for(int v:dY)fprintf(o,"%d\n",v);
        if(uv){for(int v:dU)fprintf(o,"%d\n",v);for(int v:dV)fprintf(o,"%d\n",v);}
        fclose(o);
    } else if(mode=='R'){
        int width=rd(),height=rd(),pitch=rd(),N=rd();
        int nE=rd(),nS=rd();
        vector<int> el(nE),src(nS);
        for(int&i:el)i=rd(); for(int&i:src)i=rd();
        vector<int> dst((size_t)pitch*height);
        // borders kept = src (unused; kernel is rig-bound there, needs padding)
        for(int i=0;i<(int)dst.size();i++) dst[i]=src[i];
        for(int yy=1;yy<height-1;yy++)for(int xx=1;xx<width-1;xx++){
            int off=yy*pitch+xx;
            int srcv=src[off], elv=el[off], dstv=srcv;
            if(elv!=srcv){
                int a[8];
                a[0]=src[(xx-1)+(yy-1)*pitch]; a[1]=src[xx+(yy-1)*pitch]; a[2]=src[(xx+1)+(yy-1)*pitch];
                a[3]=src[(xx-1)+yy*pitch];     a[4]=src[(xx+1)+yy*pitch];
                a[5]=src[(xx-1)+(yy+1)*pitch]; a[6]=src[xx+(yy+1)*pitch]; a[7]=src[(xx+1)+(yy+1)*pitch];
                sort8(a);
                int lo=srcv<a[N-1]?srcv:a[N-1];
                int hi=srcv>a[8-N]?srcv:a[8-N];
                dstv=clampI(elv,lo,hi);
            }
            dst[off]=dstv;
        }
        FILE* o=fopen(argv[2],"w");
        for(int v:dst)fprintf(o,"%d\n",v);
        fclose(o);
    } else if(mode=='U'){
        int width=rd(),height=rd(),sp=rd(),dp=rd(),lx=rd(),ly=rd();
        int nS=rd(); vector<int> src(nS); for(int&i:src)i=rd();
        int BW=1<<lx, BH=1<<ly;
        int DW=BW*width, DH=BH*height;
        vector<int> dst((size_t)dp*DH,0);
        for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++){
            int v00=src[xx+yy*sp];
            dst[BW*xx+0+(BH*yy+0)*dp]=v00;
            if(lx){
                int v10=(xx+1<width)?src[(xx+1)+yy*sp]:v00;
                dst[BW*xx+1+(BH*yy+0)*dp]=(v00+v10+1)>>1;
                if(ly){
                    int v01,v11;
                    if(yy+1<height){ v01=src[xx+(yy+1)*sp]; v11=(xx+1<width)?src[(xx+1)+(yy+1)*sp]:v01; }
                    else { v01=v00; v11=(xx+1<width)?v10:v00; }
                    dst[BW*xx+0+(BH*yy+1)*dp]=(v00+v01+1)>>1;
                    dst[BW*xx+1+(BH*yy+1)*dp]=(v00+v10+v01+v11+2)>>2;
                }
            }
        }
        FILE* o=fopen(argv[2],"w");
        // output only the DW x DH logical region (row-major over DW)
        for(int yy=0;yy<DH;yy++)for(int xx=0;xx<DW;xx++) fprintf(o,"%d\n",dst[yy*dp+xx]);
        fclose(o);
    } else if(mode=='D'){
        int width=rd(),height=rd(),sp=rd(),dp=rd(),lx=rd(),ly=rd();
        int nS=rd(); vector<int> src(nS); for(int&i:src)i=rd();
        int BW=1<<lx, BH=1<<ly;
        vector<int> dst((size_t)width*height);
        for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++){
            dst[xx+yy*width]=src[BW*xx+BH*yy*sp];
        }
        FILE* o=fopen(argv[2],"w");
        for(int v:dst)fprintf(o,"%d\n",v);
        fclose(o);
    }
    return 0;
}
