/* CPU mirror of the CombingAnalyze.cu batch-1 kernels in
 * src/opencl/kfm/kernels/kfm_combinganalyze.cl.  All twins are exact upstream
 * twins (bilinear's extra env arg is unused; sum_box3x3's src is const here
 * though non-const upstream — never written).
 *
 * Usage: kfm_combinganalyze_ref <in> <out>
 *   All ints on one line.  Modes:
 *     F copy_first   : F width height spitch dpitch lanes  nS  src(nS)
 *                      dst = src[(x+y*spitch)*lanes] (.x lane extract).
 *                      -> output width*height
 *     G combe_to_flag: G nBlkX nBlkY fpitch cpitch  nC  combe(nC)
 *                      2x2 round-half-up quarter mean.
 *                      -> output nBlkX*nBlkY
 *     B sum_box3x3   : B width height pitch maxv  nS  src(nS)
 *                      src halo-padded (nS = pitch*(height+2), origin at
 *                      1+pitch); dst = min((3x3 sum)>>2, maxv).
 *                      -> output width*height
 *     N binary_flag  : N nBlkX nBlkY pitch thY thC  nP  srcY(nP) srcC(nP)
 *                      in-place (on a copy of srcY, as upstream):
 *                      (Y>=thY||C>=thC)?128:0.
 *                      -> output nBlkX*nBlkY
 *     H bilinear_h   : H width height spitch scale shift  nS  src(nS)
 *                      src has a 1-col halo each side (origin col 1);
 *                      (s0*c0+s1*c1+HALF)>>SHIFT over mapped cols.
 *                      -> output width*height
 *     V bilinear_v   : V width height spitch scale shift  nS  src(nS)
 *                      src has a 1-row halo above (origin row 1; rows =
 *                      1+readrows); same blend over mapped rows.
 *                      -> output width*height
 *     S soften       : S width height pitch  nP  s0(nP) s1(nP) s2(nP)
 *                      float32 (int)(((f0+f1)+f2)*(1.0f/3.0f)), mod 256.
 *                      -> output width*height
 *     R remove_combe2: R width height pitch cpitch thcombe  nS nC
 *                      src(nS) combe(nC); src padded (nS = pitch*(height+2),
 *                      origin at row 1); combe = .x lane plane; 4x4 blocks
 *                      scoring >= thcombe get the vertical binomial.
 *                      -> output width*height
 *     C clean_super  : C width height pitch thresh  nP
 *                      prevx(nP) prevy(nP) curx(nP) cury(nP); v=cur,
 *                      zero .x where prev.y<=th && cur.y<=th.
 *                      -> output .x plane then .y plane
 *     D durty_block  : D width height pitch  nP  flagp(nP)
 *                      work = 0 then OR-scan (init folded in).
 *                      -> output single 0/1
 *     M combe8       : M L0 L1 L2 L3 L4 L5 L6 L7 (8 taps)
 *                      -> output single int
 *     A diff8        : A L00 L10 L01 L11 L02 L12 L03 L13 (8 taps)
 *                      -> output single int
 *     J super_analyze: J nBlkX nBlkY fpitch_f fpitch shift parity  nF nFl
 *                      f0(nF) f1(nF); f dims 4*nBlk (nF = fpitch_f*4*nBlkY);
 *                      flag bufs nFl = fpitch*2*nBlkY, split .x/.y; cells
 *                      with bx==nBlkX-1 or by==nBlkY-1 write nothing.
 *                      -> output flag_x buf then flag_y buf (-1 unwritten)
 *     W count        : W width height pitch parity thM thS thLS
 *                      m0 s0 l0 m1 s1 l1 (init FMCount[2])  nP
 *                      c0x(nP) c0y(nP) c1x(nP) c1y(nP); serial accumulate
 *                      (== block-reduce + atomics, ints are order-exact).
 *                      -> output 6 ints (slot0 move/shima/lshima, slot1)
 *     Q count_2planes: Q width height pitch parity thM thS thLS
 *                      m0 s0 l0 m1 s1 l1  nP  U c0x c0y c1x c1y V c0x c0y
 *                      c1x c1y (8 planes); fused U+V loop (the golden runs
 *                      two single-plane passes instead — see the runner).
 *                      -> output 6 ints
 *     I init_fmcount : I (no payload; zero-fill check)
 *                      -> output 6 zeros
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace std;

static int AbsDiff(int a,int b){ int d=a-b; if(d<0)d=-d; return d; }

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: kfm_combinganalyze_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    char m=(char)rd();
    vector<int> out;
    auto read=[&](int n){ vector<int> a(n); for(int&i:a)i=rd(); return a; };

    if(m=='F'){
        int width=rd(),height=rd(),spitch=rd(),dpitch=rd(),lanes=rd(),nS=rd();
        vector<int> src=read(nS);
        for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++)
            out.push_back(src[(xx+yy*spitch)*lanes]);
        (void)dpitch; // logical output is dense
    } else if(m=='G'){
        int nBlkX=rd(),nBlkY=rd(),fpitch=rd(),cpitch=rd(),nC=rd();
        vector<int> combe=read(nC);
        for(int yy=0;yy<nBlkY;yy++)for(int xx=0;xx<nBlkX;xx++){
            int s=combe[(2*xx+0)+(2*yy+0)*cpitch]
                + combe[(2*xx+1)+(2*yy+0)*cpitch]
                + combe[(2*xx+0)+(2*yy+1)*cpitch]
                + combe[(2*xx+1)+(2*yy+1)*cpitch];
            out.push_back((s+2)>>2);
        }
        (void)fpitch;
    } else if(m=='B'){
        int width=rd(),height=rd(),pitch=rd(),maxv=rd(),nS=rd();
        vector<int> src=read(nS);
        int org=1+pitch; // 1-px halo origin
        for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++){
            int off=org+xx+yy*pitch;
            int sumv=src[off-1-pitch]+src[off-pitch]+src[off+1-pitch]
                + src[off-1]+src[off]+src[off+1]
                + src[off-1+pitch]+src[off+pitch]+src[off+1+pitch];
            int v=sumv>>2;
            out.push_back((v<maxv)?v:maxv);
        }
    } else if(m=='N'){
        int nBlkX=rd(),nBlkY=rd(),pitch=rd(),thY=rd(),thC=rd(),nP=rd();
        vector<int> srcY=read(nP),srcC=read(nP);
        out.assign(srcY.begin(),srcY.end()); // in-place on a copy, as upstream
        for(int yy=0;yy<nBlkY;yy++)for(int xx=0;xx<nBlkX;xx++){
            int off=xx+yy*pitch;
            out[off]=((out[off]>=thY)||(srcC[off]>=thC))?128:0;
        }
        vector<int> outlog; outlog.reserve((size_t)nBlkX*nBlkY);
        for(int yy=0;yy<nBlkY;yy++)for(int xx=0;xx<nBlkX;xx++)
            outlog.push_back(out[xx+yy*pitch]);
        out.swap(outlog);
    } else if(m=='H'){
        int width=rd(),height=rd(),spitch=rd(),scale=rd(),shift=rd(),nS=rd();
        vector<int> src=read(nS);
        int half=scale/2, org=1; // 1-col left halo
        for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++){
            int x0=(xx-half)>>shift;
            int c0=((x0+1)<<shift)-(xx-half);
            int c1=scale-c0;
            int s0=src[org+(x0+0)+yy*spitch];
            int s1=src[org+(x0+1)+yy*spitch];
            out.push_back((s0*c0+s1*c1+half)>>shift);
        }
    } else if(m=='V'){
        int width=rd(),height=rd(),spitch=rd(),scale=rd(),shift=rd(),nS=rd();
        vector<int> src=read(nS);
        int half=scale/2, org=spitch; // 1-row top halo
        for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++){
            int y0=(yy-half)>>shift;
            int c0=((y0+1)<<shift)-(yy-half);
            int c1=scale-c0;
            int s0=src[org+xx+(y0+0)*spitch];
            int s1=src[org+xx+(y0+1)*spitch];
            out.push_back((s0*c0+s1*c1+half)>>shift);
        }
    } else if(m=='S'){
        int width=rd(),height=rd(),pitch=rd(),nP=rd();
        vector<int> s0=read(nP),s1=read(nP),s2=read(nP);
        for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++){
            int off=xx+yy*pitch;
            float t=(float)s0[off]+(float)s1[off]+(float)s2[off];
            int v=(int)(t*(1.0f/3.0f));
            out.push_back(v&0xFF); // VHelper::cast_to wrap
        }
    } else if(m=='R'){
        int width=rd(),height=rd(),pitch=rd(),cpitch=rd(),thcombe=rd();
        int nS=rd(),nC=rd();
        vector<int> src=read(nS),combe=read(nC);
        int org=pitch; // 1 pad row above
        for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++){
            int off=org+xx+yy*pitch;
            int score=combe[(xx>>2)+(yy>>2)*cpitch];
            if(score>=thcombe){
                int v=(src[off-pitch]+2*src[off]+src[off+pitch]+2)>>2;
                out.push_back(v);
            } else {
                out.push_back(src[off]);
            }
        }
    } else if(m=='C'){
        int width=rd(),height=rd(),pitch=rd(),thresh=rd(),nP=rd();
        vector<int> px=read(nP),py=read(nP),cx=read(nP),cy=read(nP);
        vector<int> ox,oy; ox.reserve((size_t)width*height);
        oy.reserve((size_t)width*height);
        for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++){
            int off=xx+yy*pitch;
            oy.push_back(cy[off]);
            ox.push_back(((py[off]<=thresh)&&(cy[off]<=thresh))?0:cx[off]);
        }
        out.reserve(ox.size()+oy.size());
        out.insert(out.end(),ox.begin(),ox.end());
        out.insert(out.end(),oy.begin(),oy.end());
        (void)px; // .x of prev never read, as upstream
    } else if(m=='D'){
        int width=rd(),height=rd(),pitch=rd(),nP=rd();
        vector<int> flagp=read(nP);
        int work=0; // kf_init_contains_durty_block folded in
        for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++)
            if(flagp[xx+yy*pitch]) work=1;
        out.push_back(work);
    } else if(m=='M'){
        int L[8]; for(int i=0;i<8;i++)L[i]=rd();
        int diff8=AbsDiff(L[0],L[7]);
        int diffT=AbsDiff(L[0],L[1])+AbsDiff(L[1],L[2])+AbsDiff(L[2],L[3])
            +AbsDiff(L[3],L[4])+AbsDiff(L[4],L[5])+AbsDiff(L[5],L[6])
            +AbsDiff(L[6],L[7])-diff8;
        int diffE=AbsDiff(L[0],L[2])+AbsDiff(L[2],L[4])+AbsDiff(L[4],L[6])
            +AbsDiff(L[6],L[7])-diff8;
        int diffO=AbsDiff(L[0],L[1])+AbsDiff(L[1],L[3])+AbsDiff(L[3],L[5])
            +AbsDiff(L[5],L[7])-diff8;
        out.push_back(diffT-diffE-diffO);
    } else if(m=='A'){
        int L[8]; for(int i=0;i<8;i++)L[i]=rd();
        out.push_back(AbsDiff(L[0],L[1])+AbsDiff(L[2],L[3])
            +AbsDiff(L[4],L[5])+AbsDiff(L[6],L[7]));
    } else if(m=='J'){
        int nBlkX=rd(),nBlkY=rd(),fpitch_f=rd(),fpitch=rd();
        int shift=rd(),parity=rd(),nF=rd(),nFl=rd();
        vector<int> f0=read(nF),f1=read(nF);
        vector<int> fx((size_t)nFl,-1),fy((size_t)nFl,-1);
        for(int by=0;by<nBlkY-1;by++)for(int bx=0;bx<nBlkX-1;bx++){
            int x0=bx*4,y0=by*4;
            int sum[4]={0,0,0,0};
            for(int tx=0;tx<8;tx++){
                int x=x0+tx;
                int r0[8],r1[8];
                for(int k=0;k<8;k++){
                    r0[k]=f0[x+(y0+k)*fpitch_f];
                    r1[k]=f1[x+(y0+k)*fpitch_f];
                }
                int d8=AbsDiff(r0[0],r0[7]);
                int dT=AbsDiff(r0[0],r0[1])+AbsDiff(r0[1],r0[2])
                    +AbsDiff(r0[2],r0[3])+AbsDiff(r0[3],r0[4])
                    +AbsDiff(r0[4],r0[5])+AbsDiff(r0[5],r0[6])
                    +AbsDiff(r0[6],r0[7])-d8;
                int dE=AbsDiff(r0[0],r0[2])+AbsDiff(r0[2],r0[4])
                    +AbsDiff(r0[4],r0[6])+AbsDiff(r0[6],r0[7])-d8;
                int dO=AbsDiff(r0[0],r0[1])+AbsDiff(r0[1],r0[3])
                    +AbsDiff(r0[3],r0[5])+AbsDiff(r0[5],r0[7])-d8;
                int c0=dT-dE-dO; // combe over f0
                if(parity){ // TFF: top=f0-combe, bottom=f1/f0-mixed
                    int e8=AbsDiff(r1[0],r0[7]);
                    int eT=AbsDiff(r1[0],r0[1])+AbsDiff(r0[1],r1[2])
                        +AbsDiff(r1[2],r0[3])+AbsDiff(r0[3],r1[4])
                        +AbsDiff(r1[4],r0[5])+AbsDiff(r0[5],r1[6])
                        +AbsDiff(r1[6],r0[7])-e8;
                    int eE=AbsDiff(r1[0],r1[2])+AbsDiff(r1[2],r1[4])
                        +AbsDiff(r1[4],r1[6])+AbsDiff(r1[6],r0[7])-e8;
                    int eO=AbsDiff(r1[0],r0[1])+AbsDiff(r0[1],r0[3])
                        +AbsDiff(r0[3],r0[5])+AbsDiff(r0[5],r0[7])-e8;
                    sum[0]+=c0; sum[2]+=eT-eE-eO;
                } else { // BFF: mirrored
                    int e8=AbsDiff(r0[0],r1[7]);
                    int eT=AbsDiff(r0[0],r1[1])+AbsDiff(r1[1],r0[2])
                        +AbsDiff(r0[2],r1[3])+AbsDiff(r1[3],r0[4])
                        +AbsDiff(r0[4],r1[5])+AbsDiff(r1[5],r0[6])
                        +AbsDiff(r0[6],r1[7])-e8;
                    int eE=AbsDiff(r0[0],r0[2])+AbsDiff(r0[2],r0[4])
                        +AbsDiff(r0[4],r0[6])+AbsDiff(r0[6],r1[7])-e8;
                    int eO=AbsDiff(r0[0],r1[1])+AbsDiff(r1[1],r1[3])
                        +AbsDiff(r1[3],r1[5])+AbsDiff(r1[5],r1[7])-e8;
                    sum[2]+=c0; sum[0]+=eT-eE-eO;
                }
                sum[1]+=AbsDiff(r0[0],r1[0])+AbsDiff(r0[2],r1[2])
                    +AbsDiff(r0[4],r1[4])+AbsDiff(r0[6],r1[6]);
                sum[3]+=AbsDiff(r0[1],r1[1])+AbsDiff(r0[3],r1[3])
                    +AbsDiff(r0[5],r1[5])+AbsDiff(r0[7],r1[7]);
            }
            int c=bx+1,r0=2*(by+1)+0,r1=2*(by+1)+1;
            int v0=sum[0]>>shift,v1=sum[1]>>shift;
            int v2=sum[2]>>shift,v3=sum[3]>>shift;
            fx[(size_t)c+r0*fpitch]=(v0<0)?0:((v0>255)?255:v0);
            fy[(size_t)c+r0*fpitch]=(v1<0)?0:((v1>255)?255:v1);
            fx[(size_t)c+r1*fpitch]=(v2<0)?0:((v2>255)?255:v2);
            fy[(size_t)c+r1*fpitch]=(v3<0)?0:((v3>255)?255:v3);
        }
        out.reserve(fx.size()+fy.size());
        out.insert(out.end(),fx.begin(),fx.end());
        out.insert(out.end(),fy.begin(),fy.end());
    } else if(m=='W'){
        int width=rd(),height=rd(),pitch=rd(),parity=rd();
        int thM=rd(),thS=rd(),thLS=rd();
        int m0=rd(),s0=rd(),l0=rd(),m1=rd(),s1=rd(),l1=rd(),nP=rd();
        vector<int> c0x=read(nP),c0y=read(nP),c1x=read(nP),c1y=read(nP);
        int dstM[2]={m0,m1},dstS[2]={s0,s1},dstL[2]={l0,l1};
        int np=parity?0:1; // !parity
        for(int by=0;by<height;by++)for(int bx=0;bx<width;bx++){
            int off=bx+by*pitch;
            for(int i=0;i<2;i++){
                int vx=(i==0)?c0x[off]:c1x[off];
                int vy=(i==0)?c0y[off]:c1y[off];
                int slot=i^np;
                if(vy>=thM)dstM[slot]++;
                if(vx>=thS)dstS[slot]++;
                if(vx>=thLS)dstL[slot]++;
            }
        }
        out.push_back(dstM[0]);out.push_back(dstS[0]);out.push_back(dstL[0]);
        out.push_back(dstM[1]);out.push_back(dstS[1]);out.push_back(dstL[1]);
    } else if(m=='Q'){
        int width=rd(),height=rd(),pitch=rd(),parity=rd();
        int thM=rd(),thS=rd(),thLS=rd();
        int m0=rd(),s0=rd(),l0=rd(),m1=rd(),s1=rd(),l1=rd(),nP=rd();
        vector<int> c0Ux=read(nP),c0Uy=read(nP),c1Ux=read(nP),c1Uy=read(nP);
        vector<int> c0Vx=read(nP),c0Vy=read(nP),c1Vx=read(nP),c1Vy=read(nP);
        int dstM[2]={m0,m1},dstS[2]={s0,s1},dstL[2]={l0,l1};
        int np=parity?0:1;
        for(int by=0;by<height;by++)for(int bx=0;bx<width;bx++){
            int off=bx+by*pitch;
            for(int i=0;i<2;i++){
                int slot=i^np;
                int vx=(i==0)?c0Ux[off]:c1Ux[off];
                int vy=(i==0)?c0Uy[off]:c1Uy[off];
                if(vy>=thM)dstM[slot]++;
                if(vx>=thS)dstS[slot]++;
                if(vx>=thLS)dstL[slot]++;
                vx=(i==0)?c0Vx[off]:c1Vx[off];
                vy=(i==0)?c0Vy[off]:c1Vy[off];
                if(vy>=thM)dstM[slot]++;
                if(vx>=thS)dstS[slot]++;
                if(vx>=thLS)dstL[slot]++;
            }
        }
        out.push_back(dstM[0]);out.push_back(dstS[0]);out.push_back(dstL[0]);
        out.push_back(dstM[1]);out.push_back(dstS[1]);out.push_back(dstL[1]);
    } else if(m=='I'){
        out.push_back(0);out.push_back(0);out.push_back(0);
        out.push_back(0);out.push_back(0);out.push_back(0);
    } else { fprintf(stderr,"bad mode %c\n",m); return 2; }
    FILE* o=fopen(argv[2],"w");
    for(int v:out)fprintf(o,"%d\n",v);
    fclose(o);
    return 0;
}
