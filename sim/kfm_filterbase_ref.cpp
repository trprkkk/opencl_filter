/* CPU mirror of the six KFM KFMFilterBase.cu kernels in
 * src/opencl/kfm/kernels/kfm_filterbase.cl.  cpu_calc_combe / cpu_merge_uvcoefs
 * / cpu_apply_uvcoefs_420 / cpu_padv / cpu_padh are exact upstream twins;
 * mode E replicates the CUDA kl_extend_coef2 device kernel (the .cl
 * transliteration target — the upstream CPU *fallback* branch differs at rows
 * 0 and height-1, see the .cl header).
 *
 * Usage: kfm_filterbase_ref <in> <out>
 *   All ints on one line.  Modes:
 *     K calc_combe   : K width height pitch  nS  src(...)
 *                      -> output width*height (logical, row-major); rows are
 *                         verified interior; give source height enough rows.
 *     M merge_uvcoefs: M width height pitchY pitchUV lx ly  nF
 *                      fY(...) fU(...) fV(...)  (each length nF = pitchY*height
 *                      for Y; U/V read at subsampled offsets, sized pitchUV)
 *                      -> output Y plane after in-place fold (width*height)
 *     E extend_coef2 : E width height pitch  nS  src(...)
 *                      -> output width*height
 *     A apply_uvcoefs_420: A widthUV heightUV pitchY pitchUV  nY nUV
 *                      fY(...) fU(...) fV(...)
 *                      -> output = U plane then V plane (each widthUV*heightUV)
 *     V padv         : V width height pitch vpad  nBuf  buf(...)
 *                      buf = full plane pitch*(height+2*vpad), interior origin
 *                      at vpad*pitch; in-place vertical mirror pad.
 *                      -> output whole buffer
 *     H padh         : H width height pitch hpad  nBuf  buf(...)
 *                      buf = full plane pitch*height (pitch >= width+2*hpad),
 *                      interior origin at hpad; in-place horizontal mirror pad.
 *                      -> output whole buffer
 *     B both (padv then padh, upstream Deblock order):
 *                      B width height pitch vpad hpad  nBuf  buf(...)
 *                      buf = full plane pitch*(height+2*vpad)
 *                      (pitch >= width+2*hpad), interior origin at
 *                      hpad+vpad*pitch; padv over (width,height), then padh
 *                      over (width,height+2*vpad) from the padded top.
 *                      -> output whole buffer
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace std;

static int CalcCombe(int a,int b,int c,int d,int e){
    int v=a + c*4 + e - (b+d)*3; if(v<0)v=-v; return v;
}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: kfm_filterbase_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    char m=(char)rd();
    vector<int> out;

    auto read=[&](int n){ vector<int> a(n); for(int&i:a)i=rd(); return a; };

    if(m=='K'){
        int width=rd(),height=rd(),pitch=rd(),nS=rd();
        vector<int> src=read(nS);
        out.assign((size_t)width*height,-1); // border sentinel (rig-bound: padded plane)
        for(int yy=2;yy<height-2;yy++)for(int xx=0;xx<width;xx++){
            int off=xx+yy*pitch;
            int combe=CalcCombe(src[off-2*pitch],src[off-pitch],src[off],
                                src[off+pitch],src[off+2*pitch]);
            combe>>=2; if(combe<0)combe=0; else if(combe>255)combe=255;
            out[(size_t)yy*width+xx]=combe;
        }
    } else if(m=='M'){
        int width=rd(),height=rd(),pitchY=rd(),pitchUV=rd(),lx=rd(),ly=rd();
        int nY=rd(),nU=rd();
        vector<int> fY=read(nY),fU=read(nU),fV=read(nU);
        // mirror in-place on a copy
        out.assign(fY.begin(),fY.end());
        for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++){
            int oY=xx+yy*pitchY;
            int oUV=(xx>>lx)+(yy>>ly)*pitchUV;
            int u=fU[oUV],v=fV[oUV],yv=out[oY];
            if(u>yv)yv=u; if(v>yv)yv=v;
            out[oY]=yv;
        }
        // output logical width*height
        vector<int> outlog; outlog.reserve(width*height);
        for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++)
            outlog.push_back(out[xx+yy*pitchY]);
        out.swap(outlog);
    } else if(m=='E'){
        int width=rd(),height=rd(),pitch=rd(),nS=rd();
        vector<int> src=read(nS);
        out.assign((size_t)width*height,0);
        for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++){
            int y0=yy-1; if(y0<0)y0=0;
            int y2=yy+1; if(y2>height-1)y2=height-1;
            int a=src[xx+y0*pitch], b=src[xx+yy*pitch], c=src[xx+y2*pitch];
            int mx=a; if(b>mx)mx=b; if(c>mx)mx=c;
            out[(size_t)yy*width+xx]=mx;
        }
    } else if(m=='A'){
        int widthUV=rd(),heightUV=rd(),pitchY=rd(),pitchUV=rd();
        int nY=rd(),nUV=rd();
        vector<int> fY=read(nY),fU=read(nUV),fV=read(nUV);
        for(int yy=0;yy<heightUV;yy++)for(int xx=0;xx<widthUV;xx++){
            int oY0=(2*xx+0)+(2*yy+0)*pitchY;
            int v=fY[oY0]+fY[oY0+1]+fY[oY0+pitchY]+fY[oY0+pitchY+1];
            int avg=(v+2)>>2;
            int oUV=xx+yy*pitchUV;
            fU[oUV]=avg; fV[oUV]=avg;
        }
        for(int yy=0;yy<heightUV;yy++)for(int xx=0;xx<widthUV;xx++)
            out.push_back(fU[xx+yy*pitchUV]);
        for(int yy=0;yy<heightUV;yy++)for(int xx=0;xx<widthUV;xx++)
            out.push_back(fV[xx+yy*pitchUV]);
    } else if(m=='V'){
        int width=rd(),height=rd(),pitch=rd(),vpad=rd(),nBuf=rd();
        vector<int> buf=read(nBuf);
        int org=vpad*pitch; // interior origin
        for(int yy=0;yy<vpad;yy++)for(int xx=0;xx<width;xx++){
            buf[org+xx+(-yy-1)*pitch]=buf[org+xx+yy*pitch];
            buf[org+xx+(height+yy)*pitch]=buf[org+xx+(height-yy-1)*pitch];
        }
        out.swap(buf);
    } else if(m=='H'){
        int width=rd(),height=rd(),pitch=rd(),hpad=rd(),nBuf=rd();
        vector<int> buf=read(nBuf);
        int org=hpad; // interior origin
        for(int yy=0;yy<height;yy++)for(int xx=0;xx<hpad;xx++){
            buf[org+(-xx-1)+yy*pitch]=buf[org+xx+yy*pitch];
            buf[org+(width+xx)+yy*pitch]=buf[org+(width-xx-1)+yy*pitch];
        }
        out.swap(buf);
    } else if(m=='B'){
        int width=rd(),height=rd(),pitch=rd(),vpad=rd(),hpad=rd(),nBuf=rd();
        vector<int> buf=read(nBuf);
        int org=hpad+vpad*pitch; // interior origin
        for(int yy=0;yy<vpad;yy++)for(int xx=0;xx<width;xx++){ // padv first
            buf[org+xx+(-yy-1)*pitch]=buf[org+xx+yy*pitch];
            buf[org+xx+(height+yy)*pitch]=buf[org+xx+(height-yy-1)*pitch];
        }
        int orgT=org-vpad*pitch; // top of the padded column range
        int heightP=height+2*vpad;
        for(int yy=0;yy<heightP;yy++)for(int xx=0;xx<hpad;xx++){ // then padh
            buf[orgT+(-xx-1)+yy*pitch]=buf[orgT+xx+yy*pitch];
            buf[orgT+(width+xx)+yy*pitch]=buf[orgT+(width-xx-1)+yy*pitch];
        }
        out.swap(buf);
    } else { fprintf(stderr,"bad mode %c\n",m); return 2; }
    FILE* o=fopen(argv[2],"w");
    for(int v:out)fprintf(o,"%d\n",v);
    fclose(o);
    return 0;
}
