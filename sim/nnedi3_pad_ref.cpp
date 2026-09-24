/* CPU mirror of the four NNEDI3 pad/copy kernels in
 * src/opencl/nnedi3/kernels/nnedi3_pad.cl (kl_pad_h/v, kl_copy,
 * kl_pad_ref_and_copy_half twins, NNEDI3/nnedi3/nnedi3_kernel.cu @01931aa).
 * Integer pass-through; PX width is covered by the value range the runner
 * feeds (0..255 / 0..65535).  Pad kernels emulate in place on a copy of
 * the input buffer (reads are interior-only, writes margin-only, so the
 * update order is irrelevant — same as the parallel device update).
 *
 * Usage: nnedi3_pad_ref <in> <out>
 *   All ints on one line.  Modes:
 *     A padh  : A w h pitch hPad  nB buf(nB) -> pitch*h full buffer
 *               (buf rows hold interior at [hPad,hPad+w) + margins)
 *     B padv  : B w h pitch vPad  nB buf(nB)
 *               -> pitch*(h+2*vPad) full buffer, interior rows [vPad,vPad+h)
 *     C copy  : C w h dp sp  nS src(nS) -> w*h dense pixels
 *     D padref: D w4 h rpitch dpitch spitch hpad4 vpad  nS src(nS)
 *               src = w4*4 px/row interior at spitch stride
 *               -> ref dense ((w4+2*hpad4)*4 x (h+2*vpad)) then
 *                  dst dense (w4*4 x h)
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace std;

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: nnedi3_pad_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    char m=(char)rd();
    vector<int> out;
    auto read=[&](int n){ vector<int> a(n); for(int&i:a)i=rd(); return a; };

    if(m=='A'){
        int w=rd(),h=rd(),pitch=rd(),hPad=rd();
        int nB=rd(); vector<int> buf=read(nB);
        for(int y=0;y<h;y++)for(int i=0;i<hPad;i++){
            buf[(size_t)(hPad-(i+1))+ (size_t)y*pitch]=buf[(size_t)(hPad+(i+1))+(size_t)y*pitch];
            buf[(size_t)(hPad+w+i)   + (size_t)y*pitch]=buf[(size_t)(hPad+w-(i+2))+(size_t)y*pitch];
        }
        out=buf;
    } else if(m=='B'){
        int w=rd(),h=rd(),pitch=rd(),vPad=rd();
        int nB=rd(); vector<int> buf=read(nB);
        for(int i=0;i<vPad;i++)for(int xx=0;xx<w;xx++){
            buf[(size_t)xx+(size_t)(vPad-(i+1))*pitch]=buf[(size_t)xx+(size_t)(vPad+(i+1))*pitch];
            buf[(size_t)xx+(size_t)(vPad+h+i)*pitch]  =buf[(size_t)xx+(size_t)(vPad+h-(i+2))*pitch];
        }
        out=buf;
    } else if(m=='C'){
        int w=rd(),h=rd(),dp=rd(),sp=rd();
        (void)dp;
        int nS=rd(); vector<int> src=read(nS);
        for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++)
            out.push_back(src[(size_t)(xx+yy*sp)]);
    } else if(m=='D'){
        int w4=rd(),h=rd(),rp=rd(),dp=rd(),sp=rd(),hpad4=rd(),vpad=rd();
        int nS=rd(); vector<int> src=read(nS);
        int rw=(w4+2*hpad4)*4, rh=h+2*vpad;
        vector<int> ref((size_t)rp*rh,-1), dst((size_t)dp*h,-1);
        for(int y=-vpad;y<h+vpad;y++)for(int xx=-hpad4;xx<w4+hpad4;xx++){
            bool padx=true,pady=true; int sx=xx,sy=y;
            if(sx<0)sx=-sx-1; else if(sx>=w4)sx=w4-(sx-w4)-1; else padx=false;
            if(sy<0)sy=-sy-1; else if(sy>=h)sy=h-(sy-h)-1; else pady=false;
            int v[4];
            for(int k=0;k<4;k++)v[k]=src[(size_t)(sx*4+k+sy*sp)];
            size_t ro=(size_t)((xx+hpad4)*4)+(size_t)(y+vpad)*rp;
            if(padx){ref[ro+0]=v[3];ref[ro+1]=v[2];ref[ro+2]=v[1];ref[ro+3]=v[0];}
            else{ref[ro+0]=v[0];ref[ro+1]=v[1];ref[ro+2]=v[2];ref[ro+3]=v[3];}
            if(!padx&&!pady){
                size_t dout=(size_t)(xx*4)+(size_t)y*dp;
                for(int k=0;k<4;k++)dst[dout+k]=v[k];
            }
        }
        for(int yy=0;yy<rh;yy++)for(int xx=0;xx<rw;xx++)
            out.push_back(ref[(size_t)xx+(size_t)yy*rp]);
        for(int yy=0;yy<h;yy++)for(int xx=0;xx<w4*4;xx++)
            out.push_back(dst[(size_t)xx+(size_t)yy*dp]);
    } else {
        fprintf(stderr,"unknown mode %c\n",m); return 2;
    }

    FILE* o=fopen(argv[2],"w");
    for(int v:out)fprintf(o,"%d\n",v);
    fclose(o);
    return 0;
}
