/* ============================================================================
 * ktgmc_cpu_ref.cpp — CPU reference mirror of the KTGMC simple kernels.
 *
 * Algorithm reference used to validate the OpenCL port.  It reproduces,
 * element-for-element, the math of the CUDA kernels in
 * AviSynthCUDAFilters/KTGMC/Kernel.cu (plus the host ResamplingProgram).
 * Scalar and single-threaded — each KTGMC simple kernel is per-pixel
 * separable, so the translation is exact regardless of vectorization.
 *
 * Build:
 *     g++ -O2 -std=c++17 sim/ktgmc_cpu_ref.cpp -o build/ktgmc_cpu_ref
 * Run:
 *     build/ktgmc_cpu_ref <in_dir> <out_dir> <bitdepth 8|16>
 *
 * in_dir planes (raw = pitch*height*px bytes, rows of `pitch` samples, only
 * `width` <= pitch columns valid): a,b,c,ref,n2,n1,p1,p2,field  +  plane.info
 *   plane.info line:  "<name> <width> <height> <pitch>"
 * ==========================================================================*/
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
using namespace std;

template<typename PX> constexpr int PixelMax();
template<> constexpr int PixelMax<uint8_t>(){return 255;}
template<> constexpr int PixelMax<uint16_t>(){return 65535;}

template<typename PX> struct Plane {
    int w=0,h=0,pitch=0;
    vector<PX> data;
    const PX& at(int x,int y) const { return data[(size_t)y*pitch+x]; }
          PX& at(int x,int y)       { return data[(size_t)y*pitch+x]; }
};

template<typename PX> bool readPlane(const string& dir,const string& name,Plane<PX>& pl){
    string infoPath=dir+"/plane.info";
    FILE* fi=fopen(infoPath.c_str(),"r"); if(!fi)return false;
    char buf[512]; bool ok=false;
    while(fgets(buf,sizeof(buf),fi)){
        char nm[64]; int w,h,pt;
        if(sscanf(buf,"%63s %d %d %d",nm,&w,&h,&pt)==4 && name==nm){ok=true;pl.w=w;pl.h=h;pl.pitch=pt;break;}
    }
    fclose(fi); if(!ok)return false;
    string p=dir+"/"+name+".raw";
    FILE* f=fopen(p.c_str(),"rb"); if(!f)return false;
    pl.data.resize((size_t)pl.pitch*pl.h);
    size_t want=(size_t)pl.pitch*pl.h;
    if(fread(pl.data.data(),sizeof(PX),want,f)!=want){fclose(f);return false;}
    fclose(f); return true;
}
template<typename PX> bool writePlane(const string& dir,const string& name,const Plane<PX>& pl){
    string p=dir+"/"+name+".raw";
    FILE* f=fopen(p.c_str(),"wb"); if(!f)return false;
    fwrite(pl.data.data(),sizeof(PX),(size_t)pl.pitch*pl.h,f); fclose(f); return true;
}

/* ---------------- ResamplingProgram (host side) ---------------- */
struct ResamplingProgram { int filter_size=0,target_size=0; vector<int> offset; vector<double> coef; };

struct Mitchell {
    double p0,p2,p3,q0,q1,q2,q3;
    Mitchell(double b,double c){
        p0=(6.-2.*b)/6.; p2=(-18.+12.*b+6.*c)/6.; p3=(12.-9.*b-6.*c)/6.;
        q0=(8.*b+24.*c)/6.; q1=(-12.*b-48.*c)/6.; q2=(6.*b+30.*c)/6.; q3=(-b-6.*c)/6.;}
    double support(){return 2.0;}
    double f(double x){x=fabs(x); return (x<1)?(p0+x*x*(p2+x*p3)):(x<2)?(q0+x*(q1+x*(q2+x*q3))):0.0;}
};

ResamplingProgram build_resampling_program(int source_size,double crop_start,double crop_size,
                                           int target_size,double b,double c){
    Mitchell mit(b,c);
    double filter_scale=double(target_size)/crop_size, filter_step=min(filter_scale,1.0);
    double filter_support=mit.support()/filter_step; int fir=int(ceil(filter_support*2));
    ResamplingProgram P; P.filter_size=fir; P.target_size=target_size;
    P.offset.assign(target_size,0); P.coef.assign((size_t)target_size*fir,0.0);
    double pos,pos_step=crop_size/target_size;
    pos=(fir==1)?crop_start:crop_start+((crop_size-target_size)/(target_size*2));
    for(int i=0;i<target_size;++i){
        int end_pos=int(pos+filter_support); if(end_pos>source_size-1)end_pos=source_size-1;
        int start_pos=end_pos-fir+1; if(start_pos<0)start_pos=0;
        P.offset[i]=start_pos;
        double total=0.0, ok_pos=clamp(pos,0.0,double(source_size-1));
        for(int j=0;j<fir;++j)total+=mit.f((start_pos+j-ok_pos)*filter_step);
        if(total==0.0)total=1.0;
        double value=0.0;
        for(int k=0;k<fir;++k){double nv=value+mit.f((start_pos+k-ok_pos)*filter_step)/total;
            P.coef[i*fir+k]=nv-value; value=nv;}
        pos+=pos_step;
    }
    return P;
}

struct Gaussian {
    double param;
    Gaussian(double p){ param=clamp(p,0.1,100.0); } /* upstream ctor clamps */
    double support(){return 4.0;}
    /* Upstream GaussianFilter::f verbatim. Bit-identity of pow() with the
     * Python golden holds on the same host libm (both call C pow); cross-libm
     * 1-ulp variance is out of scope — these coefs are host-computed and
     * float-rounded for the device (see write_gprog). */
    double f(double value){ double p=param*0.1; return pow(2.0,-p*value*value); }
};

/* KGaussResize program builder: the same upstream GetResamplingProgram
 * skeleton as build_resampling_program, with the GaussianFilter f/support.
 * Kept as a full transcription (not a template) so the Mitchell path stays
 * byte-stable. Upstream dispatches fir 8/9 only and throws otherwise; the
 * runner asserts the fir of every program built here. */
ResamplingProgram build_gaussian_program(int source_size,double crop_start,double crop_size,
                                         int target_size,double p){
    Gaussian gs(p);
    double filter_scale=double(target_size)/crop_size, filter_step=min(filter_scale,1.0);
    double filter_support=gs.support()/filter_step; int fir=int(ceil(filter_support*2));
    ResamplingProgram P; P.filter_size=fir; P.target_size=target_size;
    P.offset.assign(target_size,0); P.coef.assign((size_t)target_size*fir,0.0);
    double pos,pos_step=crop_size/target_size;
    pos=(fir==1)?crop_start:crop_start+((crop_size-target_size)/(target_size*2));
    for(int i=0;i<target_size;++i){
        int end_pos=int(pos+filter_support); if(end_pos>source_size-1)end_pos=source_size-1;
        int start_pos=end_pos-fir+1; if(start_pos<0)start_pos=0;
        P.offset[i]=start_pos;
        double total=0.0, ok_pos=clamp(pos,0.0,double(source_size-1));
        for(int j=0;j<fir;++j)total+=gs.f((start_pos+j-ok_pos)*filter_step);
        if(total==0.0)total=1.0;
        double value=0.0;
        for(int k=0;k<fir;++k){double nv=value+gs.f((start_pos+k-ok_pos)*filter_step)/total;
            P.coef[i*fir+k]=nv-value; value=nv;}
        pos+=pos_step;
    }
    return P;
}

/* ---------------- per-pixel kernels (scalar, double-accumulate) ------- */
template<typename PX>
void kernel_resample_v(const Plane<PX>& src,Plane<PX>& dst,int outH,const ResamplingProgram& prog){
    for(int y=0;y<outH;++y){int begin=prog.offset[y];
        for(int x=0;x<dst.w;++x){double acc=0;
            for(int i=0;i<prog.filter_size;++i)acc+=(double)src.at(x,begin+i)*prog.coef[y*prog.filter_size+i];
            acc=clamp(acc,0.0,(double)PixelMax<PX>()); dst.at(x,y)=(PX)(int)(acc+0.5);}}
}
template<typename PX>
void kernel_resample_h(const Plane<PX>& src,Plane<PX>& dst,const ResamplingProgram& prog){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){int begin=prog.offset[x]; double acc=0;
        for(int i=0;i<prog.filter_size;++i)acc+=(double)src.at(begin+i,y)*prog.coef[x*prog.filter_size+i];
        acc=clamp(acc,0.0,(double)PixelMax<PX>()); dst.at(x,y)=(PX)(int)(acc+0.5);}
}
template<typename PX>
void kernel_makediff(const Plane<PX>& a,const Plane<PX>& b,Plane<PX>& dst,int mode,int range_half){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){int va=a.at(x,y),vb=b.at(x,y);
        int v=(mode==0)?(va-vb+range_half):(va+vb-range_half); v=clamp(v,0,PixelMax<PX>()); dst.at(x,y)=(PX)v;}}
template<typename PX>
void kernel_rg_box3x3(const Plane<PX>& src,Plane<PX>& dst,int mode){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){int s=src.at(x,y),v=s;
        if(x>=1&&x<dst.w-1&&y>=1&&y<dst.h-1){
            auto g=[&](int dx,int dy){return (int)src.at(x+dx,y+dy);};
            int ul=g(-1,-1),uc=g(0,-1),ur=g(1,-1),ml=g(-1,0),mr=g(1,0),ll=g(-1,1),lc=g(0,1),lr=g(1,1),vt;
            if(mode==20)vt=(ul+uc+ur+ml+s+mr+ll+lc+lr+4)/9;
            else vt=(ul+2*uc+ur+2*(ml+2*s+mr)+ll+2*lc+lr+8)>>4;
            v=clamp(vt,0,PixelMax<PX>());}
        dst.at(x,y)=(PX)v;}}
static inline void cswap(int& a,int& b){int x=a,y=b;a=min(x,y);b=max(x,y);}
static inline void sort8(int*a){
    cswap(a[0],a[1]);cswap(a[2],a[3]);cswap(a[4],a[5]);cswap(a[6],a[7]);cswap(a[0],a[2]);
    cswap(a[1],a[3]);cswap(a[4],a[6]);cswap(a[5],a[7]);cswap(a[1],a[2]);cswap(a[5],a[6]);
    cswap(a[0],a[4]);cswap(a[1],a[5]);cswap(a[2],a[6]);cswap(a[3],a[7]);cswap(a[2],a[4]);
    cswap(a[3],a[5]);cswap(a[1],a[2]);cswap(a[3],a[4]);cswap(a[5],a[6]);
}
static inline void sort9(int*a){
    cswap(a[0],a[1]);cswap(a[3],a[4]);cswap(a[6],a[7]);cswap(a[1],a[2]);cswap(a[4],a[5]);
    cswap(a[7],a[8]);cswap(a[0],a[1]);cswap(a[3],a[4]);cswap(a[6],a[7]);cswap(a[0],a[3]);
    cswap(a[1],a[4]);cswap(a[2],a[5]);cswap(a[3],a[6]);cswap(a[4],a[7]);cswap(a[5],a[8]);
    cswap(a[0],a[3]);cswap(a[1],a[4]);cswap(a[2],a[5]);cswap(a[1],a[3]);cswap(a[5],a[7]);
    cswap(a[2],a[6]);cswap(a[4],a[6]);cswap(a[2],a[4]);cswap(a[2],a[3]);cswap(a[5],a[6]);
}
template<typename PX>
void kernel_removegrain(const Plane<PX>& src,Plane<PX>& dst,int n){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){int s=src.at(x,y),v=s;
        if(x>=1&&x<dst.w-1&&y>=1&&y<dst.h-1){
            int a[8]={src.at(x-1,y-1),src.at(x,y-1),src.at(x+1,y-1),src.at(x-1,y),
                      src.at(x+1,y),src.at(x-1,y+1),src.at(x,y+1),src.at(x+1,y+1)};
            sort8(a); v=clamp(s,a[n-1],a[7-(n-1)]);} dst.at(x,y)=(PX)v;}}
template<typename PX>
void kernel_repair(const Plane<PX>& src,const Plane<PX>& ref,Plane<PX>& dst,int n){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){int s=src.at(x,y),v=s;
        if(x>=1&&x<dst.w-1&&y>=1&&y<dst.h-1){
            int a[9]={ref.at(x-1,y-1),ref.at(x,y-1),ref.at(x+1,y-1),ref.at(x-1,y),s,
                      ref.at(x+1,y),ref.at(x-1,y+1),ref.at(x,y+1),ref.at(x+1,y+1)};
            sort9(a); v=clamp(s,a[n-1],a[8-(n-1)]);} dst.at(x,y)=(PX)v;}}
template<typename PX>
void kernel_vcleaner(const Plane<PX>& src,Plane<PX>& dst){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){int b=src.at(x,y),v=b;
        if(y>=1&&y<dst.h-1){int a=src.at(x,y-1),c=src.at(x,y+1); v=min(max(min(a,b),c),max(a,b));}
        dst.at(x,y)=(PX)v;}}
template<typename PX>
void kernel_to_full_range(const Plane<PX>& src,Plane<PX>& dst,int is_uv){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){double s=(double)src.at(x,y);
        double d=(is_uv)?(s-128.0)*(128.0/112.0)+128.0:(s-16.0)*(255.0/219.0);
        d=clamp(d+0.5,0.0,(double)PixelMax<PX>()); dst.at(x,y)=(PX)(int)d;}}
template<typename PX>
void kernel_merge(const Plane<PX>& a,const Plane<PX>& b,Plane<PX>& dst,int weight){
    int inv=32767-weight;
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){int v=((int)a.at(x,y)*inv+(int)b.at(x,y)*weight+16384)>>15;
        v=clamp(v,0,PixelMax<PX>()); dst.at(x,y)=(PX)v;}}
template<typename PX>
void kernel_box5(const Plane<PX>& src,Plane<PX>& dst,int is_min){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){int v2=src.at(x,y);
        int v0=(y-2>=0)?src.at(x,y-2):v2, v1=(y-1>=0)?src.at(x,y-1):v2;
        int v3=(y+1<dst.h)?src.at(x,y+1):v2, v4=(y+2<dst.h)?src.at(x,y+2):v2;
        int m = is_min?min(min(min(v0,v1),min(v2,v3)),v4):max(max(max(v0,v1),max(v2,v3)),v4);
        dst.at(x,y)=(PX)m;}}
template<typename PX>
void kernel_logic(const Plane<PX>& a,const Plane<PX>& b,Plane<PX>& dst,int is_min){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){int va=a.at(x,y),vb=b.at(x,y);
        dst.at(x,y)=(PX)(is_min?min(va,vb):max(va,vb));}}
template<typename PX>
void kernel_vresharpen(const Plane<PX>& src,Plane<PX>& dst){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){int v1=src.at(x,y);
        int v0=(y==0)?v1:src.at(x,y-1), v2=(y==dst.h-1)?v1:src.at(x,y+1);
        dst.at(x,y)=(PX)((min(v0,min(v1,v2))+max(v0,max(v1,v2))+1)>>1);}}
template<typename PX>
void kernel_resharpen(const Plane<PX>& s0,const Plane<PX>& s1,Plane<PX>& dst,double sharpAdj){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){double a=s0.at(x,y),b=s1.at(x,y);
        double lut=clamp(a+(a-b)*sharpAdj+0.5,0.0,(double)PixelMax<PX>());
        dst.at(x,y)=(PX)(int)lut;}}
template<typename PX>
void kernel_limitos(const Plane<PX>& src,const Plane<PX>& ref,const Plane<PX>& cb,const Plane<PX>& cf,Plane<PX>& dst,int osv){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){int s=src.at(x,y),r=ref.at(x,y),
        b=cb.at(x,y),f=cf.at(x,y),mn=min(r,min(b,f)),mx=max(r,max(b,f));
        dst.at(x,y)=(PX)clamp(s,mn-osv,mx+osv);}}
template<typename PX>
void kernel_lossless(const Plane<PX>& xa,const Plane<PX>& ya,Plane<PX>& dst,double half,double maxval){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){double vx=xa.at(x,y),vy=ya.at(x,y),v;
        if((vx-half)*(vy-half)<0)v=half; else if(fabs(vx-half)<fabs(vy-half))v=vx; else v=vy;
        dst.at(x,y)=(PX)(int)clamp(v,0.0,maxval);}}
template<typename PX>
void kernel_tweak(const Plane<PX>& rep,const Plane<PX>& bob,const Plane<PX>& blr,Plane<PX>& dst,double scale,double invscale){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){
        double repair=rep.at(x,y)*invscale,bobbed=bob.at(x,y)*invscale,blur=blr.at(x,y)*invscale;
        double tweaked=clamp(bobbed,repair-3,repair+3),ret;
        if((blur+7)<tweaked)ret=blur+2; else if((blur-7)>tweaked)ret=blur-2;
        else ret=(blur*51+tweaked*49)*(1.0/100.0);
        double d=clamp(ret*scale+0.5,0.0,(double)PixelMax<PX>()); dst.at(x,y)=(PX)(int)d;}}
template<typename PX>
void kernel_erroradjust(const Plane<PX>& src,const Plane<PX>& mt,Plane<PX>& dst,double errorAdj){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){double a=src.at(x,y),b=mt.at(x,y);
        double lut=clamp(a*(errorAdj+1)-b*errorAdj+0.5,0.0,(double)PixelMax<PX>());
        dst.at(x,y)=(PX)(int)lut;}}
template<typename PX>
void kernel_bobshimmer(const Plane<PX>& src,const Plane<PX>& diff,const Plane<PX>& c1,const Plane<PX>& c2,Plane<PX>& dst,int scale){
    int h=128<<scale;
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){int s=src.at(x,y),df=diff.at(x,y),
        c1v=c1.at(x,y),c2v=c2.at(x,y);
        df=(df<(129<<scale))?df:((c1v<h)?h:c1v);
        df=(df>(127<<scale))?df:((c2v>h)?h:c2v);
        dst.at(x,y)=(PX)clamp(s+df-h,0,PixelMax<PX>());}}
template<typename PX>
void kernel_soften1(const Plane<PX>& src,const Plane<PX>& r0,const Plane<PX>& r1,Plane<PX>& dst,int sc0,int sc1){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){int s=src.at(x,y),
        a0=sc0?s:r0.at(x,y),a1=sc1?s:r1.at(x,y);
        dst.at(x,y)=(PX)clamp((a0+2*s+a1+2)>>2,0,PixelMax<PX>());}}
template<typename PX>
void kernel_soften2(const Plane<PX>& src,const Plane<PX>& r0,const Plane<PX>& r1,
                    const Plane<PX>& r2,const Plane<PX>& r3,Plane<PX>& dst,int sc0,int sc1,int sc2,int sc3){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){int s=src.at(x,y),
        a0=sc0?s:r0.at(x,y),a1=sc1?s:r1.at(x,y),a2=sc2?s:r2.at(x,y),a3=sc3?s:r3.at(x,y);
        dst.at(x,y)=(PX)clamp((a2+4*a0+6*s+4*a1+a3+4)>>4,0,PixelMax<PX>());}}
template<typename PX>
void kernel_weave(const Plane<PX>& top,const Plane<PX>& bottom,Plane<PX>& dst){
    for(int y=0;y<top.h;++y)for(int x=0;x<top.w;++x){
        dst.at(x,2*y+0)=top.at(x,y); dst.at(x,2*y+1)=bottom.at(x,y);}}
template<typename PX>
void kernel_wiener_v(const Plane<PX>& src,Plane<PX>& dst){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){
        auto P=[&](int dy)->int{return (int)src.at(x,y+dy);};
        int v;
        if(y<2) v=(P(0)+P(1)+1)>>1;
        else if(y<(int)dst.h-4){ int num=P(-2)+((-P(-1)+4*P(0)+4*P(1)-P(2))*5)+P(3)+16;
            v=clamp(num>>5,0,PixelMax<PX>()); }
        else if(y<(int)dst.h-1) v=(P(0)+P(1)+1)>>1;
        else v=P(0);
        dst.at(x,y)=(PX)v;}}
template<typename PX>
void kernel_wiener_h(const Plane<PX>& src,Plane<PX>& dst){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x){
        auto P=[&](int dx)->int{return (int)src.at(x+dx,y);};
        int v;
        if(x<2) v=(P(0)+P(1)+1)>>1;
        else if(x<(int)dst.w-4){ int num=P(-2)+((-P(-1)+4*P(0)+4*P(1)-P(2))*5)+P(3)+16;
            v=clamp(num>>5,0,PixelMax<PX>()); }
        else if(x<(int)dst.w-1) v=(P(0)+P(1)+1)>>1;
        else v=P(0);
        dst.at(x,y)=(PX)v;}}

template<typename PX>
void kernel_copy(const Plane<PX>& src,Plane<PX>& dst){
    for(int y=0;y<dst.h;++y)for(int x=0;x<dst.w;++x)dst.at(x,y)=src.at(x,y);}

template<typename PX>
long long kernel_plane_sad(const Plane<PX>& a,const Plane<PX>& b){
    long long sum=0;
    for(int y=0;y<a.h;++y)for(int x=0;x<a.w;++x){ long long d=(long long)a.at(x,y)-b.at(x,y);
        sum += d<0?-d:d; }
    return sum;
}

/* kl_init_sad twin: zero a float SAD buffer of length N. Upstream's launch is
 * <<<1, radius*2*3>>> with no length arg; the length+guard follows the repo
 * convention (kf_init_uint64). The driver prefills with deterministic garbage
 * (sanity-checked nonzero) so a broken zeroing loop would show in the file. */
static void kernel_init_sad(float* sad,int N){
    for(int i=0;i<N;++i)sad[i]=0.0f;}

/* Write one gaussian program's device-bound tables: int offsets, double coefs
 * (%.17g round-trips), and the float coefs the device actually consumes, as
 * %08x bits. Upstream stores pixel_coefficient_float[i] = float(new_value);
 * the (float) cast here is the same correctly-rounded conversion. */
static void write_gprog(const string& outDir,const string& tag,const ResamplingProgram& P){
    FILE* fo=fopen((outDir+"/gprog_"+tag+"_offset.txt").c_str(),"w");
    for(int i:P.offset)fprintf(fo,"%d\n",i); fclose(fo);
    FILE* fc=fopen((outDir+"/gprog_"+tag+"_coef.txt").c_str(),"w");
    for(double c:P.coef)fprintf(fc,"%.17g\n",c); fclose(fc);
    FILE* fx=fopen((outDir+"/gprog_"+tag+"_coef_f32.txt").c_str(),"w");
    for(double c:P.coef){ float f=(float)c; uint32_t u; memcpy(&u,&f,4); fprintf(fx,"%08x\n",u); }
    fclose(fx);
}

/* ---------------- driver ---------------- */
template<typename PX>
bool run_all(const string& inDir,const string& outDir,const ResamplingProgram& progV,const ResamplingProgram& progH,int FIELD_H,
              const ResamplingProgram& gV9,const ResamplingProgram& gH9,
              const ResamplingProgram& gV8,const ResamplingProgram& gH8,int GH_H){
    Plane<PX> a,b,c,ref,n2,n1,p1,p2,field,gsrc;
    if(!readPlane(inDir,"a",a)||!readPlane(inDir,"b",b)||!readPlane(inDir,"c",c)||
       !readPlane(inDir,"ref",ref)||!readPlane(inDir,"n2",n2)||!readPlane(inDir,"n1",n1)||
       !readPlane(inDir,"p1",p1)||!readPlane(inDir,"p2",p2)||!readPlane(inDir,"field",field)||
       !readPlane(inDir,"gsrc",gsrc)) return false;
    int W=a.w,H=a.h,pitch=a.pitch;
    auto out=[&](int w,int h){Plane<PX> o;o.w=w;o.h=h;o.pitch=pitch;o.data.assign((size_t)pitch*h,0);return o;};
    const int bits = (sizeof(PX)==1)?8:16;
    int RANGE=(1<<(bits-1));
    {auto o=out(W,H);kernel_makediff(a,b,o,0,RANGE);writePlane(outDir,"makediff",o);}
    {auto o=out(W,H);kernel_makediff(a,b,o,1,RANGE);writePlane(outDir,"adddiff",o);}
    {auto o=out(W,H);kernel_rg_box3x3(a,o,11);writePlane(outDir,"rg11",o);}
    {auto o=out(W,H);kernel_rg_box3x3(a,o,20);writePlane(outDir,"rg20",o);}
    for(int n=1;n<=4;++n){auto o=out(W,H);kernel_removegrain(a,o,n);writePlane(outDir,"rgclip"+to_string(n),o);}
    for(int n=1;n<=4;++n){auto o=out(W,H);kernel_repair(a,ref,o,n);writePlane(outDir,"repair"+to_string(n),o);}
    {auto o=out(W,H);kernel_vcleaner(a,o);writePlane(outDir,"vclean",o);}
    {auto o=out(W,H);kernel_to_full_range(a,o,0);writePlane(outDir,"tfr_y",o);}
    {auto o=out(W,H);kernel_to_full_range(a,o,1);writePlane(outDir,"tfr_uv",o);}
    {auto o=out(W,H);kernel_merge(a,b,o,(int)(0.5f*32767));writePlane(outDir,"merge",o);}
    {auto o=out(W,FIELD_H*2);kernel_resample_v(field,o,FIELD_H*2,progV);writePlane(outDir,"resample_v",o);}
    {auto o=out(W,H);kernel_resample_h(a,o,progH);writePlane(outDir,"resample_h",o);}
    {auto o=out(W,GH_H);kernel_resample_v(gsrc,o,GH_H,gV9);writePlane(outDir,"gres_v9",o);}
    {auto o=out(W,GH_H);kernel_resample_h(gsrc,o,gH9);writePlane(outDir,"gres_h9",o);}
    {auto o=out(W,GH_H);kernel_resample_v(gsrc,o,GH_H,gV8);writePlane(outDir,"gres_v8",o);}
    {auto o=out(W,GH_H);kernel_resample_h(gsrc,o,gH8);writePlane(outDir,"gres_h8",o);}
    {auto o=out(W,H);kernel_box5(a,o,1);writePlane(outDir,"box5min",o);}
    {auto o=out(W,H);kernel_box5(a,o,0);writePlane(outDir,"box5max",o);}
    {auto o=out(W,H);kernel_logic(a,b,o,1);writePlane(outDir,"logicmin",o);}
    {auto o=out(W,H);kernel_logic(a,b,o,0);writePlane(outDir,"logicmax",o);}
    {auto o=out(W,H);kernel_vresharpen(a,o);writePlane(outDir,"vresharpen",o);}
    {auto o=out(W,H);kernel_resharpen(a,b,o,0.2);writePlane(outDir,"resharpen",o);}
    {auto o=out(W,H);kernel_limitos(a,ref,b,c,o,3);writePlane(outDir,"limitos",o);}
    {auto o=out(W,H);kernel_lossless(a,b,o,(double)RANGE,(double)PixelMax<PX>());writePlane(outDir,"lossless",o);}
    {double scale=(double)(1<<(bits-8)),inv=1.0/scale;auto o=out(W,H);kernel_tweak(a,b,ref,o,scale,inv);writePlane(outDir,"tweak",o);}
    {auto o=out(W,H);kernel_erroradjust(a,b,o,0.05);writePlane(outDir,"erroradj",o);}
    {auto o=out(W,H);kernel_bobshimmer(a,b,ref,c,o,bits-8);writePlane(outDir,"bobshimmer",o);}
    {auto o=out(W,H);kernel_soften1(a,n1,p1,o,1,0);writePlane(outDir,"soften1",o);}
    {auto o=out(W,H);kernel_soften2(a,n2,n1,p1,p2,o,0,1,0,1);writePlane(outDir,"soften2",o);}
    {auto o=out(W,2*H);kernel_weave(n1,p1,o);writePlane(outDir,"weave",o);}
    {auto o=out(W,H);kernel_copy(a,o);writePlane(outDir,"copy",o);}
    {auto o=out(W,H);kernel_wiener_v(a,o);writePlane(outDir,"wiener_v",o);}
    {auto o=out(W,H);kernel_wiener_h(a,o);writePlane(outDir,"wiener_h",o);}
    { FILE* f=fopen((outDir+"/plane_sad.txt").c_str(),"w"); fprintf(f,"%lld\n",kernel_plane_sad(a,b)); fclose(f); }
    { const int NS[]={1,6,12,18,24,30,61,63,64,65,255,256,257,2048};
      for(int N:NS){ vector<float> buf(N);
        for(int i=0;i<N;++i){ unsigned u=(unsigned)(i*2654435761u+11u); buf[i]=(float)(u%1000)/1000.0f+0.001f; }
        int nz=0; for(float v:buf) if(v!=0.0f) nz++;
        if(nz==0){ fprintf(stderr,"init_sad prefill vacuous N=%d\n",N); return false; }
        kernel_init_sad(buf.data(),N);
        FILE* f=fopen((outDir+"/init_sad_"+to_string(N)+".txt").c_str(),"w");
        for(float v:buf) fprintf(f,"%.1f\n",v);
        fclose(f); } }
    return true;
}

int main(int argc,char** argv){
    if(argc<4){fprintf(stderr,"usage: ref <in_dir> <out_dir> <bitdepth>\n");return 2;}
    string inDir=argv[1],outDir=argv[2]; int bits=atoi(argv[3]);
    // KTGMC_Bob vertical: 5-row field -> 10 rows, Mitchell b=0,c=0.5 (Catmull-Rom)
    ResamplingProgram progV=build_resampling_program(5,0.25,5,10,0.0,0.5);
    // horizontal identity-size Mitchell
    ResamplingProgram progH=build_resampling_program(16,0,16,16,0.0,0.5);
    // KGaussResize: same-size gaussian, crop = size + 0.0001 (upstream), p = 30 default
    const int GH_H=12;
    ResamplingProgram gV9=build_gaussian_program(GH_H,0,GH_H+0.0001,GH_H,30.0);
    ResamplingProgram gH9=build_gaussian_program(16,0,16.0001,16,30.0);
    ResamplingProgram gV8=build_gaussian_program(GH_H,0,GH_H,GH_H,30.0);
    ResamplingProgram gH8=build_gaussian_program(16,0,16,16,30.0);
    // p edges + clamp probes (tables only; runner asserts -5 == 0.1, 250 == 100)
    ResamplingProgram gV9_p01=build_gaussian_program(GH_H,0,GH_H+0.0001,GH_H,0.1);
    ResamplingProgram gV9_p100=build_gaussian_program(GH_H,0,GH_H+0.0001,GH_H,100.0);
    ResamplingProgram gV9_neg5=build_gaussian_program(GH_H,0,GH_H+0.0001,GH_H,-5.0);
    ResamplingProgram gV9_p250=build_gaussian_program(GH_H,0,GH_H+0.0001,GH_H,250.0);
    if(bits==8){
        if(!run_all<uint8_t>(inDir,outDir,progV,progH,5,gV9,gH9,gV8,gH8,GH_H))return 1;
        {FILE* f=fopen((outDir+"/prog_offset.txt").c_str(),"w");for(int i:progV.offset)fprintf(f,"%d\n",i);fclose(f);
         FILE* g=fopen((outDir+"/prog_coef.txt").c_str(),"w");for(double c:progV.coef)fprintf(g,"%.17g\n",c);fclose(g);}
        write_gprog(outDir,"v9",gV9); write_gprog(outDir,"h9",gH9);
        write_gprog(outDir,"v8",gV8); write_gprog(outDir,"h8",gH8);
        write_gprog(outDir,"v9_p01",gV9_p01); write_gprog(outDir,"v9_p100",gV9_p100);
        write_gprog(outDir,"v9_neg5",gV9_neg5); write_gprog(outDir,"v9_p250",gV9_p250);
        fprintf(stderr,"bitdepth 8 done\n");
    } else {
        if(!run_all<uint16_t>(inDir,outDir,progV,progH,5,gV9,gH9,gV8,gH8,GH_H))return 1;
        fprintf(stderr,"bitdepth 16 done\n");
    }
    return 0;
}
