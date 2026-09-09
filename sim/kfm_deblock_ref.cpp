/* CPU mirror of the KFM KDeblock core kernel in src/opencl/kfm/kernels/
 * kfm_deblock.cl, a faithful scalar transcription of the CUDA device kernel
 * kl_deblock (KFM/Deblock.cu).  Float32, no FMA; compile -ffp-contract=off.
 * (The as-shipped CPU fallback cpu_deblock has a `thresh<=0` multiply-by-64
 * identity shortcut that kl_deblock does NOT have; this mirror follows the
 * device kernel / the .cl: always run DCT->hardthresh->IDCT.)
 *
 * Usage: kfm_deblock_ref <in> <out>
 *   in:  D sw sh bh out_pitch qp_pitch count_minus_1 shift maxv
 *        strength_i thresh_a_i thresh_b_i       (float bits)
 *        bw   nqp qp[...]   nsrc src[...]       (src is sw-wide, sh tall)
 *        out buffer region compared is out_pitch*(32*bh+8); mirror inits -1.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace std;

#define S1   0.19509032201612825f
#define C1   0.9807852804032304f
#define S3   0.5555702330196022f
#define C3   0.8314696123025452f
#define S2S6 1.3065629648763766f
#define S2C6 0.5411961001461971f
#define S2   1.4142135623730951f

static const int g_offx[127] = {
  0,0,4, 0,2,6,4, 0,5,2,7,4,1,6,3, 0,4,1,5,3,7,2,6,0,4,1,5,3,7,2,6,
  0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,
  0,4,0,4,2,6,2,6,0,4,0,4,2,6,2,6,1,5,1,5,3,7,3,7,1,5,1,5,3,7,3,7,
  0,4,0,4,2,6,2,6,0,4,0,4,2,6,2,6,1,5,1,5,3,7,3,7,1,5,1,5,3,7,3,7,
};
static const int g_offy[127] = {
  0,0,4, 0,2,4,6, 0,1,2,3,4,5,6,7, 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,
  0,2,4,6,1,3,5,7,0,2,4,6,1,3,5,7,0,2,4,6,1,3,5,7,0,2,4,6,1,3,5,7,
  0,4,4,0,2,6,6,2,2,6,6,2,0,4,4,0,1,5,5,1,3,7,7,3,3,7,7,3,1,5,5,1,
  1,5,5,1,3,7,7,3,3,7,7,3,1,5,5,1,0,4,4,0,2,6,6,2,2,6,6,2,0,4,4,0,
};

static float clampf(float v,float a,float b){return v<a?a:(v>b?b:v);}
static float qp_thresh(int qp,float ta,float tb){return clampf((float)qp*ta+tb,0.0f,(float)qp);}

static void dct8(float* d,int stride){
    float a0=d[7*stride]+d[0*stride];
    float a1=d[6*stride]+d[1*stride];
    float a2=d[5*stride]+d[2*stride];
    float a3=d[4*stride]+d[3*stride];
    float a4=d[3*stride]-d[4*stride];
    float a5=d[2*stride]-d[5*stride];
    float a6=d[1*stride]-d[6*stride];
    float a7=d[0*stride]-d[7*stride];
    float b0=a3+a0,b1=a2+a1,b2=a1-a2,b3=a0-a3;
    float b4=(S3-C3)*a7+C3*(a4+a7);
    float b5=(S1-C1)*a6+C1*(a5+a6);
    float b6=-(C1+S1)*a5+C1*(a5+a6);
    float b7=-(C3+S3)*a4+C3*(a4+a7);
    float c0=b1+b0,c1=b0-b1;
    float c2=(S2S6-S2C6)*b3+S2C6*(b2+b3);
    float c3=-(S2C6+S2S6)*b2+S2C6*(b2+b3);
    float c4=b6+b4,c5=b7-b5,c6=b4-b6,c7=b5+b7;
    float d4=c7-c4,d5=c5*S2,d6=c6*S2,d7=c4+c7;
    d[0*stride]=c0;d[4*stride]=c1;d[2*stride]=c2;d[6*stride]=c3;
    d[7*stride]=d4;d[3*stride]=d5;d[5*stride]=d6;d[1*stride]=d7;
}
static void idct8(float* d,int stride){
    float c0=d[0*stride],c1=d[4*stride],c2=d[2*stride],c3=d[6*stride];
    float d4=d[7*stride],d5=d[3*stride],d6=d[5*stride],d7=d[1*stride];
    float c4=d7-d4,c5=d5*S2,c6=d6*S2,c7=d4+d7;
    float b0=c1+c0,b1=c0-c1;
    float b2=-(S2C6+S2S6)*c3+S2C6*(c2+c3);
    float b3=(S2S6-S2C6)*c2+S2C6*(c2+c3);
    float b4=c6+c4,b5=c7-c5,b6=c4-c6,b7=c5+c7;
    float a0=b3+b0,a1=b2+b1,a2=b1-b2,a3=b0-b3;
    float a4=-(C3+S3)*b7+C3*(b4+b7);
    float a5=-(C1+S1)*b6+C1*(b5+b6);
    float a6=(S1-C1)*b5+C1*(b5+b6);
    float a7=(S3-C3)*b4+C3*(b4+b7);
    d[0*stride]=a7+a0;d[1*stride]=a6+a1;d[2*stride]=a5+a2;d[3*stride]=a4+a3;
    d[4*stride]=a3-a4;d[5*stride]=a2-a5;d[6*stride]=a1-a6;d[7*stride]=a0-a7;
}
static void dct8x8(float* d){for(int i=0;i<8;i++)dct8(d+i*8,1);for(int i=0;i<8;i++)dct8(d+i,8);}
static void idct8x8(float* d){for(int i=0;i<8;i++)idct8(d+i,8);for(int i=0;i<8;i++)idct8(d+i*8,1);}
static void hardthresh(float* d,float th){
    for(int i=1;i<64;i++){ if(d[i] < -th || d[i] > th) continue; d[i]=0.0f; }
}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: kfm_deblock_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    if((char)rd()!='D'){fprintf(stderr,"bad mode\n");return 2;}
    int sw=rd(),sh=rd(),bh=rd(),out_pitch=rd(),qp_pitch=rd();
    int count_minus_1=rd(),shift=rd(),maxv=rd();
    unsigned su=rd(),sta=rd(),stb=rd();
    float strength; memcpy(&strength,&su,4);
    float ta; memcpy(&ta,&sta,4);
    float tb; memcpy(&tb,&stb,4);
    int bw=rd(),nqp=rd();
    vector<int> qp(nqp); for(int&i:qp)i=rd();
    int nsrc=rd(); vector<int> src(nsrc); for(int&i:src)i=rd();

    int rows=32*bh+8;
    vector<int> out((size_t)out_pitch*rows,-1);
    int local_out[16][16];
    for(int by=0;by<bh;by++)for(int bx=0;bx<bw;bx++){
        for(int r=0;r<16;r++)for(int c=0;c<16;c++)local_out[r][c]=0;
        for(int ty=0;ty<=count_minus_1;ty++){
            int ox0=g_offx[count_minus_1+ty],oy0=g_offy[count_minus_1+ty];
            int ox=bx*8+ox0, oy=by*8+oy0;
            float d[64];
            for(int y=0;y<8;y++)for(int xx=0;xx<8;xx++)
                d[xx+y*8]=(float)src[(ox+xx)+(oy+y)*sw];
            int qpv=qp[bx+by*qp_pitch];
            float thresh=qp_thresh(qpv,ta,tb)*((1<<2)+strength)-1.0f;
            dct8x8(d); hardthresh(d,thresh); idct8x8(d);
            int half=(1<<shift)>>1;
            for(int y=0;y<8;y++)for(int xx=0;xx<8;xx++){
                int tmp=((int)(d[xx+y*8]+(float)half))>>shift;
                if(tmp<0)tmp=0;else if(tmp>maxv)tmp=maxv;
                local_out[oy0+y][ox0+xx]+=tmp;
            }
        }
        int offz=(bx&1)+(by&1)*2;
        int offx=bx*8, offy=(bh*offz+by)*8;
        for(int y=0;y<16;y++)for(int xx=0;xx<16;xx++)
            out[(offx+xx)+(offy+y)*out_pitch]=local_out[y][xx];
    }
    FILE* o=fopen(argv[2],"w");
    for(int v:out)fprintf(o,"%d\n",v);
    fclose(o);
    return 0;
}
