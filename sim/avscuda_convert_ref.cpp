/* CPU mirror of the ten AvsCUDA Convert kernels in
 * src/opencl/avscuda/kernels/avscuda_convert.cl (kl_convert_* twins,
 * AvsCUDA/filters/Convert.cu).  Dither tables are verbatim copies of
 * c_dither2/4/6/8.  Float paths use unfused float32 (build with
 * -ffp-contract=off); from_float spells out the rgy_util.h clamp macro
 * (NaN -> MAX_VAL).
 *
 * Usage: avscuda_convert_ref <in> <out>
 *   All ints on one line.  Modes (nS = sp*h; dual pitch):
 *     A dither_u8   : A w h dp sp shift tgt_bits  nS src(nS) -> w*h u8
 *     B dither_u16  : B w h dp sp shift tgt_bits  nS src(nS) -> w*h u16
 *     C nodither_u8 : C w h dp sp shift tgt_bits  nS src(nS) -> w*h u8
 *     D nodither_u16: D w h dp sp shift tgt_bits  nS src(nS) -> w*h u16
 *     E higher_u8   : E w h dp sp shift tgt_bits  nS src(nS) -> w*h u16
 *     F higher_u16  : F w h dp sp shift tgt_bits  nS src(nS) -> w*h u16
 *     G fromf_u8    : G w h dp sp tgt_bits chroma  nS srcbits(nS) -> w*h u8
 *     H fromf_u16   : H w h dp sp tgt_bits chroma  nS srcbits(nS) -> w*h u16
 *     I tof_u8      : I w h dp sp src_bits chroma  nS src(nS) -> w*h f32 bits
 *     J tof_u16     : J w h dp sp src_bits chroma  nS src(nS) -> w*h f32 bits
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace std;

static float bits2f(int b){ float f; memcpy(&f,&b,4); return f; }
static int f2bits(float f){ int b; memcpy(&b,&f,4); return b; }

static const unsigned char c_dither2[2][2] = { { 0, 2 }, { 3, 1 } };
static const unsigned char c_dither4[4][4] = {
    { 0,  8,  2, 10 }, { 12,  4, 14,  6 }, { 3, 11,  1,  9 }, { 15,  7, 13,  5 }
};
static const unsigned char c_dither6[8][8] = {
    { 0, 32,  8, 40,  2, 34, 10, 42 }, { 48, 16, 56, 24, 50, 18, 58, 26 },
    { 12, 44,  4, 36, 14, 46,  6, 38 }, { 60, 28, 52, 20, 62, 30, 54, 22 },
    { 3, 35, 11, 43,  1, 33,  9, 41 }, { 51, 19, 59, 27, 49, 17, 57, 25 },
    { 15, 47,  7, 39, 13, 45,  5, 37 }, { 63, 31, 55, 23, 61, 29, 53, 21 }
};
static const unsigned char c_dither8[16][16] = {
    { 0,192, 48,240, 12,204, 60,252,  3,195, 51,243, 15,207, 63,255 },
    { 128, 64,176,112,140, 76,188,124,131, 67,179,115,143, 79,191,127 },
    { 32,224, 16,208, 44,236, 28,220, 35,227, 19,211, 47,239, 31,223 },
    { 160, 96,144, 80,172,108,156, 92,163, 99,147, 83,175,111,159, 95 },
    { 8,200, 56,248,  4,196, 52,244, 11,203, 59,251,  7,199, 55,247 },
    { 136, 72,184,120,132, 68,180,116,139, 75,187,123,135, 71,183,119 },
    { 40,232, 24,216, 36,228, 20,212, 43,235, 27,219, 39,231, 23,215 },
    { 168,104,152, 88,164,100,148, 84,171,107,155, 91,167,103,151, 87 },
    { 2,194, 50,242, 14,206, 62,254,  1,193, 49,241, 13,205, 61,253 },
    { 130, 66,178,114,142, 78,190,126,129, 65,177,113,141, 77,189,125 },
    { 34,226, 18,210, 46,238, 30,222, 33,225, 17,209, 45,237, 29,221 },
    { 162, 98,146, 82,174,110,158, 94,161, 97,145, 81,173,109,157, 93 },
    { 10,202, 58,250,  6,198, 54,246,  9,201, 57,249,  5,197, 53,245 },
    { 138, 74,186,122,134, 70,182,118,137, 73,185,121,133, 69,181,117 },
    { 42,234, 26,218, 38,230, 22,214, 41,233, 25,217, 37,229, 21,213 },
    { 170,106,154, 90,166,102,150, 86,169,105,153, 89,165,101,149, 85 }
};

static int dither_get(int shift, int x, int y) {
    int m = (1 << (shift >> 1)) - 1;
    int xx = x & m, yy = y & m;
    if (shift == 2) return c_dither2[yy][xx];
    if (shift == 4) return c_dither4[yy][xx];
    if (shift == 6) return c_dither6[yy][xx];
    if (shift == 8) return c_dither8[yy][xx];
    return 0;
}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: avscuda_convert_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    char m=(char)rd();
    vector<int> out;

    auto read=[&](int n){ vector<int> a(n); for(int&i:a)i=rd(); return a; };

    if(m=='A' || m=='B' || m=='C' || m=='D' || m=='E' || m=='F'){
        int w=rd(),h=rd(),dp=rd(),sp=rd(),shift=rd(),tgt=rd();
        (void)dp;
        int vmax=(1<<tgt)-1;
        int nS=rd(); vector<int> src=read(nS);
        for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++){
            int v=src[(size_t)(xx+yy*sp)], tmp;
            if(m=='A'||m=='B') tmp=(v + dither_get(shift,xx,yy)) >> shift;
            else if(m=='C'||m=='D') tmp=v >> shift;
            else tmp=v << shift;
            if(tmp>vmax)tmp=vmax;
            out.push_back(tmp);
        }
    } else if(m=='G' || m=='H'){
        int w=rd(),h=rd(),dp=rd(),sp=rd(),tgt=rd(),chroma=rd();
        (void)dp;
        float max_val=(float)(255<<(tgt-8)), half=(float)(128<<(tgt-8));
        int nS=rd(); vector<int> src=read(nS);
        for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++){
            float s=bits2f(src[(size_t)(xx+yy*sp)]);
            float tmp=chroma ? (s*max_val + half + 0.5f) : (s*max_val + 0.5f);
            float c=(tmp<=max_val)?((tmp>=0.0f)?tmp:0.0f):max_val;
            out.push_back((int)c);
        }
    } else if(m=='I' || m=='J'){
        int w=rd(),h=rd(),dp=rd(),sp=rd(),sbits=rd(),chroma=rd();
        (void)dp;
        float max_val=(float)(255<<(sbits-8));
        float factor=1.0f/max_val, half=(float)(128<<(sbits-8));
        int nS=rd(); vector<int> src=read(nS);
        for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++){
            float v=(float)src[(size_t)(xx+yy*sp)];
            out.push_back(f2bits(chroma ? ((v-half)*factor) : (v*factor)));
        }
    } else { fprintf(stderr,"bad mode %c\n",m); return 2; }
    FILE* o=fopen(argv[2],"w");
    for(int v:out)fprintf(o,"%d\n",v);
    fclose(o);
    return 0;
}
