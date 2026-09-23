/* CPU mirror of the four AvsCUDA Invert kernels in
 * src/opencl/avscuda/kernels/avscuda_filters.cl (kl_invert_plane /
 * kl_invert_rgb twins, AvsCUDA/filters/Filters.cu).  In-place upstream;
 * the mirror reads src and emits the inverted plane (same values for
 * these elementwise ops).  Widths are exact pixels/elements (no word
 * overhang — see the .cl header), so non-multiple-of-4 widths are valid.
 *
 * Usage: avscuda_filters_ref <in> <out>
 *   All ints on one line.  Modes:
 *     A inv_u8 : A w h pitch mask0  nP src(nP)
 *                -> output w*h of v ^ byte(x&3) of mask0
 *     B inv_u16: B w h pitch mask0 mask1  nP src(nP)
 *                -> output w*h of v ^ 16-bit lane ((x>>1)&1 selects
 *                   mask0/mask1, (x&1) selects the half)
 *     C inv_f32: C w h pitch  nP src(nP, float bit patterns)
 *                -> output w*h bit patterns of 1.0f - x
 *     D inv_rgb: D w h el_pitch bM gM rM maxv  nP src(nP)
 *                -> output w*h*3 of (v^mask_c) & maxv per channel element
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace std;

static float bits2f(int b){ float f; memcpy(&f,&b,4); return f; }
static int f2bits(float f){ int b; memcpy(&b,&f,4); return b; }

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: avscuda_filters_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    char m=(char)rd();
    vector<int> out;

    auto read=[&](int n){ vector<int> a(n); for(int&i:a)i=rd(); return a; };

    if(m=='A'){
        int w=rd(),h=rd(),pitch=rd();
        unsigned mask0=(unsigned)rd();
        int nP=rd(); vector<int> src=read(nP);
        for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++){
            unsigned lane=(mask0 >> (8u*((unsigned)xx & 3u))) & 0xFFu;
            out.push_back(src[(size_t)(xx+yy*pitch)] ^ (int)lane);
        }
    } else if(m=='B'){
        int w=rd(),h=rd(),pitch=rd();
        unsigned mask0=(unsigned)rd(), mask1=(unsigned)rd();
        int nP=rd(); vector<int> src=read(nP);
        for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++){
            unsigned ux=(unsigned)xx;
            unsigned m32=((((ux>>1)&1u)!=0u)?mask1:mask0);
            unsigned lane=(m32 >> (16u*(ux&1u))) & 0xFFFFu;
            out.push_back(src[(size_t)(xx+yy*pitch)] ^ (int)lane);
        }
    } else if(m=='C'){
        int w=rd(),h=rd(),pitch=rd();
        int nP=rd(); vector<int> src=read(nP);
        for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++){
            float v=bits2f(src[(size_t)(xx+yy*pitch)]);
            out.push_back(f2bits(1.0f - v));
        }
    } else if(m=='D'){
        int w=rd(),h=rd(),elp=rd(),bM=rd(),gM=rd(),rM=rd(),maxv=rd();
        int nP=rd(); vector<int> src=read(nP);
        const int mk[3]={bM,gM,rM};
        for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++)
            for(int c=0;c<3;c++)
                out.push_back((src[(size_t)(3*xx+c+yy*elp)] ^ mk[c]) & maxv);
    } else { fprintf(stderr,"bad mode %c\n",m); return 2; }
    FILE* o=fopen(argv[2],"w");
    for(int v:out)fprintf(o,"%d\n",v);
    fclose(o);
    return 0;
}
