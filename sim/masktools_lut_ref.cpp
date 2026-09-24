/* CPU mirror of the five masktools CUDA kernels in
 * src/opencl/masktools/kernels/masktools_lut.cl (kl_fill / kl_copy /
 * kl_lut_x / kl_lut_xy / kl_lut_xyz twins; masktools @24ba826).
 *
 * Integer / bit-pattern pass-through; the LUT index arithmetic (including
 * upstream's 16-bit shift-8 + mask-255 collapse) is what matters here.
 * Large index spaces use a PROCEDURAL table (nLut == 0) so the full
 * 8-bit xyz range can be exercised without shipping 16M entries.
 *
 * Usage: masktools_lut_ref <in> <out>
 *   All ints on one line (floats as bit patterns).  Modes:
 *     F fill   : F v width4 height pitch4  -> pitch4*4*height plane (-1 fill)
 *     G fillf32: G vbits width4 height pitch4 -> same, float bits
 *     C copy   : C width4 height dp4 sp4  nS src(nS)
 *                -> dp4*4*height bytes (-1 where untouched)
 *     X lut_x  : X px pitch4 width4 height mask  nL lut(nL)  nS src(nS)
 *     Y lut_xy : Y px pitch4 width4 height mask bits  nL lut(nL)
 *                nS src0(nS) src1(nS)
 *     Z lut_xyz: Z px pitch4 width4 height mask bits  nL lut(nL)
 *                nS src0(nS) src1(nS) src2(nS)
 *   px is 1 or 2 (pixel byte width: selects the mask branch, like
 *   upstream's sizeof(pixel_t) == 1 test).  LUT outputs are the
 *   pitch4*4*height plane with -1 where the kernel wrote nothing.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace std;

/* procedural stand-in table, shared verbatim with the Python golden */
static int proc_lut(int i, int px){
    unsigned h = (unsigned)i * 2654435761u;
    h ^= h >> 15;
    return (int)(h & (px == 1 ? 0xFFu : 0xFFFFu));
}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: masktools_lut_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    char m=(char)rd();
    vector<int> out;
    auto read=[&](int n){ vector<int> a(n); for(int&i:a)i=rd(); return a; };

    if(m=='F'||m=='G'){
        int v=rd(),w4=rd(),h=rd(),pitch4=rd();
        vector<int> dst((size_t)pitch4*4*h,-1);
        for(int yy=0;yy<h;yy++)for(int xx=0;xx<w4;xx++){
            size_t o=(size_t)(xx+yy*pitch4)*4;
            for(int k=0;k<4;k++)dst[o+k]=v;
        }
        out=dst;
    } else if(m=='C'){
        int w4=rd(),h=rd(),dp4=rd(),sp4=rd();
        int nS=rd(); vector<int> src=read(nS);
        vector<int> dst((size_t)dp4*4*h,-1);
        for(int yy=0;yy<h;yy++)for(int xx=0;xx<w4;xx++){
            size_t d=(size_t)(xx+yy*dp4)*4, s=(size_t)(xx+yy*sp4)*4;
            for(int k=0;k<4;k++)dst[d+k]=src[s+k];
        }
        out=dst;
    } else if(m=='X'||m=='Y'||m=='Z'){
        int px=rd(),pitch4=rd(),w4=rd(),h=rd(),mask=rd();
        int bits=(m=='X')?0:rd();
        int nL=rd(); vector<int> lut=read(nL);
        int nS=rd(); vector<int> s0=read(nS);
        vector<int> s1,s2;
        if(m!='X') s1=read(nS);
        if(m=='Z') s2=read(nS);
        vector<int> dst((size_t)pitch4*4*h,-1);
        for(int yy=0;yy<h;yy++)for(int xx=0;xx<w4;xx++){
            size_t o=(size_t)(xx+yy*pitch4)*4;
            for(int k=0;k<4;k++){
                int idx;
                if(m=='X') idx=s0[o+k];
                else if(m=='Y') idx=(s0[o+k]<<bits)+s1[o+k];
                else idx=(s0[o+k]<<(bits*2))+(s1[o+k]<<bits)+s2[o+k];
                if(px!=1) idx&=mask;
                dst[o+k]=nL?lut[idx]:proc_lut(idx,px);
            }
        }
        out=dst;
    } else {
        fprintf(stderr,"unknown mode %c\n",m); return 2;
    }

    FILE* o=fopen(argv[2],"w");
    for(int v:out)fprintf(o,"%d\n",v);
    fclose(o);
    return 0;
}
