/* CPU mirror of the four KFM MergeStatic.cu kernels in
 * src/opencl/kfm/kernels/kfm_mergestatic.cl, faithful to the CPU twins in
 * KFM/MergeStatic.cu (cpu_compare_frames / cpu_min_frames / cpu_merge_static
 * are exact twins; cpu_and_coefs is not defined upstream, so mode A is an
 * independent transliteration of the CUDA kl_and_coefs, kept float32 with no
 * FMA contraction — compile with -ffp-contract=off so the Python golden
 * matches bit-for-bit).
 *
 * Usage: kfm_mergestatic_ref <in> <out>
 *   All ints space-separated on one line.  Each mode has the layout:
 *     C  (compare_frames, 5 frames):  width height pitch  n0 n1 n2 n3 n4
 *                                     src0..src4 (each pitch*height)
 *     N  (min_frames, 3 frames):      width height pitch  n0 n1 n2  src0..src2
 *     A  (and_coefs):                 width height pitch  invcombe invdiff
 *                                     nD nF  dstp(...) diffp(...)
 *     M  (merge_static):              width height pitch  nS nT nF
 *                                     src60(...) src30(...) flag(...)
 *   Output: one int per logical (width x height) pixel, row-major.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace std;
static float fclampf(float v,float a,float b){return v<a?a:(v>b?b:v);}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: kfm_mergestatic_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    int mode=rd(); char m=(char)mode;
    int width=rd(),height=rd(),pitch=rd();
    vector<int> out((size_t)width*height);

    auto read=[&](int n){ vector<int> a(n); for(int&i:a)i=rd(); return a; };
    auto put=[&](int xx,int yy,int v){ out[(size_t)yy*width+xx]=v; };

    if(m=='C'){
        int n0=rd(),n1=rd(),n2=rd(),n3=rd(),n4=rd();
        vector<int> s0=read(n0),s1=read(n1),s2=read(n2),s3=read(n3),s4=read(n4);
        for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++){
            int o=yy*pitch+xx;
            int mn=s0[o],mx=s0[o];
            int v1=s1[o]; if(v1<mn)mn=v1; if(v1>mx)mx=v1;
            int v2=s2[o]; if(v2<mn)mn=v2; if(v2>mx)mx=v2;
            int v3=s3[o]; if(v3<mn)mn=v3; if(v3>mx)mx=v3;
            int v4=s4[o]; if(v4<mn)mn=v4; if(v4>mx)mx=v4;
            put(xx,yy,mx-mn);
        }
    } else if(m=='N'){
        int n0=rd(),n1=rd(),n2=rd();
        vector<int> s0=read(n0),s1=read(n1),s2=read(n2);
        for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++){
            int o=yy*pitch+xx;
            int mn=s0[o];
            int v1=s1[o]; if(v1<mn)mn=v1;
            int v2=s2[o]; if(v2<mn)mn=v2;
            put(xx,yy,mn);
        }
    } else if(m=='A'){
        float invcombe; { unsigned u=rd(); float tmp; memcpy(&tmp,&u,4); invcombe=tmp; }
        float invdiff;  { unsigned u=rd(); float tmp; memcpy(&tmp,&u,4); invdiff=tmp; }
        int nD=rd(),nF=rd();
        vector<int> dstp=read(nD),diffp=read(nF);
        for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++){
            int o=yy*pitch+xx;
            float combe=fclampf((float)dstp[o]*invcombe+(-1.0f),-0.5f,0.5f);
            float diffc=fclampf((float)diffp[o]*(-invdiff)+1.0f,-0.5f,0.5f);
            float s=combe+diffc; if(s<0.0f)s=0.0f;
            float tmp=s*128.0f+0.5f;
            put(xx,yy,(int)tmp);
        }
    } else if(m=='M'){
        int nS=rd(),nT=rd(),nF=rd();
        vector<int> s60=read(nS),s30=read(nT),flag=read(nF);
        for(int yy=0;yy<height;yy++)for(int xx=0;xx<width;xx++){
            int o=yy*pitch+xx;
            int coef=flag[o],v30=s30[o],v60=s60[o];
            int tmp=(coef*v30+(128-coef)*v60+64)>>7;
            put(xx,yy,tmp);
        }
    } else { fprintf(stderr,"bad mode %c\n",m); return 2; }
    FILE* o=fopen(argv[2],"w");
    for(int v:out)fprintf(o,"%d\n",v);
    fclose(o);
    return 0;
}
