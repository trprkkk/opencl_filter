/* CPU mirror of the NNEDI3 prescreener kernel in
 * src/opencl/nnedi3/kernels/nnedi3_prescreen.cl (kl_prescreening twin,
 * NNEDI3/nnedi3/nnedi3_kernel.cu @01931aa).
 *
 * Emulates the whole 32x16 group grid serially: the int neighbourhood dot
 * product is exact, the float tail is unfused float32 per operation (build
 * with -ffp-contract=off), and the workNN compaction reproduces the
 * exclusive add-scan over tid order (upstream dev_scan then `idx -= num`).
 *
 * Usage: nnedi3_prescreen_ref <in> <out>
 *   All ints on one line (floats as bit patterns).  Mode:
 *     P w4 h refpitch4 dstpitch4 val_min val_max
 *       256 ws  28 wfbits  nS refpixels(nS)
 *   Output, in order:
 *     - dst dense (w4*4 * h), -1 where the kernel wrote nothing
 *     - numblocks[nb] for nb = nblocks(w4,32)*nblocks(h,16) groups
 *     - per group: the defined workNN prefix, 2 ints (x,y) per entry
 *       (entries beyond numblocks[bid] are undefined by design, skipped)
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace std;

static float bits2f(int b){ float f; memcpy(&f,&b,4); return f; }
static int nblocks(int n,int d){ return (n+d-1)/d; }

enum { PRE_BLOCK_W = 32, PRE_BLOCK_H = 16, PRE_BLOCK_N = 512 };

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: nnedi3_prescreen_ref <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> t; int x;
    while(f&&fscanf(f,"%d",&x)==1) t.push_back(x);
    if(f)fclose(f);
    size_t p=0; auto rd=[&](){return t[p++];};
    char m=(char)rd();
    if(m!='P'){fprintf(stderr,"unknown mode %c\n",m);return 2;}

    int w4=rd(),h=rd(),rp4=rd(),dp4=rd(),val_min=rd(),val_max=rd();
    int nW=rd(); vector<int> ws(nW); for(int&v:ws)v=rd();
    int nF=rd(); vector<float> wf(nF); for(float&v:wf)v=bits2f(rd());
    int nS=rd(); vector<int> ref(nS); for(int&v:ref)v=rd();

    int gx=nblocks(w4,PRE_BLOCK_W), gy=nblocks(h,PRE_BLOCK_H), nb=gx*gy;
    vector<int> dst((size_t)dp4*4*h,-1);
    vector<int> numb(nb,-1);
    vector<vector<int>> work(nb);

    for(int by=0;by<gy;by++)for(int bx=0;bx<gx;bx++){
        int bid=bx+by*gx;
        vector<int> num(PRE_BLOCK_N,0);
        vector<float> res((size_t)PRE_BLOCK_N*4,1.0f);

        for(int tid=0;tid<PRE_BLOCK_N;tid++){
            int tx=tid%PRE_BLOCK_W, ty=tid/PRE_BLOCK_W;
            int xbase=tx+bx*PRE_BLOCK_W, ybase=ty+by*PRE_BLOCK_H;
            if(xbase<w4&&ybase<h){
                int sum[4]={0,0,0,0};
                for(int y=0;y<4;y++)for(int xx=0;xx<5;xx++){
                    size_t vb=(size_t)((xx+xbase)+(y+ybase)*rp4)*4;
                    int v[4]; for(int k=0;k<4;k++)v[k]=ref[vb+k];
                    if(xx==0){
                        int a=(0+y*16)*4,b=(1+y*16)*4;
                        for(int c=0;c<4;c++){sum[c]+=ws[a+c]*v[2];sum[c]+=ws[b+c]*v[3];}
                    } else if(xx<4){
                        int a=((xx*4-2)+y*16)*4,b=((xx*4-1)+y*16)*4;
                        int cc=((xx*4+0)+y*16)*4,d=((xx*4+1)+y*16)*4;
                        for(int c=0;c<4;c++){
                            sum[c]+=ws[a+c]*v[0]; sum[c]+=ws[b+c]*v[1];
                            sum[c]+=ws[cc+c]*v[2];sum[c]+=ws[d+c]*v[3];
                        }
                    } else {
                        int a=(14+y*16)*4,b=(15+y*16)*4;
                        for(int c=0;c<4;c++){sum[c]+=ws[a+c]*v[0];sum[c]+=ws[b+c]*v[1];}
                    }
                }
                float val[4];
                for(int c=0;c<4;c++){
                    float tt=(float)sum[c]*wf[0*4+c]+wf[1*4+c];
                    val[c]=tt/(fabsf(tt)+1.0f);
                }
                for(int c=0;c<4;c++){
                    float s=0.0f;
                    s+=wf[2*4+c]*val[0];
                    s+=wf[3*4+c]*val[1];
                    s+=wf[4*4+c]*val[2];
                    s+=wf[5*4+c]*val[3];
                    res[(size_t)tid*4+c]=s+wf[6*4+c];
                }
            }
            int n=0; for(int c=0;c<4;c++) if(res[(size_t)tid*4+c]<=0.0f) ++n;
            num[tid]=n;
        }

        int idx=0;
        for(int tid=0;tid<PRE_BLOCK_N;tid++){
            int tx=tid%PRE_BLOCK_W, ty=tid/PRE_BLOCK_W;
            for(int c=0;c<4;c++) if(res[(size_t)tid*4+c]<=0.0f){
                work[bid].push_back(tx*4+c);
                work[bid].push_back(ty);
                ++idx;
            }
            if(tid==PRE_BLOCK_N-1) numb[bid]=idx;
        }

        for(int tid=0;tid<PRE_BLOCK_N;tid++){
            int tx=tid%PRE_BLOCK_W, ty=tid/PRE_BLOCK_W;
            int xbase=tx+bx*PRE_BLOCK_W, ybase=ty+by*PRE_BLOCK_H;
            if(num[tid]<4&&xbase<w4&&ybase<h){
                size_t b0=(size_t)((xbase+2)+(ybase+0)*rp4)*4;
                size_t b1=(size_t)((xbase+2)+(ybase+1)*rp4)*4;
                size_t b2=(size_t)((xbase+2)+(ybase+2)*rp4)*4;
                size_t b3=(size_t)((xbase+2)+(ybase+3)*rp4)*4;
                size_t dout=(size_t)(xbase+ybase*dp4)*4;
                for(int k=0;k<4;k++){
                    int s3p=ref[b0+k],s2=ref[b1+k],s4=ref[b2+k],s6=ref[b3+k];
                    int tmp=(((s2+s4)*19-(s3p+s6)*3+16)>>5);
                    if(tmp<val_min)tmp=val_min; if(tmp>val_max)tmp=val_max;
                    dst[dout+k]=tmp;
                }
            }
        }
    }

    FILE* o=fopen(argv[2],"w");
    for(int yy=0;yy<h;yy++)for(int xx=0;xx<w4*4;xx++)
        fprintf(o,"%d\n",dst[(size_t)xx+(size_t)yy*dp4*4]);
    for(int i=0;i<nb;i++)fprintf(o,"%d\n",numb[i]);
    for(int i=0;i<nb;i++)for(int v:work[i])fprintf(o,"%d\n",v);
    fclose(o);
    return 0;
}
