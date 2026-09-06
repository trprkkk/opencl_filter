/* CPU mirror of kl_interpolate_prediction (kt_interpolate_prediction in
 * src/opencl/ktgmc/kernels/ktgmc_motion.cl). Independent re-implementation for
 * cross-checking against the Python golden.
 *
 * Usage: ktgmc_ip_ref <infile> <outfile>
 * Input file (space-separated ints):
 *   nSrcBlkX nSrcBlkY nDstBlkX nDstBlkY normFactor normov atotal aodd aeven
 *   then (nSrcBlkX*nSrcBlkY) triplets "vx vy sad"  (row-major src)
 * Output file: for each of (nDstBlkX*nDstBlkY) fine blocks, one line "vx vy sad"
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
using namespace std;

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: ip <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> n; int x;
    while(f&&fscanf(f,"%d",&x)==1) n.push_back(x);
    if(f)fclose(f);
    int p=0;
    int nSX=n[p++],nSY=n[p++],nDX=n[p++],nDY=n[p++],normFactor=n[p++],normov=n[p++],atotal=n[p++],aodd=n[p++],aeven=n[p++];
    int nSrc=nSX*nSY;
    vector<int> sx(nSrc),sy(nSrc),ss(nSrc);
    for(int i=0;i<nSrc;++i){sx[i]=n[p++];sy[i]=n[p++];ss[i]=n[p++];}
    FILE* o=fopen(argv[2],"w");
    for(int y=0;y<nDY;++y)for(int X=0;X<nDX;++X){
        int i=X,j=y;
        if(i>=2*nSX)i=2*nSX-1;
        if(j>=2*nSY)j=2*nSY-1;
        int offy=-1+2*(j%2),offx=-1+2*(i%2),ip2=i>>1,jp2=j>>1;
        int v1x,v1y,v2x,v2y,v3x,v3y,v4x,v4y,sad1,sad2,sad3,sad4;
        int A=ip2+jp2*nSX;
        if((i==0)||(i>=2*nSX-1)){
            if((j==0)||(j>=2*nSY-1)){
                v1x=v2x=v3x=v4x=sx[A];v1y=v2y=v3y=v4y=sy[A];
                sad1=sad2=sad3=sad4=ss[A];
            }else{
                int B=ip2+(jp2+offy)*nSX;
                v1x=v2x=sx[A];v1y=v2y=sy[A];v3x=v4x=sx[B];v3y=v4y=sy[B];
                sad1=sad2=ss[A];sad3=sad4=ss[B];
            }
        }else if((j==0)||(j>=2*nSY-1)){
            int B=ip2+offx+jp2*nSX;
            v1x=v2x=sx[A];v1y=v2y=sy[A];v3x=v4x=sx[B];v3y=v4y=sy[B];
            sad1=sad2=ss[A];sad3=sad4=ss[B];
        }else{
            int B=ip2+offx+jp2*nSX,C=ip2+(jp2+offy)*nSX,D=ip2+offx+(jp2+offy)*nSX;
            v1x=sx[A];v1y=sy[A];v2x=sx[B];v2y=sy[B];v3x=sx[C];v3y=sy[C];v4x=sx[D];v4y=sy[D];
            sad1=ss[A];sad2=ss[B];sad3=ss[C];sad4=ss[D];
        }
        int ax1=(offx>0)?aodd:aeven,ax2=atotal-ax1,ay1=(offy>0)?aodd:aeven,ay2=atotal-ay1;
        int a11=ax1*ay1,a12=ax1*ay2,a21=ax2*ay1,a22=ax2*ay2;
        int vx=(a11*v1x+a21*v2x+a12*v3x+a22*v4x)/normov;
        int vy=(a11*v1y+a21*v2y+a12*v3y+a22*v4y)/normov;
        int ts=(a11*sad1+a21*sad2+a12*sad3+a22*sad4)/normov;
        if(normFactor>0){vx>>=normFactor;vy>>=normFactor;}else{vx<<=-normFactor;vy<<=-normFactor;}
        fprintf(o,"%d %d %d\n",vx,vy,ts>>4);
    }
    if(o)fclose(o);
    return 0;
}
