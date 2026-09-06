/* CPU mirror of kl_mean_global_mv (kt_mean_global_mv in
 * src/opencl/ktgmc/kernels/ktgmc_motion.cl). Independent re-implementation
 * for cross-checking against the Python golden.
 *
 * Usage: ktgmc_mean_ref <infile> <outfile>
 * Input file (space-separated ints):
 *   nRows
 *   for each row: medianx mediany nVec  then (nVec) pairs "vx vy"
 * Output file: for each row, one line "meanx meany".
 * Each row reduces its vectors: num = # vectors with |vx-medianx|<6 &&
 * |vy-mediany|<6; result row = (2*sum_vx/num, 2*sum_vy/num), C truncating div.
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace std;

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: mean <in> <out>\n");return 2;}
    FILE* f=fopen(argv[1],"r"); vector<int> n; int x;
    while(f&&fscanf(f,"%d",&x)==1) n.push_back(x);
    if(f)fclose(f);
    int p=0, nRows=n[p++];
    FILE* o=fopen(argv[2],"w");
    for(int r=0;r<nRows;r++){
        int medianx=n[p++],mediany=n[p++],nVec=n[p++];
        long long sx=0, sy=0; int num=0;
        for(int i=0;i<nVec;i++){
            int vx=n[p++],vy=n[p++];
            int dx=vx-medianx; if(dx<0)dx=-dx;
            int dy=vy-mediany; if(dy<0)dy=-dy;
            if(dx<6&&dy<6){sx+=vx;sy+=vy;num++;}
        }
        fprintf(o,"%d %d\n",(int)((2*sx)/num),(int)((2*sy)/num));
    }
    if(o)fclose(o);
    return 0;
}
