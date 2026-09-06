/* CPU mirror of kl_load_mv / kl_store_mv / kl_init_const_vec (kt_load_mv,
 * kt_store_mv, kt_init_const_vec in src/opencl/ktgmc/kernels/ktgmc_motion.cl).
 *
 * Usage: ktgmc_mvio_ref <mode> ...   (results printed to stdout)
 *   mode 0 (load):   0 nBlk  then nBlk triplets "x y sad"
 *                    -> prints nBlk "vx vy sad" (int2 vector + sad split)
 *   mode 1 (store):  1 nBlk  then nBlk triplets "vx vy sad"
 *                    -> prints nBlk "x y sad" (int3 VECTOR recombined)
 *   mode 2 (init):   2 nRows vectorsPitch gx gy nPel
 *                    -> prints nRows "s0x s0y s1x s1y" (slots -2 and -1)
 *   mode 3 (batch):  3 nBlk then nBlk triplets "x y sad"
 *                    -> prints nBlk "x y sad" (VECTOR split to vec/sad + out copy)
 */
#include <cstdio>
#include <cstdlib>
using namespace std;
int main(int argc,char**argv){
    int mode=0,n=0;
    if(argc<2){return 2;}
    mode=atoi(argv[1]);
    if(mode==0||mode==1||mode==3){
        if(argc<3)return 2; n=atoi(argv[2]);
        for(int i=0;i<n;i++){
            int x,y,s; if(scanf("%d%d%d",&x,&y,&s)!=3)return 2;
            if(mode==3) printf("%d %d %d\n",x,y,s);   /* batch: split + out copy */
            else if(mode==0) printf("%d %d %d\n",x,y,s); /* pass x,y to vec; s to sad */
            else        printf("%d %d %d\n",x,y,s);   /* recombine x,y,sad */
        }
    }else if(mode==2){
        int nRows,pitch,gx,gy,nPel;
        if(argc<7)return 2;
        nRows=atoi(argv[2]);pitch=atoi(argv[3]);gx=atoi(argv[4]);
        gy=atoi(argv[5]);nPel=atoi(argv[6]);
        for(int r=0;r<nRows;r++)
            printf("0 0 %d %d\n",gx*nPel,gy*nPel);
    }
    return 0;
}
