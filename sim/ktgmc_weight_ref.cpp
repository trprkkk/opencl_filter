/* CPU mirror of the degrain weight helpers (kt_degrain_weight / kt_norm_weights
 * in src/opencl/ktgmc/kernels/ktgmc_motion.cl). Independent re-implementation
 * for cross-checking against the Python golden.
 *
 * Reads test cases from stdin:
 *   line types:
 *     "w <thSAD> <blockSAD>"            -> print "w <result>"
 *     "n <delta> <binomial> <b0> <f0> ... <bk> <fk>"   (k = delta, so 2*delta ints)
 *        -> print "n <wsrc> <b0> <f0> ... "
 */
#include <cstdio>
#include <cstdlib>
using namespace std;

static int kt_degrain_weight(int thSAD, int blockSAD){
    if(thSAD<=blockSAD) return 0;
    float sqt=(float)thSAD*(float)thSAD;
    float sqb=(float)blockSAD*(float)blockSAD;
    return (int)(256.0f*(sqt-sqb)/(sqt+sqb));
}

// arrays sized 6
static int kt_norm_weights(int delta,int binomial,int*WRefB,int*WRefF){
    int WSrc=256;
    if(binomial){
        if(delta==1){WSrc*=2;}
        else if(delta==2){WSrc*=6;WRefB[0]*=4;WRefF[0]*=4;}
        else if(delta==3){WSrc*=20;WRefB[0]*=15;WRefF[0]*=15;WRefB[1]*=6;WRefF[1]*=6;}
        else if(delta==4){WSrc*=70;WRefB[0]*=56;WRefF[0]*=56;WRefB[1]*=28;WRefF[1]*=28;WRefB[2]*=8;WRefF[2]*=8;}
    }
    int WSum;
    if(delta==6)WSum=WRefB[0]+WRefF[0]+WSrc+WRefB[1]+WRefF[1]+WRefB[2]+WRefF[2]+WRefB[3]+WRefF[3]+WRefB[4]+WRefF[4]+WRefB[5]+WRefF[5]+1;
    else if(delta==5)WSum=WRefB[0]+WRefF[0]+WSrc+WRefB[1]+WRefF[1]+WRefB[2]+WRefF[2]+WRefB[3]+WRefF[3]+WRefB[4]+WRefF[4]+1;
    else if(delta==4)WSum=WRefB[0]+WRefF[0]+WSrc+WRefB[1]+WRefF[1]+WRefB[2]+WRefF[2]+WRefB[3]+WRefF[3]+1;
    else if(delta==3)WSum=WRefB[0]+WRefF[0]+WSrc+WRefB[1]+WRefF[1]+WRefB[2]+WRefF[2]+1;
    else if(delta==2)WSum=WRefB[0]+WRefF[0]+WSrc+WRefB[1]+WRefF[1]+1;
    else WSum=WRefB[0]+WRefF[0]+WSrc+1;
    WRefB[0]=WRefB[0]*256/WSum;WRefF[0]=WRefF[0]*256/WSum;
    if(delta>=2){WRefB[1]=WRefB[1]*256/WSum;WRefF[1]=WRefF[1]*256/WSum;}
    if(delta>=3){WRefB[2]=WRefB[2]*256/WSum;WRefF[2]=WRefF[2]*256/WSum;}
    if(delta>=4){WRefB[3]=WRefB[3]*256/WSum;WRefF[3]=WRefF[3]*256/WSum;}
    if(delta>=5){WRefB[4]=WRefB[4]*256/WSum;WRefF[4]=WRefF[4]*256/WSum;}
    if(delta>=6){WRefB[5]=WRefB[5]*256/WSum;WRefF[5]=WRefF[5]*256/WSum;}
    if(delta==6)WSrc=256-WRefB[0]-WRefF[0]-WRefB[1]-WRefF[1]-WRefB[2]-WRefF[2]-WRefB[3]-WRefF[3]-WRefB[4]-WRefF[4]-WRefB[5]-WRefF[5];
    else if(delta==5)WSrc=256-WRefB[0]-WRefF[0]-WRefB[1]-WRefF[1]-WRefB[2]-WRefF[2]-WRefB[3]-WRefF[3]-WRefB[4]-WRefF[4];
    else if(delta==4)WSrc=256-WRefB[0]-WRefF[0]-WRefB[1]-WRefF[1]-WRefB[2]-WRefF[2]-WRefB[3]-WRefF[3];
    else if(delta==3)WSrc=256-WRefB[0]-WRefF[0]-WRefB[1]-WRefF[1]-WRefB[2]-WRefF[2];
    else if(delta==2)WSrc=256-WRefB[0]-WRefF[0]-WRefB[1]-WRefF[1];
    else WSrc=256-WRefB[0]-WRefF[0];
    return WSrc;
}

int main(){
    char line[4096];
    while(fgets(line,sizeof(line),stdin)){
        char t;
        if(line[0]=='w'){ int th,bs; if(sscanf(line,"w %d %d",&th,&bs)==2) printf("w %d\n",kt_degrain_weight(th,bs)); }
        else if(line[0]=='n'){ int d,b; int a[6],a2[6];
            int n=sscanf(line,"n %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
                &d,&b,&a[0],&a2[0],&a[1],&a2[1],&a[2],&a2[2],&a[3],&a2[3],&a[4],&a2[4],&a[5],&a2[5]);
            int WSrc=kt_norm_weights(d,b,a,a2);
            printf("n %d",WSrc);
            for(int i=0;i<d;++i) printf(" %d %d",a[i],a2[i]);
            printf("\n");
        }
    }
    return 0;
}
