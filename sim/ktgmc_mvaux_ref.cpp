/* CPU mirror of the MV-aux kernels in src/opencl/ktgmc/kernels/ktgmc_motion.cl
 * (kt_write_default_mv, kt_scene_change/_x2, kt_short_to_byte). Independent
 * re-implementation for cross-checking against the Python golden.
 *
 * Usage:  ktgmc_mvaux_ref <type> <infile> <outfile>
 *   type = "def" | "sc" | "scx2" | "stb"
 *
 * Input text formats (space separated):
 *   def : <n> <verybigSAD>
 *   sc  : <nTh1> <sad_0..sad_{n-1}>
 *   scx2: <nTh1> <sadA_0..> <sadB_0..>   (two full sets of n sads)
 *   stb : <width> <height> <pitch> <shift> <maxval> <tmp_0..tmp_{pitch*height-1}>
 * Output: one value per result line (see header comments).
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

static vector<int> readNums(const char* path){
    FILE* f=fopen(path,"r"); vector<int> v; int x;
    while(f&&fscanf(f,"%d",&x)==1) v.push_back(x);
    if(f) fclose(f); return v;
}
static void writeFile(const char* path,const vector<string>& lines){
    FILE* f=fopen(path,"w"); for(auto&l:lines) fprintf(f,"%s\n",l.c_str()); if(f)fclose(f);
}

int main(int argc,char**argv){
    if(argc<4){fprintf(stderr,"usage: mvaux <type> <in> <out>\n");return 2;}
    string type=argv[1]; auto n=readNums(argv[2]); vector<string> out;

    if(type=="def"){ // n, verybigSAD
        int cnt=n[0], big=n[1];
        for(int i=0;i<cnt;++i) out.push_back("0 0 "+to_string(big));
    }
    else if(type=="sc"){ // nTh1, sads...
        int nTh1=n[0], count=0;
        for(size_t i=1;i<n.size();++i) if(n[i]>nTh1) ++count;
        out.push_back(to_string(count));
    }
    else if(type=="scx2"){ // nTh1, sadA..., sadB...
        int nTh1=n[0]; size_t m=(n.size()-1)/2; int ca=0,cb=0;
        for(size_t i=0;i<m;++i){ if(n[1+i]>nTh1)++ca; if(n[1+m+i]>nTh1)++cb; }
        out.push_back(to_string(ca)); out.push_back(to_string(cb));
    }
    else if(type=="stb"){ // width height pitch shift maxval tmp...
        int w=n[0],h=n[1],pitch=n[2],shift=n[3],maxv=n[4];
        for(int i=0;i<h;++i)for(int x=0;x<w;++x){
            int v=n[5+x+i*pitch]>>shift; v=min(v,maxv); if(v<0)v=0; out.push_back(to_string(v));
        }
    }
    writeFile(argv[3],out);
    return 0;
}
