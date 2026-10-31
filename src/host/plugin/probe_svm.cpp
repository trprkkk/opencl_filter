// probe_svm.cpp — does this driver support SVM (clSVMAlloc) + interior-origin
// pointer passing, needed by KMSuper's kt_copy_pad / kt_rb2b_*_with_pad?
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cstdio>
#include <cstring>
#include <vector>

static const char* SRC =
"kernel void fill(__global int* p, int base){ int i=get_global_id(0); p[base+i]=i*7+1; }\n"
"kernel void readback(__global const int* p, int base, __global int* out){ int i=get_global_id(0); out[i]=p[base+i]; }\n";

int main(){
    cl_uint np=0; clGetPlatformIDs(0,nullptr,&np);
    if(!np){printf("no platform\n");return 1;}
    std::vector<cl_platform_id> ps(np); clGetPlatformIDs(np,ps.data(),nullptr);
    cl_device_id dev=nullptr;
    for(auto p:ps){ if(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&dev,nullptr)==CL_SUCCESS) break; }
    if(!dev){printf("no gpu\n");return 1;}
    char name[256]={0}; clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof(name)-1,name,nullptr);
    cl_device_svm_capabilities svmc=0;
    cl_int e=clGetDeviceInfo(dev,CL_DEVICE_SVM_CAPABILITIES,sizeof(svmc),&svmc,nullptr);
    printf("device=%s svm_query=%d caps=0x%x\n",name,e,svmc);
    cl_context ctx=clCreateContext(nullptr,1,&dev,nullptr,nullptr,&e);
    if(!ctx){printf("ctx fail %d\n",e);return 1;}
    cl_command_queue q=clCreateCommandQueue(ctx,dev,0,&e);
    cl_program prog=clCreateProgramWithSource(ctx,1,&SRC,nullptr,&e);
    e=clBuildProgram(prog,1,&dev,"-cl-std=CL1.2",nullptr,nullptr);
    if(e!=CL_SUCCESS){ size_t l=0; clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,0,nullptr,&l);
        std::vector<char> b(l+1); clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,l,b.data(),nullptr);
        printf("build fail: %s\n",b.data()); return 1; }
    // SVM allocation
    void* svm=clSVMAlloc(ctx,CL_MEM_READ_WRITE,(size_t)256*sizeof(int),0);
    printf("clSVMAlloc -> %p\n",svm);
    if(!svm) return 1;
    e=clEnqueueSVMMap(q,CL_TRUE,CL_MAP_WRITE,(void*)svm,(size_t)256*sizeof(int),0,nullptr,nullptr);
    printf("svmmap write=%d\n",e);
    int* host=(int*)svm; for(int i=0;i<256;i++) host[i]=-1;
    e=clEnqueueSVMUnmap(q,(void*)svm,0,nullptr,nullptr); printf("svmunmap=%d\n",e);
    cl_kernel kf=clCreateKernel(prog,"fill",&e);
    void* pbase=(char*)svm + 32*sizeof(int); // interior origin at index 32
    e=clSetKernelArgSVMPointer(kf,0,pbase); printf("setargSVM fill=%d\n",e);
    int base=0; clSetKernelArg(kf,1,sizeof(base),&base);
    size_t g=64; e=clEnqueueNDRangeKernel(q,kf,1,nullptr,&g,nullptr,0,nullptr,nullptr);
    printf("enqueue fill=%d\n",e);
    e=clFinish(q); printf("finish=%d\n",e);
    e=clEnqueueSVMMap(q,CL_TRUE,CL_MAP_READ,(void*)svm,(size_t)256*sizeof(int),0,nullptr,nullptr);
    printf("svmmap read=%d\n",e);
    bool ok=true;
    for(int i=0;i<32;i++) if(host[i]!=-1) ok=false;
    for(int i=0;i<64;i++) if(host[32+i]!=i*7+1) { ok=false; printf("mismatch at %d got %d exp %d\n",i,host[32+i],i*7+1); break; }
    printf("interior-origin SVM write: %s\n", ok?"OK":"BAD");
    clEnqueueSVMUnmap(q,(void*)svm,0,nullptr,nullptr);
    clSVMFree(ctx,svm);
    clReleaseKernel(kf); clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx);
    return ok?0:2;
}
