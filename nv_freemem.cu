#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <mex.h> 
#include <stdio.h>


void mexFunction()
{
   
  cuInit(0);



  CUdevice dev;

  int nGPUs;

  cuDeviceGetCount(&nGPUs);

  mexPrintf("Device Info: %d GPUs found in system.\n", nGPUs);



  CUcontext ctx;

  cuDeviceGet(&dev,0);       // use 1st CUDA device

  cuCtxCreate(&ctx, 0, dev); // create context for it



  CUresult memres;

  unsigned int free, total;

  memres = cuMemGetInfo(&free, &total);

  mexPrintf("After all allocation(%d):     free(MB) %d     total(MB) %d \n", memres, free/1024/1024, total/1024/1024);



  cuCtxDetach(ctx);
}
