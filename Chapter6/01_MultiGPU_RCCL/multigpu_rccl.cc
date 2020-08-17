#include <stdio.h>
#include "hip/hip_runtime.h"
#include "rccl.h"

#define HIPCHECK(cmd) do {                         \
  hipError_t e = cmd;                              \
  if( e != hipSuccess ) {                          \
    printf("Failed: Hip error %s:%d '%s'\n",             \
        __FILE__,__LINE__,hipGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


int main(int argc, char* argv[])
{
  ncclComm_t comms[2];


  //managing 4 devices
  int nDev = 2;
  int size = 1*1024*1024;
  int devs[4] = { 1, 2 };


  //allocating and initializing device buffers
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  hipStream_t* s = (hipStream_t*)malloc(sizeof(hipStream_t)*nDev);


  for (int i = 0; i < nDev; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipMalloc(sendbuff + i, size * sizeof(float)));
    HIPCHECK(hipMalloc(recvbuff + i, size * sizeof(float)));
    HIPCHECK(hipMemset(sendbuff[i], 1, size * sizeof(float)));
    HIPCHECK(hipMemset(recvbuff[i], 0, size * sizeof(float)));
    HIPCHECK(hipStreamCreate(s+i));
  }


  //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));


   //calling NCCL communication API. Group API is required when using
   //multiple devices per thread
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i)
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
        comms[i], s[i]));
  NCCLCHECK(ncclGroupEnd());


  //synchronizing on HIP streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipStreamSynchronize(s[i]));
  }


  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipFree(sendbuff[i]));
    HIPCHECK(hipFree(recvbuff[i]));
  }


  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);


  printf("Success \n");
  return 0;
}
