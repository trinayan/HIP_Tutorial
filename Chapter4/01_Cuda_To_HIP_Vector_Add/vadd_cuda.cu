#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// GPU  kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}
 
int main( int argc, char* argv[] )
{
    // Size of vectors
    int n = 100000;
 
    // Host input vectors
    double *h_a;
    double *h_b;
    //Host output vector
    double *h_c;
    //Host output vector for verification
    double *h_verify_c;
 
    // Device input vectors
    double *d_a;
    double *d_b;
    //Device output vector
    double *d_c;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
 
    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);
    h_verify_c = (double*)malloc(bytes);
    
    // Allocate memory for each vector on GPU
   cudaMalloc(&d_a, bytes);
   cudaMalloc(&d_b, bytes);
   cudaMalloc(&d_c, bytes);
 
    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = i;
        h_b[i] = i;
    }


 
    // Copy host vectors to device
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes,  cudaMemcpyHostToDevice);
 
    int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 1024;
 
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n/blockSize);
 
    // Execute the kernel
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    // Copy array back to host
   cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost);
 

   //Compute for CPU 
   for(i=0; i <n; i++)
   {
    h_verify_c[i] = h_a[i] + h_b[i];
   }


    //Verfiy results
    for(i=0; i <n; i++)
    {
    if (abs(h_verify_c[i] - h_c[i]) > 1e-5) 
     {
     printf("Error at position i %d, Expected: %f, Found: %f \n", i, h_c[i], d_c[i]);
     }  
    }


     printf("Printing a subset of results till index 1024\n");

     for(i = 0; i < 1024 ; i++)
     {
	printf("Value at index %d is %f\n",i, h_c[i]);
     }

    
     
    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
 
    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);
 
    return 0;
}
