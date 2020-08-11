// Author: Wes Kendall
// Copyright 2011 www.mpitutorial.com
// This code is provided freely with the tutorials on mpitutorial.com. Feel
// free to modify it for your own use. Any distribution of the code must
// either provide a link to www.mpitutorial.com or keep this header intact.
//
// MPI_Send, MPI_Recv example. Communicates the number -1 from process 0
// to process 1.
//
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <hipcub/hipcub.hpp>


__global__ void scalarMul(int *d_vector, int *d_results, int n)
{

     int id = blockIdx.x*blockDim.x+threadIdx.x;

       // Make sure we do not go out of bounds
    if (id < n)
    {

            d_results[id] = d_vector[id] * d_vector[id];
    }

}



void Reduction_Sum( int *d_array, int *d_dest, size_t n )
{
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    /* determine temporary device storage requirements */
    hipcub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes,
            d_array, d_dest, n );
    hipDeviceSynchronize( );

    /* allocate temporary storage */
      hipMalloc(&d_temp_storage, temp_storage_bytes);
    /* run sum-reduction */
    hipcub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes,
            d_array, d_dest, n );
    hipDeviceSynchronize();


    /* deallocate temporary storage */
    hipFree(d_temp_storage);
}

// Creates an array of random numbers. Each number has a value from 0 - 1
int *create_rand_nums(int num_elements) {
  int *rand_nums = (int *)malloc(sizeof(int) * num_elements);
  assert(rand_nums != NULL);
  int i;
  for (i = 0; i < num_elements; i++) {
    rand_nums[i] = rand() % 10 + 1;
    }
  return rand_nums;
}



int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);
  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // We are assuming at least 2 processes for this task
  if (world_size < 2) {
    fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }


  int num_elements_per_proc = atoi(argv[1]);
  int total_elements = world_size * num_elements_per_proc;
  int data_in_bytes_per_node = num_elements_per_proc * sizeof(int);
  int total_data_in_bytes = total_elements * sizeof(int);


   // Seed the random number generator to get different results each time
  srand(time(NULL));

   // Create a random array of elements on the root process. Its total
  // size will be the number of elements per process times the number
  // of processes
  int *rand_nums = NULL;
  if (world_rank == 0) {
    rand_nums = create_rand_nums(total_elements);
  }




  // For each process, create a buffer that will hold a subset of the entire
  // array
  int *sub_rand_nums = (int *)malloc(data_in_bytes_per_node);
  assert(sub_rand_nums != NULL);


  // Scatter the random numbers from the root process to all processes in
  // the MPI world

  if(world_rank == 0)
  {  
  printf("Root node scattering data \n");
  }

  MPI_Scatter(rand_nums, num_elements_per_proc, MPI_INT, sub_rand_nums,
              num_elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);


  hipSetDevice(world_rank);
  int *d_sub_rand_nums,*d_results;


  hipMalloc(&d_sub_rand_nums,data_in_bytes_per_node);
  hipMalloc(&d_results,data_in_bytes_per_node);

  int *results = (int *)malloc(data_in_bytes_per_node);

  hipMemcpy(d_sub_rand_nums, sub_rand_nums, data_in_bytes_per_node, hipMemcpyHostToDevice);

  int blockSize, gridSize;

 // Number of threads in each thread block
  blockSize = 32;

  // Number of thread blocks in grid
  gridSize = (int)ceil((float)num_elements_per_proc/blockSize);


  hipLaunchKernelGGL(scalarMul, dim3(gridSize), dim3(blockSize), 0, 0, d_sub_rand_nums, d_results, num_elements_per_proc);
  hipDeviceSynchronize();

  printf("GPU %d completed its squaring. Copying data back to local CPU\n", world_rank);
 
  hipMemcpy(results, d_results, data_in_bytes_per_node, hipMemcpyDeviceToHost);


  
  int *aggregated_results = NULL;
  if (world_rank == 0) {
    aggregated_results = (int *)malloc(total_data_in_bytes);	  
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Gather(results, num_elements_per_proc, MPI_INT, aggregated_results, num_elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

   

  if(world_rank == 0)
  {
    printf("Root node finished gathering data. Running reduce kernel\n");  

    int *d_aggregated_results = NULL;
    hipMalloc(&d_aggregated_results, total_data_in_bytes);

    hipMemcpy(d_aggregated_results, aggregated_results,total_data_in_bytes, hipMemcpyHostToDevice);

   
    int *d_output;
    hipMalloc(&d_output, total_data_in_bytes);

    Reduction_Sum(d_aggregated_results , d_output, total_elements);

    int reduce_sum = 0;

    hipMemcpy(&reduce_sum, d_output, sizeof(int), hipMemcpyDeviceToHost);

    printf("Reduce sum %d\n", reduce_sum);
    
   }
  MPI_Finalize();
}

