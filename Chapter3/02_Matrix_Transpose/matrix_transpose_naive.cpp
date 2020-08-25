#include <hip/hip_runtime.h>

#include <cstdlib>
#include <vector>

const static int width = 32;
const static int height = 32;
const static int tile_dim = 32;

__global__ void transpose_kernel(float *in, float *out, int width, int height) {
  int x_index = blockIdx.x * tile_dim + threadIdx.x;
  int y_index = blockIdx.y * tile_dim + threadIdx.y;

  int in_index = y_index * width + x_index;
  int out_index = x_index * height + y_index;

  out[out_index] = in[in_index];
}

int main() {
  std::vector<float> matrix_in;
  std::vector<float> matrix_out;

  matrix_in.resize(width * height);
  matrix_out.resize(width * height);

  for (int i = 0; i < width * height; i++) {
    matrix_in[i] = (float)rand() / (float)RAND_MAX;
  }

  printf("\nInput matrix:\n");
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      printf("%.2f ", matrix_in[i*width + j]);
    }
    printf("\n");
  }



  float *d_in;
  float *d_out;

  hipMalloc((void **)&d_in, width * height * sizeof(float));
  hipMalloc((void **)&d_out, width * height * sizeof(float));

  hipMemcpy(d_in, matrix_in.data(), width * height * sizeof(float),
            hipMemcpyHostToDevice);

  int block_x = width / tile_dim;
  int block_y = height / tile_dim;

  hipLaunchKernelGGL(transpose_kernel, dim3(block_x, block_y),
                      dim3(tile_dim, tile_dim), 0, 0, d_in, d_out, width,
                      height);
  hipMemcpy(matrix_out.data(), d_out, width * height * sizeof(float),
            hipMemcpyDeviceToHost);
  hipDeviceSynchronize();

  printf("\nOutput matrix:\n");
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      printf("%.2f ", matrix_out[i*width + j]);
    }
    printf("\n");
  }

  return 0;
}

