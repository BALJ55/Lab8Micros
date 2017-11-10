#include "kernel.h"
#include <stdio.h>

#define TX 32
#define TY 32
#define RAD 1

/*  
int divUp(int a, int b){
	return (a + b - 1)/b;
}
*/
// clip values to [0 , 255]
__device__ unsigned char clip(int n){
  return n > 255 ? 255 : (n < 0 ? 0 : n);
}

// bound index values to max size
__device__ int idxClip(int idx, int idxMax){
 return idx >(idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

// transform img(c,r) to flat index i
__device__ int flatten(int col, int row, int width, int height) {
  return idxClip(col, width) + idxClip(row, height)*width;
}


__global__ void filter_kernel(unsigned char *d_input, unsigned char *d_output, int rows, int cols, float* d_window, int window_size) {


  //const int s_idx = threadIdx.x + RAD;
  extern __shared__ float smem[];
  const int s_idx = threadIdx.x + RAD;

  //define image row, col position
  const int c = threadIdx.x + blockDim.x * blockIdx.x;
  const int r = threadIdx.y + blockDim.y * blockIdx.y;

  // exit if out of image bounds
  if((c >= cols) || (r >= rows)) return;

  // compute flat index
  const int i = flatten(c, r, cols, rows);

  smem[s_idx] = d_input[i];
 
  float pixel_result = 0;

  // apply the convolution window to the image
  // -----------------------------------------
  for(int rd = -RAD; rd <= RAD; ++rd){
    for(int cd = -RAD; cd <= RAD; ++cd){
      
      // compute image and window indexes
      int imgIdx = flatten(c + cd, r + rd, cols, rows);
      int fltIdx = flatten(RAD + cd, RAD + rd, window_size, window_size);

      // read pixel value and window constant
      uchar pixel_val = d_input[imgIdx];
      float weight = d_window[fltIdx];

      // accumulate value
      pixel_result += pixel_val * weight;
    }
  }
 
 // bound pixel value to [0 255] and store in output
  d_output[i] = clip((uchar) pixel_result);
}


// wrapper function definition
void filter_gpu(Mat input, Mat output){

  // define convoltion window size
  const int window_size = 2 * RAD + 1;

  // define convoltion window values

 /*
    const float sharpen[9] = {-1.0, -1.0, -1.0,
                             -1.0,  9.0, -1.0,
                             -1.0, -1.0, -1.0};
  
   const float gaussianBlur[9] = {1/16, 2/16, 1/16,
                                  2/16, 4/16, 2/16,
                                  1/16, 2/16, 1/16};
*/
  const float edgeDetect[9] = {-1, -1, -1,
                               -1,  8, -1,
                               -1, -1, -1};

  unsigned char *inputPtr = (unsigned char*) input.data;
  unsigned char *outputPtr = (unsigned char*) output.data;
  unsigned int cols = input.cols;
  unsigned int rows = input.rows;

  // grid size dimensions (blocks)
  int Bx = (TX + cols -1)/TX;
  int By = (TY + rows -1)/TY;

  // declare pointers to device memory
  unsigned char *d_in  = 0;
  unsigned char *d_out = 0;
  float *d_window = 0;

  // allocate input and output memory in device
  cudaMalloc(&d_in, cols*rows*sizeof(unsigned char));
  cudaMalloc(&d_out, cols*rows*sizeof(unsigned char));
  cudaMalloc(&d_window, window_size*window_size * sizeof(float));

  // copy input data and filter element from host to device
  cudaMemcpy(d_in, inputPtr, cols*rows*sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_window, edgeDetect, window_size*window_size*sizeof(float), cudaMemcpyHostToDevice);

  // prepare kernel lauch dimensions
  const dim3 blockSize = dim3(TX, TY);
  const dim3 gridSize = dim3(Bx, By);

  const size_t smemSize = ((TX * TY) + (2 * RAD) * sizeof(float));

  // launch kernel in GPU
  filter_kernel<<<gridSize, blockSize, smemSize>>>(d_in, d_out, rows, cols, d_window, window_size);

  // copy output from device to host
  cudaMemcpy(outputPtr, d_out, rows*cols*sizeof(unsigned char), cudaMemcpyDeviceToHost);

  // free the memory allocated for device arrays
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_window);

}
