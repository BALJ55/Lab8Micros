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



__global__ void
filter_kernel(unsigned char *d_input, unsigned char *d_output, int rows, int cols,float* d_window, int window_size) {


  //define image row, col position

  const int c = threadIdx.x + blockDim.x * blockIdx.x;

  const int r = threadIdx.y + blockDim.y * blockIdx.y;
  // exit if out of image bounds

  if((c >= cols) || (r >= rows)) return;


  // compute flat index

  const int i = flatten(c, r, cols, rows);


  const int gloc= threadIdx.x + RAD;
  extern_shared_float data[];

  data[gloc]= d_input[i];

  if (threadIdx.x <RAD){
      data[gloc -RAD]-d_input[i-RAD];
      data[gloc +blockDim.x] - d_input[i +blockDim.x];
  }
  _syncthreads();

  float pixel_result = 0;
  //se aplica la convolucion a la imagen para esto se llegara a utilizar un for 
  for(int rd = -RAD; rd<= RAD; ++rd){
   for(int cd = -RAD; cd<= RAD; ++cd){
  //compute image and windows indexes
  int imgIdx= flatten(c +cd, r +rd, cols, rows);
  int fltIdx= flatten(RAD + cd, RAD +rd, window_size);  
  // lee la funcion del pixel y los indexes de la ventana
  uchar pixel_val =i_input[imgIdx];
  float weight = d_window[fltIdx];
  //se acumula el valor
  pixel_result += pixel_val * weight;
 }
}
//bound pixel value to [0,255] and store in output
d_output[i] = clip((uchar) pixel_result);
}

//definicion del wrapper function
void filter_gpu(Mat input, Mat output){
  //convolucion del window size
  const int window_size = 2 * RAD +1;
  const float edgeDetected[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
  unsigned char *inputPtr = (unsigned char*) input.data;
  unsigned char *outputPtr = (unsigned char*) output.data;
  unsigned char rows = input.rows;
  unsigned char cols= input.cols;
  //grid size dimensions (blocks)
  int Bx = (TX +cols -1)/TX;
  int By = (TY +rows -1)TY;
// se declaran punteros para memoria 
  unsigned char *d_in = 0;
  unsigned char *d_out= 0;
  float *d_window = 0;
//input y output en el device
  cudaMalloc(&d_in, cols*rows*sizeof(unsigned char));
  cudaMalloc(&d_out, cols*rows* sizeof(unsigned char)); 
  cudaMalloc(&d_window, window_size*window_size * sizeof(float));

//de ejemplos en clase, del host al device
  cudaMemcpy(d_in, inputPtr, cols*rows*sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_window, edgeDetect, window_size*window_size*sizeof(float),  cudaMemcpyHosttoDevice);
//dimensiones del kernel
  size_t sharedMemSize = (TX +2 *RAD) *sizeof(float);
  const dim3 blockSize = dim3(TX, TY);
  const dim3 gridSize = dim3(Bx, By);
//GPU y el kernel
  filter_kernel<<gridSize, blockSize, sharedMemSize >>> (d_in, d_out, rows, cols, d_window, window_size);
//copia del device al host
  cudaMemcpy(outputPtr, d_out, rows*cols*sizeof(unsigned char) , cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_window);
}
