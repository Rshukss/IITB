#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>

const int N = 100000000;

__global__ void memoryIntensiveKernel(float *a, float *b, float *c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
  	{
    	c[i] = a[i] + b[i];
		  b[i] = a[i] + c[i];
		  a[i] = b[i] + c[i];
  	}
  
}

int main()
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float *a, *b, *c;
  cudaMalloc((void **)&a, N * sizeof(float));
  cudaMalloc((void **)&b, N * sizeof(float));
  cudaMalloc((void **)&c, N * sizeof(float));

  float *a_h, *b_h;
  a_h = new float[N];
  b_h = new float[N];

  for (int i = 0; i < N; i++)
  {
    a_h[i] = i;
    b_h[i] = i;
  }

  cudaMemcpy(a, a_h, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b, b_h, N * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 512;
  int numBlocks = (N + blockSize - 1) / blockSize;
  cudaEventRecord(start);
  for (int iter=1; iter<5001; iter++)
  {  
  	memoryIntensiveKernel<<<numBlocks, blockSize>>>(a, b, c);
	  std::cout << iter <<" Iteration completed"<<std::endl;
  }
  cudaEventRecord(stop);

  float *c_h;
  c_h = new float[N];
  cudaMemcpy(c_h,c,N*sizeof(float),cudaMemcpyDeviceToHost);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  cudaEventSynchronize(stop);
  float mil = 0;
  cudaEventElapsedTime(&mil,start,stop);
  printf("ET: %f ms\n",mil);
  return 0;
}
