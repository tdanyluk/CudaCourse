#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <utility>
#include "compare.h"
#include "gputimer.h"
#include <cub/cub.cuh>

template<typename T>
__host__ __device__
void swap(T& a, T& b)
{
    T c = a;
    a = b;
    b = c;
}

template<typename T>
__host__ __device__
void sort(T& a, T& b)
{
    if(a > b)
    {
        swap(a, b);
    }
}

// http://en.wikipedia.org/wiki/Bitonic_sort
__global__ void batcherBitonicMergesort64(float * d_out, const float * d_in)
{
    // you are guaranteed this is called with <<<1, 64, 64*4>>>
    extern __shared__ float sdata[];
    int t  = threadIdx.x;
    sdata[t] = d_in[t];
    __syncthreads();
    
    for (int stage = 0; stage <= 5; stage++)
    {
        for(int substage = stage; substage >= 0; substage--)
        {
            int e = (2 << substage);
            int r = t & (e-1);
            if(r < (e >> 1))
            {
                int p = (substage == stage) ? t - 1 - (r << 1) + e : t + (e >> 1);
                sort(sdata[t], sdata[p]);
            }
            __syncthreads();
        }
    }

    d_out[t] = sdata[t];
}

int compareFloat (const void * a, const void * b)
{
  if ( *(float*)a <  *(float*)b ) return -1;
  if ( *(float*)a == *(float*)b ) return 0;
  if ( *(float*)a >  *(float*)b ) return 1;
  return 0;                     // should never reach this
}

int main(int argc, char **argv)
{
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // generate the input array on the host
    float h_in[ARRAY_SIZE];
    float h_sorted[ARRAY_SIZE];
    float h_out[ARRAY_SIZE];
    for(int i = 0; i < ARRAY_SIZE; i++) {
        // generate random float in [0, 1]
        h_in[i] = (float)random()/(float)RAND_MAX;
        h_sorted[i] = h_in[i];
    }
    qsort(h_sorted, ARRAY_SIZE, sizeof(float), compareFloat);

    // declare GPU memory pointers
    float * d_in, * d_out;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 

    // launch the kernel
    GpuTimer timer;
    timer.Start();
    batcherBitonicMergesort64<<<1, ARRAY_SIZE, ARRAY_SIZE * sizeof(float)>>>(d_out, d_in);
    cudaDeviceSynchronize();
    timer.Stop();
    
    printf("Your code executed in %g ms\n", timer.Elapsed());
    
    // copy back the sum from GPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    compare(h_out, h_sorted, ARRAY_SIZE);
  
    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out);
        
    return 0;
}
