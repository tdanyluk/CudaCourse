/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/

#include "utils.h"
#include <cstdio>

__host__ __device__
int div_up(int a, int b)
{
  return (a + b - 1) / b;
}

__global__
void atomic_histogram(const unsigned int* const values, unsigned int* const bins, int size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
  {
    atomicAdd(&bins[values[i]], 1);
  }
}

__global__
void atomic_histogram2(const unsigned int* const values, unsigned int* const bins, int size, 
  int sizePerBlock, int numBins)
{
  extern __shared__ unsigned int sbins[];

  int binPerThread = div_up(numBins, blockDim.x);
  for(int i = 0; i<binPerThread; i++)
    if(threadIdx.x * binPerThread + i < numBins)
      sbins[threadIdx.x * binPerThread + i] = 0;

  __syncthreads();

  int offset = blockIdx.x * sizePerBlock;
  int valPerThread = div_up(sizePerBlock, blockDim.x);

  for(int i = 0; i<valPerThread; i++)
    if(offset + threadIdx.x * valPerThread + i < size)
      atomicAdd(&sbins[values[offset + threadIdx.x * valPerThread + i]], 1);

  __syncthreads();

  for(int i = 0; i<binPerThread; i++)
    if(threadIdx.x * binPerThread + i < numBins)
      atomicAdd(&bins[threadIdx.x * binPerThread + i], sbins[threadIdx.x * binPerThread + i]);
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  printf("numElems: %u numBins: %u\n", numElems, numBins);
  int block_size = 1024;
  int grid_size = div_up(numElems, block_size);

  // d_histo is already all zeros
  atomic_histogram<<<grid_size, block_size>>>(d_vals, d_histo, numElems);

  // int block_size = 1024;
  // int grid_size = 100;
  // int mem_size = numBins * sizeof(unsigned int);

  // printf("perThread: %d\n", div_up(numElems, grid_size*block_size));

  // atomic_histogram2<<<grid_size, block_size, mem_size>>>(
  //   d_vals, d_histo, numElems, div_up(numElems, grid_size), numBins);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
