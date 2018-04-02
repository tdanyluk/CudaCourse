/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <float.h>
#include <stdio.h>

__host__ __device__ int div_up(int a, int b)
{
  return (a + b - 1) / b;
}

int log2_up(int i)
{
  int j = 1;
  int l = 0;
  while(j < i)
  {
    j*=2;
    l+=1;
  }
  return l;
}

__global__ void reduce_min(const float* const values, const int size, float *min_values)
{
  extern __shared__ char ext[];
  float *from = (float*)ext;
  float *to = (float*)ext + blockDim.x * 2;

  int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  
  if(i < size)
  {
    from[2 * threadIdx.x] = values[i];
  }
  else 
  {
    from[2 * threadIdx.x] = FLT_MAX;
  }

  if(i+1 < size)
  {
    from[2 * threadIdx.x + 1] = values[i + 1];
  }
  else
  {
    from[2 * threadIdx.x + 1] = FLT_MAX;
  }

  __syncthreads();
  
  int curr_size = blockDim.x * 2;
  
  while(curr_size > 1)
  {
    if(2 * threadIdx.x < curr_size)
    {
      to[threadIdx.x] = min(from[2 * threadIdx.x], from[2 * threadIdx.x + 1]);
    }

    __syncthreads();

    float* tmp = from;
    from = to;
    to = tmp;
    curr_size /= 2;
  }

  if(threadIdx.x == 0)
  {
    min_values[blockIdx.x] = from[0];
  }
}

__global__ void reduce_max(const float* const values, const int size, float *max_values)
{
  extern __shared__ char ext[];
  float *from = (float*)ext;
  float *to = (float*)ext + blockDim.x * 2;

  int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  
  if(i < size)
  {
    from[2 * threadIdx.x] = values[i];
  }
  else 
  {
    from[2 * threadIdx.x] = -FLT_MAX;
  }

  if(i+1 < size)
  {
    from[2 * threadIdx.x + 1] = values[i + 1];
  }
  else
  {
    from[2 * threadIdx.x + 1] = -FLT_MAX;
  }

  __syncthreads();
  
  int curr_size = blockDim.x * 2;
  
  while(curr_size > 1)
  {
    if(2 * threadIdx.x < curr_size)
    {
      to[threadIdx.x] = max(from[2 * threadIdx.x], from[2 * threadIdx.x + 1]);
    }

    __syncthreads();

    float* tmp = from;
    from = to;
    to = tmp;
    curr_size /= 2;
  }

  if(threadIdx.x == 0)
  {
    max_values[blockIdx.x] = from[0];
  }
}

__global__ void atomic_histogram(
  const float* const values, const int size, const float min, const float max, const int nBins, unsigned int *bins)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
  {
    const float range = max - min;
    int bin = (int)((values[i] - min) / range * nBins);
    if(bin == nBins)
    {
      bin = nBins-1;
    }

    atomicAdd(&bins[bin], 1);
  }
}

__global__ void exclusive_scan(unsigned int *arr, const int size, const int log2_up_size)
{
  extern __shared__ char ext[];
  unsigned int *s = (unsigned int*)ext;

  const int t = threadIdx.x;

  float val = 0;
  if(t < size && t > 0)
  {
    val = arr[t-1];
  }
  __syncthreads();

  if(t < size)
  {
    s[t] = val;
  }
  __syncthreads();

  int d = 1;
  for(int i = 0; i<log2_up_size; i++, d*=2)
  {
    float val = 0;
    if(t<size && t-d>=0)
    {
      val = s[t-d];
    }
    __syncthreads();
    if(t<size)
    {
      s[t] += val;
    }
    __syncthreads();
  }

  if(t < size)
  {
    arr[t] = s[t];
  }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  int block_size = 1024;
  int grid_size = div_up(div_up(numRows*numCols, block_size), 2);
  int shared_memory_size = 3*block_size*sizeof(float);

  float *d_mins;
  checkCudaErrors(cudaMalloc(&d_mins, grid_size * sizeof(float)));

  reduce_min<<<grid_size, block_size, shared_memory_size>>>(d_logLuminance, numRows*numCols, d_mins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  reduce_min<<<1, grid_size, shared_memory_size>>>(d_mins, grid_size, d_mins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(&min_logLum, d_mins, sizeof(float), cudaMemcpyDeviceToHost));
  
  reduce_max<<<grid_size, block_size, shared_memory_size>>>(d_logLuminance, numRows*numCols, d_mins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  reduce_max<<<1, grid_size, shared_memory_size>>>(d_mins, grid_size, d_mins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(&max_logLum, d_mins, sizeof(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_mins));

  checkCudaErrors(cudaMemset(d_cdf, 0, numBins * sizeof(unsigned int)));

  block_size = 1024;
  grid_size = div_up(numRows*numCols, block_size);
  shared_memory_size = numBins * sizeof(unsigned int);
  atomic_histogram<<<grid_size,block_size,shared_memory_size>>>(d_logLuminance, numRows*numCols, min_logLum, max_logLum, numBins, d_cdf);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  block_size = numBins;
  grid_size = 1;
  shared_memory_size = numBins * sizeof(unsigned int);
  exclusive_scan<<<grid_size,block_size,shared_memory_size>>>(d_cdf, numBins, log2_up(numBins));
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

       
}
