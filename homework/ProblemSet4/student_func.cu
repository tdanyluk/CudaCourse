//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

#define SWAP(a, b, tmp) tmp = a; a = b; b = tmp

int div_up(int a, int b)
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

__global__ void histogram(const unsigned int* const in, int size, unsigned int shift_bits, unsigned int mask, unsigned int *histogram)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i < size)
  {
    atomicAdd(&histogram[(in[i]>>(shift_bits)) & mask], 1);
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
  //__syncthreads();

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


__global__ void scan_locations1(const unsigned int * const arr, const int size, const int log2_up_size,
  const int bins, unsigned int shift_bits, unsigned int mask,
  unsigned int *loc_tmp, unsigned int* loc_tmp2, unsigned int *locations, int bin, int step)
{
  int t = blockDim.x * blockIdx.x + threadIdx.x;

  unsigned int val = 0;
  if(t<size && t > 0)
  {
    val = (((arr[t-1] >> shift_bits) & mask) == bin) ? 1 : 0;
  }
  if(t < size)
  {
    loc_tmp[t] = val;
  }
}


__global__ void scan_locations2(const unsigned int * const arr, const int size, const int log2_up_size,
  const int bins, unsigned int shift_bits, unsigned int mask,
  unsigned int *loc_tmp, unsigned int* loc_tmp2, unsigned int *locations, int bin, int step)
{
  int t = blockDim.x * blockIdx.x + threadIdx.x;

  int d = 1<<step;
  unsigned int val = 0;
  if(t<size && t-d>=0)
  {
    val = loc_tmp[t-d];
  }
  if(t<size)
  {
    loc_tmp2[t] = loc_tmp[t] + val;
  }
}

__global__ void scan_locations3(const unsigned int * const arr, const int size, const int log2_up_size,
  const int bins, unsigned int shift_bits, unsigned int mask,
  unsigned int *loc_tmp, unsigned int* loc_tmp2, unsigned int *locations, int bin, int step)
{
  int t = blockDim.x * blockIdx.x + threadIdx.x;

  if(t<size && ((arr[t] >> shift_bits) & mask) == bin)
  {
    locations[t] = loc_tmp[t];
  }
}


__global__ void reorder(const unsigned int *const in_key, const unsigned int *const in_val, const int size,
  unsigned int *const out_key, unsigned int *const out_val,
  const unsigned int *hist, unsigned int shift_bits, unsigned int mask, const unsigned int* rel_loc)
{
  int t = blockDim.x * blockIdx.x + threadIdx.x;

  if(t < size)
  {
    int bin = (in_key[t] >> shift_bits) & mask;
    int i = hist[bin] + rel_loc[t];
    out_key[i] = in_key[t];
    out_val[i] = in_val[t];
  }

}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  int block_size = 1024;
  int grid_size = div_up(numElems, block_size);

  unsigned int *d_histogram;
  unsigned int *d_locations;
  unsigned int *d_tmp_locations;
  unsigned int *d_tmp_locations2;
  int n_hist = 16;
  int log_n_hist = 4;
  checkCudaErrors(cudaMalloc(&d_histogram, n_hist * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_locations, numElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_tmp_locations, numElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_tmp_locations2, numElems * sizeof(unsigned int)));

  unsigned int *d_in_vals = d_inputVals;
  unsigned int *d_out_vals = d_outputVals;
  unsigned int *d_in_pos = d_inputPos;
  unsigned int *d_out_pos = d_outputPos;

  for(int i = 0; i<sizeof(unsigned int)*8/log_n_hist; i++)
  {
    checkCudaErrors(cudaMemset(d_histogram, 0, n_hist * sizeof(unsigned int)));
    histogram<<<grid_size, block_size>>>(d_in_vals, numElems, i*log_n_hist, n_hist-1, d_histogram);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    exclusive_scan<<<1, block_size, n_hist*sizeof(unsigned int)>>>(d_histogram, n_hist, log_n_hist);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //checkCudaErrors(cudaMemset(d_locations, 0, numElems * sizeof(unsigned int)));
    for(int bin = 0; bin < n_hist; bin++)
    {
      scan_locations1<<<grid_size, block_size>>>(d_in_vals, numElems, log2_up(numElems), n_hist, i*log_n_hist, n_hist-1, 
      d_tmp_locations, d_tmp_locations2, d_locations, bin, -1);
  
      for(int step = 0; step < log2_up(numElems); step++)
      {
        scan_locations2<<<grid_size, block_size>>>(d_in_vals, numElems, log2_up(numElems), n_hist, i*log_n_hist, n_hist-1, 
        d_tmp_locations, d_tmp_locations2, d_locations, bin, step);
        unsigned int *tmp;
        SWAP(d_tmp_locations, d_tmp_locations2, tmp);
      }

      // {
      //   unsigned int *h_locations = new unsigned int[numElems];
      //   cudaMemcpy(h_locations, d_tmp_locations, numElems*sizeof(unsigned int), cudaMemcpyDeviceToHost);
      //   for(int i = 0; i<numElems; i++)
      //   {
      //     printf("loc %d %u\n", i, h_locations[i]);
      //   }
      //   printf("\n");
      //   delete [] h_locations;
      // }  

      scan_locations3<<<grid_size, block_size>>>(d_in_vals, numElems, log2_up(numElems), n_hist, i*log_n_hist, n_hist-1, 
      d_tmp_locations, d_tmp_locations2, d_locations, bin, -1);

      // {
      //   unsigned int *h_locations = new unsigned int[numElems];
      //   cudaMemcpy(h_locations, d_locations, numElems*sizeof(unsigned int), cudaMemcpyDeviceToHost);
      //   for(int i = 0; i<numElems; i++)
      //   {
      //     printf("loc2 %d %u\n", i, h_locations[i]);
      //   }
      //   printf("\n");
      //   delete [] h_locations;
      // }
    }
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    reorder<<<grid_size, block_size>>>(d_in_vals, d_in_pos, numElems, d_out_vals, d_out_pos, 
      d_histogram, i*log_n_hist, n_hist-1, d_locations);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    unsigned int *tmp;
    SWAP(d_in_vals, d_out_vals, tmp);
    SWAP(d_in_pos, d_out_pos, tmp);
  }

  if(d_outputVals != d_in_vals)
  {
    checkCudaErrors(cudaMemcpy(d_outputVals, d_in_vals, numElems*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_outputPos, d_in_pos, numElems*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  }

  checkCudaErrors(cudaFree(d_histogram));
  checkCudaErrors(cudaFree(d_locations));
  checkCudaErrors(cudaFree(d_tmp_locations));
  checkCudaErrors(cudaFree(d_tmp_locations2));
}
