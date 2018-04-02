//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>
#include <algorithm>

enum PixelType {
  PTOutside = 0,
  PTBorder = 1,
  PTInterior = 2
};

__host__ __device__
int div_up(int a, int b)
{
  return (a + b - 1) / b;
}

template<class T>
__host__ __device__
T clamp(T i, T min, T max)
{
  if(i<min)
  {
    return min;
  }
  if(i>max)
  {
    return max;
  }
  return i;
}

__global__
void compute_mask(const uchar4 *d_sourceImg, unsigned char *d_mask, int sourceSize)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<sourceSize)
  {
    uchar4 p = d_sourceImg[i];
    d_mask[i] = (static_cast<int>(p.x) + p.y + p.z < 3*255) ? 1 : 0;
  }
}

// TODO blokkosítani
__global__
void compute_pixel_type(const unsigned char *d_mask, unsigned char *d_pixelTypes, int numRowsSource, int numColsSource)
{
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  if(row < numRowsSource && col < numColsSource)
  {
    if(d_mask[row*numColsSource + col])
    {
      int sum = d_mask[ clamp(row-1, 0, numRowsSource-1)*numColsSource + col ]
        + d_mask[ clamp(row+1, 0, numRowsSource-1)*numColsSource + col ]
        + d_mask[ row*numColsSource + clamp(col - 1, 0, numColsSource - 1) ]
        + d_mask[ row*numColsSource + clamp(col + 1, 0, numColsSource - 1) ];

      d_pixelTypes[row*numColsSource + col] = (sum == 4) ? PTInterior: PTBorder;
    }
    else
    {
      d_pixelTypes[row*numColsSource + col] = PTOutside;
    }
  }
}

template<class OutputType>
__global__
void split_channels(const uchar4 *d_img, OutputType* d_r, OutputType* d_g, OutputType* d_b, int size)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<size)
  {
    uchar4 p = d_img[i];
    d_r[i] = static_cast<OutputType>(p.x);
    d_g[i] = static_cast<OutputType>(p.y);
    d_b[i] = static_cast<OutputType>(p.z);
  }
}

__global__
void compose_img(uchar4 *d_destImg, float *d_r, float *d_g, float *d_b, unsigned char *d_pixelTypes, int size)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<size && d_pixelTypes[i] == PTInterior)
  {
    uchar4 p;
    p.x = static_cast<unsigned char>(d_r[i]);
    p.y = static_cast<unsigned char>(d_g[i]);
    p.z = static_cast<unsigned char>(d_b[i]);
    p.w = 255;
    d_destImg[i] = p;
  }
}

__global__
void process(const unsigned char *src, const unsigned char *dst, const unsigned char *type,
  const float *guess_prev, float *guess_next, int height, int width)
{
  // 1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
  //   Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
  //         else if the neighbor in on the border then += DestinationImg[neighbor]

  //   Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

  // 2) Calculate the new pixel value:
  //   float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
  //   ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]

  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int i = row*width + col;

  if(row < height && col < width && type[i] == PTInterior)
  {
    int i_nb[] = { 
      clamp(row-1, 0, height-1)*width + col,  
      clamp(row+1, 0, height-1)*width + col,
      row*width + clamp(col - 1, 0, width-1),  
      row*width + clamp(col + 1, 0, width-1)
    };

    float sum = 0;
    for (int j = 0; j < 4; j++)
    {
      if(type[i_nb[j]] == PTInterior)
      {
        sum += guess_prev[i_nb[j]];
      }
      else if(type[i_nb[j]] == PTBorder)
      {
        sum += dst[i_nb[j]];
      }
      sum += static_cast<float>(src[i]) - static_cast<float>(src[i_nb[j]]);
    }

    guess_next[i] = clamp(sum/4.f, 0.f, 255.f);
  }
}


void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */

  int sourceSize = numRowsSource * numColsSource;

  unsigned char* d_mask;
  unsigned char* d_pixelTypes;
  uchar4* d_sourceImg;
  unsigned char* d_src[3];
  unsigned char* d_dst[3];
  float* d_guess[3][2];
  uchar4* d_destImg;
  checkCudaErrors(cudaMalloc(&d_mask, sourceSize * sizeof(unsigned char)));
  checkCudaErrors(cudaMalloc(&d_pixelTypes, sourceSize * sizeof(unsigned char)));
  checkCudaErrors(cudaMalloc(&d_sourceImg, sourceSize * sizeof(uchar4)));
  for(int color = 0; color < 3; color++)
  {
    for(int i = 0; i<2; i++)
      checkCudaErrors(cudaMalloc(&d_guess[color][i], sourceSize * sizeof(float))); 
    checkCudaErrors(cudaMalloc(&d_src[color], sourceSize * sizeof(unsigned char))); 
    checkCudaErrors(cudaMalloc(&d_dst[color], sourceSize * sizeof(unsigned char))); 

  }
  checkCudaErrors(cudaMalloc(&d_destImg, sourceSize * sizeof(uchar4)));

  checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, sourceSize * sizeof(uchar4), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, sourceSize * sizeof(uchar4), cudaMemcpyHostToDevice));

  {
    int block_size = 1024;
    int grid_size = div_up(sourceSize, block_size);
    compute_mask<<<grid_size, block_size>>>(d_sourceImg, d_mask, sourceSize);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  }

  {
    dim3 block_size(32, 32, 1);
    dim3 grid_size(div_up(numColsSource, block_size.x), div_up(numRowsSource, block_size.y), 1);
    compute_pixel_type<<<grid_size, block_size>>>(d_mask, d_pixelTypes, numRowsSource, numColsSource);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  }

  {
    int block_size = 1024;
    int grid_size = div_up(sourceSize, block_size);
    split_channels<<<grid_size, block_size>>>(d_sourceImg, d_src[0], d_src[1], d_src[2], sourceSize);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  }

  {
    int block_size = 1024;
    int grid_size = div_up(sourceSize, block_size);
    split_channels<<<grid_size, block_size>>>(d_destImg, d_dst[0], d_dst[1], d_dst[2], sourceSize);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  }

  {
    int block_size = 1024;
    int grid_size = div_up(sourceSize, block_size);
    split_channels<<<grid_size, block_size>>>(d_sourceImg, d_guess[0][0], d_guess[1][0], d_guess[2][0], sourceSize);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  }

  {
    dim3 block_size(32, 32, 1);
    dim3 grid_size(div_up(numColsSource, block_size.x), div_up(numRowsSource, block_size.y), 1);

    for(int i = 0; i<800; i++)
    {
      for(int color = 0; color<3; color++)
      {
        process<<<grid_size, block_size>>>(d_src[color], d_dst[color], d_pixelTypes,
          d_guess[color][0], d_guess[color][1], numRowsSource, numColsSource);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        std::swap(d_guess[color][0], d_guess[color][1]);
      }
    }
  }

  {
    int block_size = 1024;
    int grid_size = div_up(sourceSize, block_size);
    compose_img<<<grid_size, block_size>>>(d_destImg, d_guess[0][0], d_guess[1][0], d_guess[2][0], d_pixelTypes, sourceSize);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); 
  }

  checkCudaErrors(cudaMemcpy(h_blendedImg, d_destImg, sourceSize * sizeof(uchar4), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_mask));
  checkCudaErrors(cudaFree(d_pixelTypes));
  checkCudaErrors(cudaFree(d_sourceImg));
  for(int color = 0; color < 3; color++)
  {
    for(int i = 0; i<2; i++)
      checkCudaErrors(cudaFree(d_guess[color][i]));
    checkCudaErrors(cudaFree(d_src[color]));
    checkCudaErrors(cudaFree(d_dst[color]));
  }
  checkCudaErrors(cudaFree(d_destImg));
}
