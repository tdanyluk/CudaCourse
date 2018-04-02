#include <stdio.h>
#include "gputimer.h"

void print_array(int *array, int size)
{
    printf("{ ");
    for (int i = 0; i < size; i++)  { printf("%d ", array[i]); }
    printf("}\n");
}

__global__ void increment_naive(int *g, int array_size)
{
	// which thread is this?
	int i = blockIdx.x * blockDim.x + threadIdx.x; 

	// each thread to increment consecutive elements, wrapping at array_size
	i = i % array_size;  
	g[i] = g[i] + 1;
}

__global__ void increment_atomic(int *g, int array_size)
{
	// which thread is this?
	int i = blockIdx.x * blockDim.x + threadIdx.x; 

	// each thread to increment consecutive elements, wrapping at array_size
	i = i % array_size;  
	atomicAdd(& g[i], 1);
}

int main(int argc,char **argv)
{   
    for(int i = 0; i<5; i++)
    {
        GpuTimer timer;
        int block_width = 1000;
        
        int num_threads;
        int array_size;
        bool atomically = !(i == 0 || i == 2);

        if(i == 0 || i == 1)
        {
            num_threads = 1000000;
            array_size = 1000000;
        }
        else if (i == 2 || i == 3)
        {
            num_threads = 1000000;
            array_size = 100;
        }
        else
        {
            num_threads = 10000000;
            array_size = 100;
        }

        printf("%d total threads in %d blocks writing into %d array elements %s\n",
            num_threads, num_threads / block_width, array_size, atomically?"atomically":"");

        const int array_bytes = array_size * sizeof(int);
        int* h_array = (int*)malloc(array_bytes);
    
        int * d_array;
        cudaMalloc((void **) &d_array, array_bytes);
        cudaMemset((void *) d_array, 0, array_bytes); 

        timer.Start();
        if(!atomically)
        {
            increment_naive<<<num_threads/block_width, block_width>>>(d_array, array_size);
        }
        else
        {
            increment_atomic<<<num_threads/block_width, block_width>>>(d_array, array_size);
        }
        timer.Stop();
        
        cudaMemcpy(h_array, d_array, array_bytes, cudaMemcpyDeviceToHost);
        print_array(h_array, min(array_size,10));
        printf("Time elapsed = %g ms\n", timer.Elapsed());
    
        cudaFree(d_array);
        free(h_array);
    }
    return 0;
}