#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

#include "gputimer.h"

int main(void)
{
  // generate N random numbers serially
  int N = 100000;
  thrust::host_vector<uint8_t> h_vec(N);
  std::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer data to the device
  thrust::device_vector<uint8_t> d_vec = h_vec;

  // sort data on the device (846M keys per second on GeForce GTX 480)
  GpuTimer timer;
  timer.Start();
  thrust::sort(d_vec.begin(), d_vec.end());
  timer.Stop();

  // transfer data back to host
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
  
  printf("Thrust sorted %d keys in %g ms\n", N, timer.Elapsed());
  return 0;
}