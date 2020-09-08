///
/// Copyright (c) 2018, Intel Corporation
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions
/// are met:
///
/// * Redistributions of source code must retain the above copyright
///       notice, this list of conditions and the following disclaimer.
/// * Redistributions in binary form must reproduce the above
///       copyright notice, this list of conditions and the following
///       disclaimer in the documentation and/or other materials provided
///       with the distribution.
/// * Neither the name of Intel Corporation nor the names of its
///       contributors may be used to endorse or promote products
///       derived from this software without specific prior written
///       permission.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
/// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
/// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
/// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
/// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
/// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
/// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
/// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
/// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
/// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
/// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
/// POSSIBILITY OF SUCH DAMAGE.

//////////////////////////////////////////////////////////////////////
///
/// NAME:    Pipeline
///
/// PURPOSE: This program tests the efficiency with which point-to-point
///          synchronization can be carried out. It does so by executing
///          a pipelined algorithm on an n^2 grid. The first array dimension
///          is distributed among the threads (stripwise decomposition).
///
/// USAGE:   The program takes as input the
///          dimensions of the grid, and the number of iterations on the grid
///
///                <progname> <iterations> <n>
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// FUNCTIONS CALLED:
///
///          Other than standard C functions, the following
///          functions are used in this program:
///
///          wtime()
///
/// HISTORY: - Written by Rob Van der Wijngaart, February 2009.
///            C99-ification by Jeff Hammond, February 2016.
///            C++11-ification by Jeff Hammond, May 2017.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_cuda.h"

#include <cooperative_groups.h>

//using namespace cooperative_groups;

#define HALO_SIZE 1
#define BLOCK_SIZE 32

__global__ void p2p(int N, double * M)
{
    __shared__ float sm_buffer[BLOCK_SIZE + HALO_SIZE][BLOCK_SIZE + HALO_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    cooperative_groups::grid_group cuda_grid = this_grid();

    for(int i = 0; i < 2*N/BLOCK_SIZE; ++i) {

         //Compute matrix coordinates for block corner
        int g_x = bx * BLOCK_SIZE + HALO_SIZE;
        int g_y = (i - bx) * BLOCK_SIZE + HALO_SIZE;

        //Check block is in bounds
        if(g_y >= 0 && g_y < (N + HALO_SIZE)) {

            // load halo to SM
            sm_buffer[0][tx + 1] = M[(g_y - 1) * (N + HALO_SIZE) + g_x + tx];
            sm_buffer[tx + 1][0] = M[(g_y + tx)* (N + HALO_SIZE) + g_x - 1];
            if(tx == 0) //Load corner
                sm_buffer[0][0] = M[(g_y - 1) * (N + HALO_SIZE) + g_x - 1];

            // inner loop
            for(int j = 0; j <= 2*BLOCK_SIZE; j++) {
                int l_x = tx + HALO_SIZE;
                int l_y = j - tx + HALO_SIZE;
                if (l_y >= 1 && l_y <= BLOCK_SIZE)
                    sm_buffer[l_y][l_x] = sm_buffer[l_y - 1][l_x] + sm_buffer[l_y][l_x - 1] - sm_buffer[l_y - 1][l_x - 1];
            }

            // flush block to memory
            for(int j = 0; j <= BLOCK_SIZE; j++)
                M[(g_y + j) * (N + HALO_SIZE) + g_x + tx] = sm_buffer[j + HALO_SIZE][tx + HALO_SIZE];

            // sync threads
            cuda_grid.sync();
        }
    }
}

int main(int argc, char* argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/CUDA cooperative groups pipeline execution on 2D grid" << std::endl;

  prk::CUDA::info info;
  info.print();

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int n;
  try {
      if (argc < 3) {
        throw " <# iterations> <array dimension>";
      }

      // number of times to run the pipeline algorithm
      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      // grid dimensions
      n = std::atoi(argv[2]);
      if (n < 1) {
        throw "ERROR: grid dimensions must be positive";
      } else if ( n > prk::get_max_matrix_size() ) {
        throw "ERROR: grid dimension too large - overflow risk";
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Grid sizes           = " << n << ", " << n << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  auto pipeline_time = 0.0; // silence compiler warning

  const size_t nelems = (size_t)n * (size_t)n;
  const size_t bytes = nelems * sizeof(double);
  double * h_grid;
  prk::CUDA::check( cudaMallocHost((void**)&h_grid, bytes) );

  // initialize boundary conditions
  for (auto i=0; i<n; i++) {
    for (auto j=0; j<n; j++) {
      h_grid[i*n+j] = static_cast<double>(0);
    }
  }
  for (auto j=0; j<n; j++) {
    h_grid[0*n+j] = static_cast<double>(j);
  }
  for (auto i=0; i<n; i++) {
    h_grid[i*n+0] = static_cast<double>(i);
  }

  double * d_grid;
  prk::CUDA::check( cudaMalloc((void**)&d_grid, bytes) );
  prk::CUDA::check( cudaMemcpy(d_grid, &(h_grid[0]), bytes, cudaMemcpyHostToDevice) );

  for (auto iter = 0; iter<=iterations; iter++) {

    if (iter==1) pipeline_time = prk::wtime();

    dim3 cuda_threads(BLOCK_SIZE);
    dim3 cuda_grid(n / BLOCK_SIZE);
    void * kernel_args[2] = { (void*)n, (void*)&d_grid };
    cudaLaunchCooperativeKernel((void*)p2p, cuda_grid, cuda_threads, (void**)kernel_args);

    //prk::CUDA::check( cudaDeviceSynchronize() );
  }
  pipeline_time = prk::wtime() - pipeline_time;

  // copy output back to host
  prk::CUDA::check( cudaMemcpy(&(h_grid[0]), d_grid, bytes, cudaMemcpyDeviceToHost) );
  prk::CUDA::check( cudaFree(d_grid) );

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  const double epsilon = 1.e-8;
  auto corner_val = ((iterations+1.)*(2.*n-2.));
  if ( (std::fabs(h_grid[(n-1)*n+(n-1)] - corner_val)/corner_val) > epsilon) {
    std::cout << "ERROR: checksum " << h_grid[(n-1)*n+(n-1)]
              << " does not match verification value " << corner_val << std::endl;
    return 1;
  }

#ifdef VERBOSE
  std::cout << "Solution validates; verification value = " << corner_val << std::endl;
#else
  std::cout << "Solution validates" << std::endl;
#endif
  auto avgtime = pipeline_time/iterations;
  std::cout << "Rate (MFlops/s): "
            << 2.0e-6 * ( (n-1.)*(n-1.) )/avgtime
            << " Avg time (s): " << avgtime << std::endl;

  return 0;
}
