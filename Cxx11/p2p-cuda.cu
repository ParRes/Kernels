///
/// Copyright (c) 2020, Intel Corporation
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

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define BLOCK_SIZE 32

#if 0

#define HALO_SIZE 1

__global__ void p2p(int N, double * M)
{
    __shared__ float sm_buffer[BLOCK_SIZE + HALO_SIZE][BLOCK_SIZE + HALO_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int dx = blockDim.x;

    cooperative_groups::grid_group cuda_grid = cooperative_groups::this_grid();

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

#if 0
    // one thread copies the bottom right corner to the top left corner...
    if ((bx * dx + tx) == 0) {
        M[0] = -M[(N-1)*(N+HALO_SIZE)+(N-1)];
    }
    cuda_grid.sync(); // required?
#endif
}
#else
__global__ void p2p(double * grid, const int n)
{
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int dx = blockDim.x;

    const int j = bx * dx + tx + 1;

    cooperative_groups::grid_group cuda_grid = cooperative_groups::this_grid();

    for (int i=2; i<=2*n-2; i++) {
      //parallel_for (int j=std::max(2,i-n+2); j<=std::min(i,n); j++) {
      if (MAX(2,i-n+2) <= j && j <= MIN(i,n)) {
        const int x = i-j+1;
        const int y = j-1;
        grid[x*n+y] = grid[(x-1)*n+y] + grid[x*n+(y-1)] - grid[(x-1)*n+(y-1)];
      }
      //__threadfence();
      cuda_grid.sync();
      //__threadfence();
    }

    // one thread copies the bottom right corner to the top left corner...
    if (j == 1) {
      grid[0*n+0] = -grid[(n-1)*n+(n-1)];
    }
    //__threadfence();
    //cuda_grid.sync(); // required?
    //__threadfence();
}
#endif

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
  int n, nc, nb;
  try {
      if (argc < 3) {
        throw " <# iterations> <array dimension> [<chunk dimension>]";
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
      } else if (n % BLOCK_SIZE) {
        throw "ERROR: grid dimension is not a multiple of BLOCK_SIZE";
      }

      // grid chunk dimensions
      nc = (argc > 3) ? std::atoi(argv[3]) : 1;
      nc = std::max(1,nc);
      nc = std::min(n,nc);

      // number of grid blocks
      nb = (n-1)/nc;
      if ((n-1)%nc) nb++;
      //std::cerr << "n="  << n << std::endl;
      //std::cerr << "nb=" << nb << std::endl;
      //std::cerr << "nc=" << nc << std::endl;
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Grid sizes           = " << n << ", " << n << std::endl;
  std::cout << "Grid chunk sizes     = " << nc << std::endl;

  std::cout << "THIS IMPLEMENTATION IS NOT GOOD!!!!" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double pipeline_time{0}; // silence compiler warning

  const size_t nelems = (size_t)n * (size_t)n;
  const size_t bytes = nelems * sizeof(double);
  double * h_grid;
  prk::CUDA::check( cudaMallocHost((void**)&h_grid, bytes) );

  // initialize boundary conditions
  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) {
      h_grid[i*n+j] = static_cast<double>(0);
    }
  }
  for (int j=0; j<n; j++) {
    h_grid[0*n+j] = static_cast<double>(j);
  }
  for (int i=0; i<n; i++) {
    h_grid[i*n+0] = static_cast<double>(i);
  }

#ifdef DEBUG
  std::cout << "B h_grid=\n";
  for (int i=0; i<n; i++) {
    std::cout << "B ";
    for (int j=0; j<n; j++) {
        std::cout << h_grid[i*n+j] << ",";
    }
    std::cout << "\n";
  }
#endif

  double * d_grid;
  prk::CUDA::check( cudaMalloc((void**)&d_grid, bytes) );
  prk::CUDA::check( cudaMemcpy(d_grid, h_grid, bytes, cudaMemcpyHostToDevice) );
  prk::CUDA::check( cudaDeviceSynchronize() );

  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) {
      h_grid[i*n+j] = static_cast<double>(-777);
    }
  }

  for (int iter = 0; iter<=iterations; iter++) {

    if (iter==1) pipeline_time = prk::wtime();

    dim3 cuda_threads(BLOCK_SIZE);
    dim3 cuda_grid(n / BLOCK_SIZE);
#if 0
    void * kernel_args[2] = { (void*)&n, (void*)&d_grid };
    prk::CUDA::check( cudaLaunchCooperativeKernel((void*)p2p, cuda_grid, cuda_threads, (void**)kernel_args) );
#else
    void * kernel_args[2] = { (void*)&d_grid, (void*)&n };
    prk::CUDA::check( cudaLaunchCooperativeKernel((void*)p2p, cuda_grid, cuda_threads, (void**)kernel_args) );
#endif
    prk::CUDA::check( cudaDeviceSynchronize() );
    prk::CUDA::check( cudaMemcpy(h_grid, d_grid, bytes, cudaMemcpyDeviceToHost) );

#ifdef DEBUG
    std::cout << "h_grid=\n";
    for (int i=0; i<n; i++) {
      for (int j=0; j<n; j++) {
          std::cout << h_grid[i*n+j] << ",";
      }
      std::cout << "\n";
    }
#endif
  }
  prk::CUDA::check( cudaDeviceSynchronize() );
  pipeline_time = prk::wtime() - pipeline_time;

  // copy output back to host
  prk::CUDA::check( cudaMemcpy(h_grid, d_grid, bytes, cudaMemcpyDeviceToHost) );
  prk::CUDA::check( cudaFree(d_grid) );

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  const double epsilon = 1.e-8;
  auto corner_val = ((iterations+1.)*(2.*n-2.));
  if ( (prk::abs(h_grid[(n-1)*n+(n-1)] - corner_val)/corner_val) > epsilon) {
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
