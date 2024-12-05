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
/// HISTORY: - Written by Rob Van der Wijngaart, February 2009.
///            C99-ification by Jeff Hammond, February 2016.
///            C++11-ification by Jeff Hammond, May 2017.
///            CUDASTF version by Cedric Augonnet, November 2024.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_cuda.h"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

#define BLOCK_SIZE 32

int main(int argc, char* argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/CUDASTF pipeline execution on 2D grid" << std::endl;

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
      } else if (n % BLOCK_SIZE) {
        throw "ERROR: grid dimension is not a multiple of BLOCK_SIZE";
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

  double pipeline_time{0}; // silence compiler warning

  context ctx;
  auto grid = ctx.logical_data(shape_of<slice<double, 2>>(n, n));

  ctx.parallel_for(grid.shape(), grid.write())->*[] __device__ (size_t i, size_t j, auto d_grid)
  {
      d_grid(i, j) = 0.0;
  };

  // initialize boundary conditions
  ctx.parallel_for(box(n), grid.write())->*[] __device__ (size_t i, auto d_grid)
  {
      d_grid(i, 0) = static_cast<double>(i);
      d_grid(0, i) = static_cast<double>(i);
  };

#ifdef DEBUG
  ctx.host_launch(grid.read())->*[n](auto h_grid) {
  std::cout << "B h_grid=\n";
  for (int i=0; i<n; i++) {
    std::cout << "B ";
    for (int j=0; j<n; j++) {
        std::cout << h_grid(i, j) << ",";
    }
    std::cout << "\n";
  }
  };
#endif

  for (int iter = 0; iter<=iterations; iter++) {

    if (iter==1) {
        cudaStreamSynchronize(ctx.task_fence());
        pipeline_time = prk::wtime();
    }

    auto spec = con(n/BLOCK_SIZE, con<BLOCK_SIZE>(hw_scope::block));
    ctx.launch(spec, grid.rw())->*[n]__device__(auto th, auto d_grid) {
        for (int i=2; i<=2*n-2; i++) {
          for (int j = th.rank() + 1; j <= n; j += th.size()) {
              if (MAX(2,i-n+2) <= j && j <= MIN(i,n)) {
                const int x = i-j+1;
                const int y = j-1;
                d_grid(x, y) = d_grid(x - 1, y) + d_grid(x, y-1) - d_grid(x-1, y-1);
              }
          }

          th.sync();
        }

        // one thread copies the bottom right corner to the top left corner...
        if (th.rank() == 0) {
          d_grid(0, 0) = -d_grid(n-1, n-1);
        }
    };

#ifdef DEBUG
    ctx.host_launch(grid.read())->*[n](auto h_grid) {
        std::cout << "h_grid=\n";
        for (int i=0; i<n; i++) {
          for (int j=0; j<n; j++) {
              std::cout << h_grid(i, j) << ",";
          }
          std::cout << "\n";
        }
    };
#endif
  }

  cudaStreamSynchronize(ctx.task_fence());
  pipeline_time = prk::wtime() - pipeline_time;

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  const double epsilon = 1.e-8;
  auto corner_val = ((iterations+1.)*(2.*n-2.));

  double corner_res = -1.0;
  ctx.host_launch(grid.read())->*[&](auto h_grid) {
      corner_res = h_grid(n - 1, n - 1);
  };

  cudaStreamSynchronize(ctx.task_fence());

  if ( (prk::abs(corner_res - corner_val)/corner_val) > epsilon) {
    std::cout << "ERROR: checksum " << corner_res
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

  ctx.finalize();



  return 0;
}
