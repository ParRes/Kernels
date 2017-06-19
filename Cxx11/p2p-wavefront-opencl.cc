///
/// Copyright (c) 2013, Intel Corporation
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

#include <omp.h>

#include "prk_util.h"
#include "prk_opencl.h"

int main(int argc, char* argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/OpenCL WAVEFRONT pipeline execution on 2D grid" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  if (argc < 3){
    std::cout << "Usage: " << argv[0] << " <# iterations> <array dimension>" << std::endl;
    return(EXIT_FAILURE);
  }

  // number of times to run the pipeline algorithm
  int iterations  = std::atoi(argv[1]);
  if (iterations < 1){
    std::cout << "ERROR: iterations must be >= 1 : " << iterations << std::endl;
    exit(EXIT_FAILURE);
  }

  // grid dimensions
  int n = std::atoi(argv[2]);
  if (n < 1) {
    std::cout << "ERROR: grid dimensions must be positive: " << n << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Number of iterations      = " << iterations << std::endl;
  std::cout << "Grid sizes                = " << n << ", " << n << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Setup OpenCL environment
  //////////////////////////////////////////////////////////////////////

  // FIXME: allow other options here
  cl::Context context(CL_DEVICE_TYPE_DEFAULT);

  cl::Program program(context, prk::loadProgram("p2p.cl"), true);

  auto kernel = cl::make_kernel<int, int, int, cl::Buffer>(program, "p2p");

  cl::CommandQueue queue(context);

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

  // working set
  std::vector<float> h_grid;
  h_grid.resize(n*n,0.0f);

  // set boundary values (bottom and left side of grid)
  for (auto j=0; j<n; j++) {
    h_grid[0*n+j] = static_cast<float>(j);
  }
  for (auto i=0; i<n; i++) {
    h_grid[i*n+0] = static_cast<float>(i);
  }

  // copy input from host to device
  cl::Buffer d_grid = cl::Buffer(context, begin(h_grid), end(h_grid), true);

  auto pipeline_time = 0.0; // silence compiler warning

  for (auto iter = 0; iter<=iterations; iter++) {

    if (iter==1) pipeline_time = prk::wtime();

    for (auto j=1; j<n; j++) {
      kernel(cl::EnqueueArgs(queue, cl::NDRange(n)), 0, j, n, d_grid);
      queue.finish();
    }
    for (auto j=n-2; j>=1; j--) {
      kernel(cl::EnqueueArgs(queue, cl::NDRange(n)), 1, j, n, d_grid);
      queue.finish();
    }
    kernel(cl::EnqueueArgs(queue, cl::NDRange(n)), 2, n, n, d_grid);
    queue.finish();
  }

  pipeline_time = prk::wtime() - pipeline_time;

  // copy output back to host
  cl::copy(queue, d_grid, begin(h_grid), end(h_grid));

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  // error tolerance
  const auto epsilon = 1.0e-4f;

  // verify correctness, using top right value
  auto corner_val = ((iterations+1.0f)*(2.0f*n-2.0f));
  if ( (std::fabs(h_grid[(n-1)*n+(n-1)] - corner_val)/corner_val) > epsilon) {
    std::cout << "ERROR: checksum " << h_grid[(n-1)*n+(n-1)]
              << " does not match verification value " << corner_val << std::endl;
    exit(EXIT_FAILURE);
  }

#ifdef VERBOSE
  std::cout << "Solution validates; verification value = " << corner_val << std::endl;
#else
  std::cout << "Solution validates" << std::endl;
#endif
  auto avgtime = pipeline_time/iterations;
  std::cout << "Rate (MFlops/s): "
            << 1.0e-6 * 2. * ( static_cast<size_t>(n-1)*static_cast<size_t>(n-1) )/avgtime
            << " Avg time (s): " << avgtime << std::endl;

  return 0;
}
