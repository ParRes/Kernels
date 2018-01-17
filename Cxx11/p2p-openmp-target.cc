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
///          a pipelined algorithm on an m*n grid. The first array dimension
///          is distributed among the threads (stripwise decomposition).
///
/// USAGE:   The program takes as input the
///          dimensions of the grid, and the number of iterations on the grid
///
///                <progname> <iterations> <m> <n>
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

int main(int argc, char* argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/OpenMP TARGET DOACROSS pipeline execution on 2D grid" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int m, n;
  try {
      if (argc < 4){
        throw " <# iterations> <first array dimension> <second array dimension>";
      }

      // number of times to run the pipeline algorithm
      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      // grid dimensions
      m = std::atoi(argv[2]);
      n = std::atoi(argv[3]);
      if (m < 1 || n < 1) {
        throw "ERROR: grid dimensions must be positive";
      } else if ( static_cast<size_t>(m)*static_cast<size_t>(n) > INT_MAX) {
        throw "ERROR: grid dimension too large - overflow risk";
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of threads (max)   = " << omp_get_max_threads() << std::endl;
  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Grid sizes           = " << m << ", " << n << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  auto pipeline_time = 0.0; // silence compiler warning

  // working set
  double * grid = new double[m*n];

  OMP_PARALLEL()
  {
    OMP_FOR()
    for (auto i=0; i<n; i++) {
      for (auto j=0; j<n; j++) {
        grid[i*n+j] = 0.0;
      }
    }

    // set boundary values (bottom and left side of grid)
    OMP_MASTER
    {
      for (auto j=0; j<n; j++) {
        grid[0*n+j] = static_cast<double>(j);
      }
      for (auto i=0; i<m; i++) {
        grid[i*n+0] = static_cast<double>(i);
      }
    }
    OMP_BARRIER
  }

  OMP_TARGET( data map(tofrom:grid[0:m*n]) map(from:pipeline_time) )
  {
    for (auto iter = 0; iter<=iterations; iter++) {

      if (iter==1) pipeline_time = omp_get_wtime();

      OMP_PARALLEL() {
        OMP_FOR( collapse(2) ordered(2) )
        for (auto i=1; i<m; i++) {
          for (auto j=1; j<n; j++) {
            OMP_ORDERED( depend(sink: i-1,j) depend(sink: i,j-1) depend(sink: i-1,j-1) )
            grid[i*n+j] = grid[(i-1)*n+j] + grid[i*n+(j-1)] - grid[(i-1)*n+(j-1)];
            OMP_ORDERED( depend (source) )
          }
        }

        OMP_MASTER
        grid[0*n+0] = -grid[(m-1)*n+(n-1)];
      }
    }

    pipeline_time = omp_get_wtime() - pipeline_time;
  }

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  // error tolerance
  const double epsilon = 1.e-8;

  // verify correctness, using top right value
  auto corner_val = ((iterations+1.)*(n+m-2.));
  if ( (std::fabs(grid[(m-1)*n+(n-1)] - corner_val)/corner_val) > epsilon) {
    std::cout << "ERROR: checksum " << grid[(m-1)*n+(n-1)]
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
            << 2.0e-6 * ( (m-1.)*(n-1.) )/avgtime
            << " Avg time (s): " << avgtime << std::endl;

  return 0;
}
