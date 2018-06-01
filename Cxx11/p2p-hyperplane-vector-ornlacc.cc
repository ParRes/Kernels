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

#include "prk_util.h"
#include "p2p-kernel.h"

int main(int argc, char* argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/ORNL-ACC HYPERPLANE pipeline execution on 2D grid" << std::endl;

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
      if (iterations < 0) {
        throw "ERROR: iterations must be >= 1";
      }

      // grid dimensions
      n = std::atoi(argv[2]);
      if (n < 1) {
        throw "ERROR: grid dimensions must be positive";
      } else if ( static_cast<size_t>(n)*static_cast<size_t>(n) > INT_MAX) {
        throw "ERROR: grid dimension too large - overflow risk";
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

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  auto pipeline_time = 0.0;

  double * grid = new double[n*n];

  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) {
      grid[i*n+j] = 0.0;
    }
  }
  for (int j=0; j<n; j++) {
    grid[0*n+j] = static_cast<double>(j);
  }
  for (int i=0; i<n; i++) {
    grid[i*n+0] = static_cast<double>(i);
  }

  #pragma acc data pcopy(grid[0:n*n])
  {
    for (auto iter = 0; iter<=iterations; iter++) {

      if (iter==1) pipeline_time = prk::wtime();

      if (nc==1) {
        for (int i=2; i<=2*n-2; i++) {
          #pragma acc parallel loop independent
          for (int j=std::max(2,i-n+2); j<=std::min(i,n); j++) {
            const int x = i-j+1;
            const int y = j-1;
            grid[x*n+y] = grid[(x-1)*n+y] + grid[x*n+(y-1)] - grid[(x-1)*n+(y-1)];
          }
        }
      } else {
        for (int i=2; i<=2*(nb+1)-2; i++) {
          #pragma acc parallel loop gang
          for (int j=std::max(2,i-(nb+1)+2); j<=std::min(i,nb+1); j++) {
            const int ib = nc*(i-j)+1;
            const int jb = nc*(j-2)+1;
            //sweep_tile(ib, std::min(n,ib+nc), jb, std::min(n,jb+nc), n, grid);
            #pragma acc loop vector
            for (int i=ib; i<std::min(n,ib+nc); i++) {
              for (int j=jb; j<std::min(n,jb+nc); j++) {
                grid[i*n+j] = grid[(i-1)*n+j] + grid[i*n+(j-1)] - grid[(i-1)*n+(j-1)];
              }
            }
          }
        }
      }
      #pragma acc kernels
      {
        grid[0*n+0] = -grid[(n-1)*n+(n-1)];
      }
    }
    pipeline_time = prk::wtime() - pipeline_time;
  }

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  const double epsilon = 1.e-8;
  auto corner_val = ((iterations+1.)*(2.*n-2.));
  if ( (std::fabs(grid[(n-1)*n+(n-1)] - corner_val)/corner_val) > epsilon) {
    std::cout << "ERROR: checksum " << grid[(n-1)*n+(n-1)]
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
