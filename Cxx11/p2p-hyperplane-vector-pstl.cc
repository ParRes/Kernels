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
#include "prk_pstl.h"
#include "p2p-kernel.h"

int main(int argc, char* argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
#if defined(USE_PSTL)
  std::cout << "C++17/PSTL HYPERPLANE pipeline execution on 2D grid" << std::endl;
#else
  std::cout << "C++11/STL HYPERPLANE pipeline execution on 2D grid" << std::endl;
#endif

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
      } else if ( static_cast<size_t>(n)*static_cast<size_t>(n) > static_cast<size_t>(INT_MAX)) {
        throw "ERROR: grid dimension too large - overflow risk";
      }

      // grid chunk dimensions
      nc = (argc > 3) ? std::atoi(argv[3]) : 1;
      nc = std::max(1,nc);
      nc = std::min(n,nc);

      // number of grid blocks
      nb = (n-1)/nc;
      if ((n-1)%nc) nb++;
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

  double pipeline_time(0);

  std::vector<double> grid(n*n,0.0);

  // set boundary values (bottom and left side of grid)
  for (auto j=0; j<n; j++) {
    grid[0*n+j] = static_cast<double>(j);
    grid[j*n+0] = static_cast<double>(j);
  }

  for (int iter = 0; iter<=iterations; iter++) {

    if (iter==1) pipeline_time = prk::wtime();

    if (nc==1) {
      for (auto i=2; i<=2*n-2; i++) {
        const auto begin = std::max(2,i-n+2);
        const auto end   = std::min(i,n)+1;
        auto range = prk::range(begin,end);
#if defined(USE_PSTL) && defined(USE_INTEL_PSTL)
        std::for_each( exec::par, std::begin(range), std::end(range), [&] (auto j) {
#elif defined(USE_PSTL) && defined(__GNUC__) && defined(__GNUC_MINOR__) \
                        && ( (__GNUC__ == 8) || (__GNUC__ == 7) && (__GNUC_MINOR__ >= 2) )
        __gnu_parallel::for_each( std::begin(range), std::end(range), [&] (auto j) {
#else
        std::for_each( std::begin(range), std::end(range), [&] (auto j) {
#endif
          const auto x = i-j+1;
          const auto y = j-1;
          grid[x*n+y] = grid[(x-1)*n+y] + grid[x*n+(y-1)] - grid[(x-1)*n+(y-1)];
        });
      }
    } else {
      for (int i=2; i<=2*(nb+1)-2; i++) {
        const auto begin = std::max(2,i-(nb+1)+2);
        const auto end   = std::min(i,nb+1)+1;
        auto range = prk::range(begin,end);
#if defined(USE_PSTL) && ( defined(USE_INTEL_PSTL) || ( defined(__GNUC__) && (__GNUC__ >= 9) ) )
        std::for_each( exec::par, std::begin(range), std::end(range), [&] (auto j) {
#elif defined(USE_PSTL) && defined(__GNUC__) && defined(__GNUC_MINOR__) \
                        && ( (__GNUC__ == 8) || (__GNUC__ == 7) && (__GNUC_MINOR__ >= 2) )
        __gnu_parallel::for_each( std::begin(range), std::end(range), [&] (auto j) {
#else
        std::for_each( std::begin(range), std::end(range), [&] (auto j) {
#endif
          const int ib = nc*(i-j)+1;
          const int jb = nc*(j-2)+1;
          sweep_tile(ib, std::min(n,ib+nc), jb, std::min(n,jb+nc), n, grid);
        });
      }
    }
    grid[0*n+0] = -grid[(n-1)*n+(n-1)];
  }

  pipeline_time = prk::wtime() - pipeline_time;

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
