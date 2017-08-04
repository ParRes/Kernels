
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
/// NAME:    Stencil
///
/// PURPOSE: This program tests the efficiency with which a space-invariant,
///          linear, symmetric filter (stencil) can be applied to a square
///          grid or image.
///
/// USAGE:   The program takes as input the linear
///          dimension of the grid, and the number of iterations on the grid
///
///                <progname> <iterations> <grid size>
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// FUNCTIONS CALLED:
///
///          Other than standard C functions, the following functions are used in
///          this program:
///          wtime()
///
/// HISTORY: - Written by Rob Van der Wijngaart, February 2009.
///          - RvdW: Removed unrolling pragmas for clarity;
///            added constant to array "in" at end of each iteration to force
///            refreshing of neighbor data in parallel versions; August 2013
///            C++11-ification by Jeff Hammond, May 2017.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"

#include "stencil_taskloop.hpp"

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/OpenMP TASKLOOP Stencil execution on 2D grid" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations, n, radius, gs;
  bool star = true;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <array dimension> [taskloop grainsize] [<star/grid> <radius>]";
      }

      // number of times to run the algorithm
      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      // linear grid dimension
      n  = std::atoi(argv[2]);
      if (n < 1) {
        throw "ERROR: grid dimension must be positive";
      } else if (n > std::floor(std::sqrt(INT_MAX))) {
        throw "ERROR: grid dimension too large - overflow risk";
      }

      // taskloop grainsize
      gs = (argc > 3) ? std::atoi(argv[3]) : 100;
      if (gs < 1 || gs > n) {
        throw "ERROR: grainsize";
      }

      // stencil pattern
      if (argc > 4) {
          auto stencil = std::string(argv[4]);
          auto grid = std::string("grid");
          star = (stencil == grid) ? false : true;
      }

      // stencil radius
      radius = 2;
      if (argc > 5) {
          radius = std::atoi(argv[5]);
      }

      if ( (radius < 1) || (2*radius+1 > n) ) {
        throw "ERROR: Stencil radius negative or too large";
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

#ifdef _OPENMP
  std::cout << "Number of threads    = " << omp_get_max_threads() << std::endl;
  std::cout << "Taskloop grainsize   = " << gs << std::endl;
#endif
  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Grid size            = " << n << std::endl;
  std::cout << "Type of stencil      = " << (star ? "star" : "grid") << std::endl;
  std::cout << "Radius of stencil    = " << radius << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  auto stencil_time = 0.0;

  std::vector<double> in;
  std::vector<double> out;
  in.resize(n*n);
  out.resize(n*n);

  OMP_PARALLEL()
  OMP_MASTER
  {
    OMP_TASKLOOP( firstprivate(n) shared(in,out) grainsize(gs) )
    for (auto i=0; i<n; i++) {
      OMP_SIMD
      for (auto j=0; j<n; j++) {
        in[i*n+j] = static_cast<double>(i+j);
        out[i*n+j] = 0.0;
      }
    }
    OMP_TASKWAIT

    for (auto iter = 0; iter<=iterations; iter++) {

      if (iter==1) stencil_time = prk::wtime();

      // Apply the stencil operator
      if (star) {
          switch (radius) {
              case 1: star1(n, gs, in, out); break;
              case 2: star2(n, gs, in, out); break;
              case 3: star3(n, gs, in, out); break;
              case 4: star4(n, gs, in, out); break;
              case 5: star5(n, gs, in, out); break;
              case 6: star6(n, gs, in, out); break;
              case 7: star7(n, gs, in, out); break;
              case 8: star8(n, gs, in, out); break;
              case 9: star9(n, gs, in, out); break;
              default: { std::cerr << "star template not instantiated for radius " << radius << "\n"; break; }
          }
      } else {
          switch (radius) {
              case 1: grid1(n, gs, in, out); break;
              case 2: grid2(n, gs, in, out); break;
              case 3: grid3(n, gs, in, out); break;
              case 4: grid4(n, gs, in, out); break;
              case 5: grid5(n, gs, in, out); break;
              case 6: grid6(n, gs, in, out); break;
              case 7: grid7(n, gs, in, out); break;
              case 8: grid8(n, gs, in, out); break;
              case 9: grid9(n, gs, in, out); break;
              default: { std::cerr << "grid template not instantiated for radius " << radius << "\n"; break; }
          }
      }
      OMP_TASKWAIT

      // Add constant to solution to force refresh of neighbor data, if any
      OMP_TASKLOOP( firstprivate(n) shared(in) grainsize(gs) )
      for (auto i=0; i<n; i++) {
        OMP_SIMD
        for (auto j=0; j<n; j++) {
          in[i*n+j] += 1.0;
        }
      }
      OMP_TASKWAIT
    }
    stencil_time = prk::wtime() - stencil_time;
  }

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  // interior of grid with respect to stencil
  size_t active_points = static_cast<size_t>(n-2*radius)*static_cast<size_t>(n-2*radius);

  // compute L1 norm in parallel
  double norm = 0.0;
  OMP_PARALLEL_FOR_REDUCE( +:norm )
  for (auto i=radius; i<n-radius; i++) {
    for (auto j=radius; j<n-radius; j++) {
      norm += std::fabs(out[i*n+j]);
    }
  }
  norm /= active_points;

  // verify correctness
  const double epsilon = 1.0e-8;
  double reference_norm = 2.*(iterations+1.);
  if (std::fabs(norm-reference_norm) > epsilon) {
    std::cout << "ERROR: L1 norm = " << norm
              << " Reference L1 norm = " << reference_norm << std::endl;
    return 1;
  } else {
    std::cout << "Solution validates" << std::endl;
#ifdef VERBOSE
    std::cout << "L1 norm = " << norm
              << " Reference L1 norm = " << reference_norm << std::endl;
#endif
    const int stencil_size = star ? 4*radius+1 : (2*radius+1)*(2*radius+1);
    size_t flops = (2L*(size_t)stencil_size+1L) * active_points;
    auto avgtime = stencil_time/iterations;
    std::cout << "Rate (MFlops/s): " << 1.0e-6 * static_cast<double>(flops)/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  }

  return 0;
}
