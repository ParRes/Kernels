
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
#include "prk_pstl.h"
// See ParallelSTL.md for important information.
#if defined(USE_PSTL) && ( defined(USE_INTEL_PSTL) || ( defined(__GNUC__) && (__GNUC__ >= 9) ) )
#include "stencil_pstl.hpp"
#elif defined(USE_PSTL) && defined(__GNUC__) && defined(__GNUC_MINOR__) \
                        && ( (__GNUC__ == 8) || (__GNUC__ == 7) && (__GNUC_MINOR__ >= 2) )
#include "stencil_pgnu.hpp"
#else
#include "stencil_stl.hpp"
#endif

void nothing(const int n, const int t, std::vector<double> & in, std::vector<double> & out)
{
    std::cout << "You are trying to use a stencil that does not exist.\n";
    std::cout << "Please generate the new stencil using the code generator\n";
    std::cout << "and add it to the case-switch in the driver." << std::endl;
    // n will never be zero - this is to silence compiler warnings.
    if (n==0 || t==0) std::cout << in.size() << out.size() << std::endl;
    std::abort();
}

int main(int argc, char* argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
#if defined(USE_PSTL)
  std::cout << "C++17/PSTL Stencil execution on 2D grid" << std::endl;
#else
  std::cout << "C++11/STL Stencil execution on 2D grid" << std::endl;
#endif

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations, n, radius, tile_size;
  bool star = true;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <array dimension> [<tile_size> <star/grid> <radius>]";
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

      // default tile size for tiling of local transpose
      tile_size = 32;
      if (argc > 3) {
          tile_size = std::atoi(argv[3]);
          if (tile_size <= 0) tile_size = n;
          if (tile_size > n) tile_size = n;
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

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Grid size            = " << n << std::endl;
  std::cout << "Tile size            = " << tile_size << std::endl;
  std::cout << "Type of stencil      = " << (star ? "star" : "grid") << std::endl;
  std::cout << "Radius of stencil    = " << radius << std::endl;

  auto stencil = nothing;
  if (star) {
      switch (radius) {
          case 1: stencil = star1; break;
          case 2: stencil = star2; break;
          case 3: stencil = star3; break;
          case 4: stencil = star4; break;
          case 5: stencil = star5; break;
      }
  } else {
      switch (radius) {
          case 1: stencil = grid1; break;
          case 2: stencil = grid2; break;
          case 3: stencil = grid3; break;
          case 4: stencil = grid4; break;
          case 5: stencil = grid5; break;
      }
  }

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double stencil_time(0);

  std::vector<double> in(n*n);
  std::vector<double> out(n*n);

  // initialize the input and output arrays
  auto range = prk::range(0,n);
#if defined(USE_PSTL) && ( defined(USE_INTEL_PSTL) || ( defined(__GNUC__) && (__GNUC__ >= 9) ) )
  std::for_each( exec::par, std::begin(range), std::end(range), [&] (int i) {
    std::for_each( exec::unseq, std::begin(range), std::end(range), [&] (int j) {
#elif defined(USE_PSTL) && defined(__GNUC__) && defined(__GNUC_MINOR__) \
                        && ( (__GNUC__ == 8) || (__GNUC__ == 7) && (__GNUC_MINOR__ >= 2) )
  __gnu_parallel::for_each( std::begin(range), std::end(range), [&] (int i) {
    __gnu_parallel::for_each( std::begin(range), std::end(range), [&] (int j) {
#else
  std::for_each( std::begin(range), std::end(range), [&] (int i) {
    std::for_each( std::begin(range), std::end(range), [&] (int j) {
#endif
      in[i*n+j] = static_cast<double>(i+j);
      out[i*n+j] = 0.0;
    });
  });

  for (int iter = 0; iter<=iterations; iter++) {
    if (iter==1) stencil_time = prk::wtime();
    // Apply the stencil operator
    stencil(n, tile_size, in, out);
    // Add constant to solution to force refresh of neighbor data, if any
#if defined(USE_PSTL) && ( defined(USE_INTEL_PSTL) || ( defined(__GNUC__) && (__GNUC__ >= 9) ) )
    std::transform( exec::par_unseq, in.begin(), in.end(), in.begin(), [](double c) { return c+=1.0; });
#elif defined(USE_PSTL) && defined(__GNUC__) && defined(__GNUC_MINOR__) \
                        && ( (__GNUC__ == 8) || (__GNUC__ == 7) && (__GNUC_MINOR__ >= 2) )
    __gnu_parallel::transform( in.begin(), in.end(), in.begin(), [](double c) { return c+=1.0; });
#else
    std::transform( in.begin(), in.end(), in.begin(), [](double c) { return c+=1.0; });
#endif
  }

  stencil_time = prk::wtime() - stencil_time;

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  // interior of grid with respect to stencil
  size_t active_points = static_cast<size_t>(n-2*radius)*static_cast<size_t>(n-2*radius);

  // compute L1 norm in parallel
  double norm = 0.0;
  auto inside = prk::range(radius,n-radius);
  for (auto i : inside) {
    for (auto j : inside) {
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
