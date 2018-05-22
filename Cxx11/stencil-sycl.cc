
///
/// Copyright (c) 2017, Intel Corporation
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

#include "CL/sycl.hpp"

#include "prk_util.h"
#include "stencil_sycl.hpp"

#if USE_2D_INDEXING
void nothing(cl::sycl::queue & q, const size_t n, cl::sycl::buffer<double, 2> & d_in, cl::sycl::buffer<double, 2> & d_out)
#else
void nothing(cl::sycl::queue & q, const size_t n, cl::sycl::buffer<double> & d_in, cl::sycl::buffer<double> & d_out)
#endif
{
    std::cout << "You are trying to use a stencil that does not exist.\n";
    std::cout << "Please generate the new stencil using the code generator\n";
    std::cout << "and add it to the case-switch in the driver." << std::endl;
    std::abort();
}

int main(int argc, char* argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/SYCL Stencil execution on 2D grid" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  size_t n, tile_size;
  bool star = true;
  size_t radius = 2;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <array dimension> [<tile size> <star/grid> <stencil radius>]";
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
  }
#if 0
  else {
      switch (radius) {
          case 1: stencil = grid1; break;
          case 2: stencil = grid2; break;
          case 3: stencil = grid3; break;
          case 4: stencil = grid4; break;
          case 5: stencil = grid5; break;
      }
  }
#endif

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  auto stencil_time = 0.0;

  std::vector<double> h_in(n*n,0.0);
  std::vector<double> h_out(n*n,0.0);

  // SYCL device queue
  cl::sycl::queue q;
  {
    // initialize device buffers from host buffers
#if USE_2D_INDEXING
    cl::sycl::buffer<double, 2> d_in  { cl::sycl::range<2> {n, n} };
    cl::sycl::buffer<double, 2> d_out { h_out.data(), cl::sycl::range<2> {n, n} };
#else
    // FIXME: if I don't initialize this buffer from host, the results are wrong.  Why?
    //cl::sycl::buffer<double> d_in  { cl::sycl::range<1> {n*n} };
    cl::sycl::buffer<double> d_in  { h_in.data(),  h_in.size() };
    cl::sycl::buffer<double> d_out { h_out.data(), h_out.size() };
#endif

    q.submit([&](cl::sycl::handler& h) {

      // accessor methods
      auto in  = d_in.get_access<cl::sycl::access::mode::read_write>(h);

      h.parallel_for<class init>(cl::sycl::range<2> {n, n}, [=] (cl::sycl::item<2> it) {
#if USE_2D_INDEXING
          cl::sycl::id<2> xy = it.get_id();
          auto i = it[0];
          auto j = it[1];
          in[xy] = static_cast<double>(i+j);
#else
          auto i = it[0];
          auto j = it[1];
          in[i*n+j] = static_cast<double>(i+j);
#endif
      });
    });
    q.wait();

    for (auto iter = 0; iter<=iterations; iter++) {

      if (iter==1) stencil_time = prk::wtime();

      stencil(q, n, d_in, d_out);
      // This is only necessary with triSYCL
      q.wait();

      q.submit([&](cl::sycl::handler& h) {

        // accessor methods
        auto in  = d_in.get_access<cl::sycl::access::mode::read_write>(h);

        // Add constant to solution to force refresh of neighbor data, if any
        h.parallel_for<class add>(cl::sycl::range<2> {n, n}, cl::sycl::id<2> {0, 0},
                                  [=] (cl::sycl::item<2> it) {
#if USE_2D_INDEXING
            cl::sycl::id<2> xy = it.get_id();
            in[xy] += 1.0;
#else
#if 0 // This is noticeably slower :-(
            auto i = it[0];
            auto j = it[1];
            in[i*n+j] += 1.0;
#else
            in[it[0]*n+it[1]] += 1.0;
#endif
#endif
        });
      });
      q.wait();
    }
    stencil_time = prk::wtime() - stencil_time;
  }

#if 0
  for (auto i=0; i<n; i++) {
    for (auto j=0; j<n; j++) {
        std::cerr << i << "," << j << "," << h_out[i*n+j] << "\n";
    }
  }
#endif

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  // interior of grid with respect to stencil
  auto active_points = (n-2L*radius)*(n-2L*radius);

  // compute L1 norm in parallel
  double norm = 0.0;
  for (auto i=radius; i<n-radius; i++) {
    for (auto j=radius; j<n-radius; j++) {
      norm += std::fabs(h_out[i*n+j]);
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
    const size_t stencil_size = star ? 4*radius+1 : (2*radius+1)*(2*radius+1);
    size_t flops = (2L*stencil_size+1L) * active_points;
    auto avgtime = stencil_time/iterations;
    std::cout << "Rate (MFlops/s): " << 1.0e-6 * static_cast<double>(flops)/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  }

  return 0;
}
