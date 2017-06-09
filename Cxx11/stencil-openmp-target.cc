
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

const int radius = RADIUS;

_Pragma("omp declare target")
template <int radius, bool star>
void do_stencil(int n, double weight[2*radius+1][2*radius+1], double * RESTRICT in, double * RESTRICT out)
{
    _Pragma("omp for")
    for (auto i=radius; i<n-radius; i++) {
      for (auto j=radius; j<n-radius; j++) {
        if (star) {
          for (auto jj=-radius; jj<=radius; jj++) {
            out[i*n+j] += weight[radius][radius+jj]*in[i*n+j+jj];
          }
          for (auto ii=-radius; ii<0; ii++) {
            out[i*n+j] += weight[radius+ii][radius]*in[(i+ii)*n+j];
          }
          for (auto ii=1; ii<=radius; ii++) {
            out[i*n+j] += weight[radius+ii][radius]*in[(i+ii)*n+j];
          }
        } else {
          for (auto ii=-radius; ii<=radius; ii++) {
            for (auto jj=-radius; jj<=radius; jj++) {
              out[i*n+j] += weight[radius+ii][radius+jj]*in[(i+ii)*n+j+jj];
            }
          }
        }
      }
    }
}
_Pragma("omp end declare target")

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/OpenMP TARGET Stencil execution on 2D grid" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // process and test input parameters
  //////////////////////////////////////////////////////////////////////

  if (argc != 3 && argc !=4){
    std::cout << "Usage: " << argv[0] << " <# iterations> <array dimension>" << std::endl;
    return(EXIT_FAILURE);
  }

  // number of times to run the algorithm
  int iterations  = std::atoi(argv[1]);
  if (iterations < 1){
    std::cout << "ERROR: iterations must be >= 1" << iterations << std::endl;
    exit(EXIT_FAILURE);
  }

  // linear grid dimension
  int n  = std::atoi(argv[2]);
  if (n < 1){
    std::cout << "ERROR: grid dimension must be positive: " << n << std::endl;
    exit(EXIT_FAILURE);
  } else if (n > std::floor(std::sqrt(INT_MAX))) {
    std::cout << "ERROR: grid dimension too large - overflow risk: " << n << std::endl;
    exit(EXIT_FAILURE);
  }

  if (radius < 1) {
    std::cout << "ERROR: Stencil radius " << radius << " should be positive " << std::endl;
    exit(EXIT_FAILURE);
  } else if (2*radius+1 > n) {
    std::cout << "ERROR: Stencil radius " << radius << " exceeds grid size " << n << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Grid size            = " << n << std::endl;
  std::cout << "Radius of stencil    = " << radius << std::endl;
#ifdef STAR
  std::cout << "Type of stencil      = star" << std::endl;
#else
  std::cout << "Type of stencil      = compact" << std::endl;
#endif
  std::cout << "Data type            = double precision" << std::endl;
  std::cout << "Compact representation of stencil loop body" << std::endl;
  std::cout << "Number of iterations = " << iterations << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  // weights of points in the stencil
  //std::array< std::array<double,2*radius+1>, 2*radius+1> weight;
  double weight[2*radius+1][2*radius+1];
  for (auto jj=-radius; jj<=radius; jj++) {
    for (auto ii=-radius; ii<=radius; ii++) {
      weight[ii+radius][jj+radius] = 0.0;
    }
  }

  // fill the stencil weights to reflect a discrete divergence operator
#ifdef STAR
  const int stencil_size = 4*radius+1;
  for (auto ii=1; ii<=radius; ii++) {
    weight[radius][radius+ii] = weight[radius+ii][radius] = +1./(2*ii*radius);
    weight[radius][radius-ii] = weight[radius-ii][radius] = -1./(2*ii*radius);
  }
#else
  const int stencil_size = (2*radius+1)*(2*radius+1);
  for (auto jj=1; jj<=radius; jj++) {
    for (auto ii=-jj+1; ii<jj; ii++) {
      weight[radius+ii][radius+jj] = +1./(4*jj*(2*jj-1)*radius);
      weight[radius+ii][radius-jj] = -1./(4*jj*(2*jj-1)*radius);
      weight[radius+jj][radius+ii] = +1./(4*jj*(2*jj-1)*radius);
      weight[radius-jj][radius+ii] = -1./(4*jj*(2*jj-1)*radius);
    }
    weight[radius+jj][radius+jj]   = +1./(4*jj*radius);
    weight[radius-jj][radius-jj]   = -1./(4*jj*radius);
  }
#endif

  double * RESTRICT in  = new double[n*n];
  double * RESTRICT out = new double[n*n];

  auto stencil_time = 0.0;

  // HOST
  // initialize the input and output arrays
  _Pragma("omp parallel")
  {
    _Pragma("omp for")
    for (auto i=0; i<n; i++) {
      for (auto j=0; j<n; j++) {
        in[i*n+j] = static_cast<double>(i+j);
        out[i*n+j] = 0.0;
      }
    }
  }

  // DEVICE
  _Pragma("omp target map(tofrom: in[0:n*n], out[0:n*n]) map(to:weight[0:2*radius+1][0:2*radius+1]) map(from:stencil_time)")
  _Pragma("omp parallel")
  {
    for (auto iter = 0; iter<=iterations; iter++) {

      if (iter==1) {
          _Pragma("omp barrier")
          _Pragma("omp master")
          stencil_time = prk::wtime();
      }

      // Apply the stencil operator
#ifdef STAR
      do_stencil<RADIUS,true>(n, weight, in, out);
#else
      do_stencil<RADIUS,false>(n, weight, in, out);
#endif

      // add constant to solution to force refresh of neighbor data, if any
      _Pragma("omp for")
      for (auto i=0; i<n; i++) {
        for (auto j=0; j<n; j++) {
          in[i*n+j] += 1.0;
        }
      }
    }
    {
        _Pragma("omp barrier")
        _Pragma("omp master")
        stencil_time = prk::wtime() - stencil_time;
    }
  }

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  // interior of grid with respect to stencil
  size_t active_points = static_cast<size_t>(n-2*radius)*static_cast<size_t>(n-2*radius);

  // HOST
  // compute L1 norm in parallel
  double norm = 0.0;
  _Pragma("omp parallel for reduction(+:norm)")
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
    size_t flops = (2L*(size_t)stencil_size+1L) * active_points;
    auto avgtime = stencil_time/iterations;
    std::cout << "Rate (MFlops/s): " << 1.0e-6 * static_cast<double>(flops)/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  }

  return 0;
}
