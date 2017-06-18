
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

std::vector<std::vector<double>> initialize_w(bool star, int radius)
{
  std::vector<std::vector<double>> weight;
  weight.resize(2*radius+1);
  for (auto i=0; i<2*radius+1; i++) {
    weight[i].resize(2*radius+1, 0.0);
  }

  // fill the stencil weights to reflect a discrete divergence operator
  if (star) {
    for (auto ii=1; ii<=radius; ii++) {
      weight[radius][radius+ii] = weight[radius+ii][radius] = +1./(2*ii*radius);
      weight[radius][radius-ii] = weight[radius-ii][radius] = -1./(2*ii*radius);
    }
  } else {
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
  }
  return weight;
}

struct Initialize
{
    public:
        void operator()( const tbb::blocked_range2d<int>& r ) const {
            for (tbb::blocked_range<int>::const_iterator i=r.rows().begin(); i!=r.rows().end(); ++i ) {
                for (tbb::blocked_range<int>::const_iterator j=r.cols().begin(); j!=r.cols().end(); ++j ) {
                    A_[i*n_+j] = static_cast<double>(i+j);
                    B_[i*n_+j] = 0.0;
                }
            }
        }

        Initialize(int n, std::vector<double> & A, std::vector<double> & B) : n_(n), A_(A), B_(B) { }

    private:
        int n_;
        std::vector<double> & A_;
        std::vector<double> & B_;

};

struct Add
{
    public:
        void operator()( const tbb::blocked_range2d<int>& r ) const {
            for (tbb::blocked_range<int>::const_iterator i=r.rows().begin(); i!=r.rows().end(); ++i ) {
                for (tbb::blocked_range<int>::const_iterator j=r.cols().begin(); j!=r.cols().end(); ++j ) {
                    A_[i*n_+j] += 1.0;
                }
            }
        }

        Add(int n, std::vector<double> & A) : n_(n), A_(A) { }

    private:
        int n_;
        std::vector<double> & A_;

};

template <bool s_, int r_>
struct Stencil
{
    public:
        void operator()( const tbb::blocked_range2d<int>& r ) const {
            for (tbb::blocked_range<int>::const_iterator i=r.rows().begin(); i!=r.rows().end(); ++i ) {
                for (tbb::blocked_range<int>::const_iterator j=r.cols().begin(); j!=r.cols().end(); ++j ) {
                    if (s_) {
                        for (auto jj=-r_; jj<=r_; jj++) {
                            B_[i*n_+j] += w_[r_][r_+jj]*A_[i*n_+j+jj];
                        }
                        for (auto ii=-r_; ii<0; ii++) {
                            B_[i*n_+j] += w_[r_+ii][r_]*A_[(i+ii)*n_+j];
                        }
                        for (auto ii=1; ii<=r_; ii++) {
                            B_[i*n_+j] += w_[r_+ii][r_]*A_[(i+ii)*n_+j];
                        }
                    } else {
                        for (auto ii=-r_; ii<=r_; ii++) {
                            for (auto jj=-r_; jj<=r_; jj++) {
                                B_[i*n_+j] += w_[r_+ii][r_+jj]*A_[(i+ii)*n_+j+jj];
                            }
                        }
                    }
                }
            }
        }

        Stencil(int n, std::vector<std::vector<double>> & w,
                std::vector<double> & A, std::vector<double> & B)
              : n_(n), w_(w), A_(A), B_(B) { }

    private:
        int n_;
        std::vector<std::vector<double>> & w_;
        std::vector<double> & A_;
        std::vector<double> & B_;

};

void ParallelInitialize(int n, int tile_size, std::vector<double> & A, std::vector<double> & B)
{
    Initialize i(n, A, B);
    const tbb::blocked_range2d<int> r(0, n, tile_size, 0, n, tile_size);
    parallel_for(r,i);
}

void ParallelAdd(int n, int tile_size, std::vector<double> & A)
{
    Add a(n, A);
    const tbb::blocked_range2d<int> r(0, n, tile_size, 0, n, tile_size);
    parallel_for(r,a);
}

template <bool star, int radius>
void ParallelStencil(int n, int tile_size, std::vector<std::vector<double>> & weights,
                     std::vector<double> & A, std::vector<double> & B)
{
    Stencil<star, radius> s(n, weights, A, B);
    const tbb::blocked_range2d<int> r(radius, n-radius, tile_size, radius, n-radius, tile_size);
    parallel_for(r,s);
}

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/TBB Stencil execution on 2D grid" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // process and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int n, radius, tile_size;
  bool star = true;
  try {
      if (argc < 3){
        throw "Usage: <# iterations> <array dimension> [tile_size] [<star/grid> <radius>]";
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

      // linear grid dimension
      tile_size = 32;
      if (argc > 3) {
        tile_size  = std::atoi(argv[3]);
        if (tile_size < 1 || tile_size > n) tile_size = n;
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
  std::cout << "Compact representation of stencil loop body" << std::endl;

  tbb::task_scheduler_init init(tbb::task_scheduler_init::automatic);

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  std::vector<std::vector<double>> weight = initialize_w(star, radius);

  std::vector<double> A;
  std::vector<double> B;
  A.resize(n*n);
  B.resize(n*n);

  auto stencil_time = 0.0;

  ParallelInitialize(n, tile_size, A, B);

  for (auto iter = 0; iter<=iterations; iter++) {

    if (iter==1) stencil_time = prk::wtime();

    // Apply the stencil operator
    if (star) {
        switch (radius) {
            case 1: ParallelStencil<1,true>(n, tile_size, weight, A, B); break;
            case 2: ParallelStencil<2,true>(n, tile_size, weight, A, B); break;
            case 3: ParallelStencil<3,true>(n, tile_size, weight, A, B); break;
            case 4: ParallelStencil<4,true>(n, tile_size, weight, A, B); break;
            case 5: ParallelStencil<5,true>(n, tile_size, weight, A, B); break;
            case 6: ParallelStencil<6,true>(n, tile_size, weight, A, B); break;
            case 7: ParallelStencil<7,true>(n, tile_size, weight, A, B); break;
            case 8: ParallelStencil<8,true>(n, tile_size, weight, A, B); break;
            case 9: ParallelStencil<9,true>(n, tile_size, weight, A, B); break;
            default: { std::cerr << "Template not instantiated for radius " << radius << "\n"; break; }
        }
    } else {
        switch (radius) {
            case 1: ParallelStencil<1,false>(n, tile_size, weight, A, B); break;
            case 2: ParallelStencil<2,false>(n, tile_size, weight, A, B); break;
            case 3: ParallelStencil<3,false>(n, tile_size, weight, A, B); break;
            case 4: ParallelStencil<4,false>(n, tile_size, weight, A, B); break;
            case 5: ParallelStencil<5,false>(n, tile_size, weight, A, B); break;
            case 6: ParallelStencil<6,false>(n, tile_size, weight, A, B); break;
            case 7: ParallelStencil<7,false>(n, tile_size, weight, A, B); break;
            case 8: ParallelStencil<8,false>(n, tile_size, weight, A, B); break;
            case 9: ParallelStencil<9,false>(n, tile_size, weight, A, B); break;
            default: { std::cerr << "Template not instantiated for radius " << radius << "\n"; break; }
        }
    }
    ParallelAdd(n, tile_size, A);
  }
  stencil_time = prk::wtime() - stencil_time;

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  // interior of grid with respect to stencil
  size_t active_points = static_cast<size_t>(n-2*radius)*static_cast<size_t>(n-2*radius);

  // compute L1 norm A parallel
  double norm = 0.0;
  for (auto i=radius; i<n-radius; i++) {
    for (auto j=radius; j<n-radius; j++) {
      norm += std::fabs(B[i*n+j]);
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
