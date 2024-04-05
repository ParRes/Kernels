///
/// Copyright (c) 2018, Intel Corporation
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
/// NAME:    transpose
///
/// PURPOSE: This program measures the time for the transpose of a
///          column-major stored matrix into a row-major stored matrix.
///
/// USAGE:   Program input is the matrix order and the number of times to
///          repeat the operation:
///
///          transpose <matrix_size> <# iterations>
///
///          The output consists of diagnostics to make sure the
///          transpose worked and timing statistics.
///
/// HISTORY: Written by  Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, February 2016 and May 2017.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_kokkos.h"

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/Kokkos Matrix transpose: B = A^T" << std::endl;

  Kokkos::initialize(argc, argv);
  {
    // row-major 2D array
    typedef Kokkos::View<double**, Kokkos::LayoutRight> matrix;
    // column-major 2D array
    //typedef Kokkos::View<double**, Kokkos::LayoutLeft> matrix;
    // default 2D array
    //typedef Kokkos::View<double**> matrix;

    //////////////////////////////////////////////////////////////////////
    /// Read and test input parameters
    //////////////////////////////////////////////////////////////////////

    int iterations;
    int order;
    int tile_size;
    bool permute = false;
    try {
        if (argc < 3) {
          throw "Usage: <# iterations> <matrix order> [<tile_size> <permute=0/1>]";
        }

        iterations  = std::atoi(argv[1]);
        if (iterations < 1) {
          throw "ERROR: iterations must be >= 1";
        }

        order = std::atoi(argv[2]);
        if (order <= 0) {
          throw "ERROR: Matrix Order must be greater than 0";
        } else if (order > prk::get_max_matrix_size()) {
          throw "ERROR: matrix dimension too large - overflow risk";
        }

        // default tile size for tiling of local transpose
        tile_size = (argc>3) ? std::atoi(argv[3]) : 32;
        // a negative tile size means no tiling of the local transpose
        if (tile_size <= 0) tile_size = order;

        auto permute_input = (argc>4) ? std::atoi(argv[4]) : 0;
        if (permute_input != 0 && permute_input != 1) {
          throw "ERROR: permute must be 0 (no) or 1 (yes)";
        }
        permute = (permute_input == 1);
    }
    catch (const char * e) {
      std::cout << e << std::endl;
      return 1;
    }

    std::cout << "Number of iterations = " << iterations << std::endl;
    std::cout << "Matrix order         = " << order << std::endl;
    std::cout << "Tile size            = " << tile_size << std::endl;
    std::cout << "Permute loops        = " << (permute ? "yes" : "no") << std::endl;
    std::cout << "Kokkos execution space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;

    //////////////////////////////////////////////////////////////////////
    // Allocate space and perform the computation
    //////////////////////////////////////////////////////////////////////

    double trans_time{0};

    matrix A("A", order, order);
    matrix B("B", order, order);

    auto order2 = {order,order};
    auto tile2  = {tile_size,tile_size};

    const auto policy    = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {order,order}, {tile_size,tile_size});
    typedef Kokkos::Rank<2,Kokkos::Iterate::Right,Kokkos::Iterate::Left > rl;
    typedef Kokkos::Rank<2,Kokkos::Iterate::Left, Kokkos::Iterate::Right> lr;
    const auto policy_lr = Kokkos::MDRangePolicy<rl>({0,0}, {order,order}, {tile_size,tile_size});
    const auto policy_rl = Kokkos::MDRangePolicy<lr>({0,0}, {order,order}, {tile_size,tile_size});

    {
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(int i, int j) {
          A(i,j) = static_cast<double>(i*order+j);
          B(i,j) = 0.0;
      });
      Kokkos::fence();

      for (int iter = 0; iter<=iterations; ++iter) {

        if (iter==1) {
          Kokkos::fence();
          trans_time = prk::wtime();
        }

        if (permute) {
            Kokkos::parallel_for(policy_rl, KOKKOS_LAMBDA(int i, int j) {
                B(i,j) += A(j,i);
                A(j,i) += 1.0;
            });
        } else {
            Kokkos::parallel_for(policy_lr, KOKKOS_LAMBDA(int i, int j) {
                B(i,j) += A(j,i);
                A(j,i) += 1.0;
            });
        }
      }
      Kokkos::fence();
      trans_time = prk::wtime() - trans_time;
    }

    //////////////////////////////////////////////////////////////////////
    /// Analyze and output results
    //////////////////////////////////////////////////////////////////////

    double const addit = (iterations+1.) * (0.5*iterations);
    double abserr(0);
    Kokkos::parallel_reduce(policy, KOKKOS_LAMBDA(int i, int j, double & update) {
        size_t const ij = i*order+j;
        double const reference = static_cast<double>(ij)*(1.+iterations)+addit;
        using Kokkos::fabs;
        update += fabs(B(j,i) - reference);
    }, abserr);

#ifdef VERBOSE
    std::cout << "Sum of absolute differences: " << abserr << std::endl;
#endif


    double epsilon(1.0e-8);
    if (abserr < epsilon) {
      std::cout << "Solution validates" << std::endl;
      auto avgtime = trans_time/iterations;
      auto bytes = (size_t)order * (size_t)order * sizeof(double);
      std::cout << "Rate (MB/s): " << 1.0e-6 * (2.*bytes)/avgtime
                << " Avg time (s): " << avgtime << std::endl;
    } else {
      std::cout << "ERROR: Aggregate squared error " << abserr
                << " exceeds threshold " << epsilon << std::endl;
      return 1;
    }

  }
  Kokkos::finalize();
  return 0;
}


