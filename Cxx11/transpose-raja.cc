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
#include "prk_raja.h"

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/RAJA Matrix transpose: B = A^T" << std::endl;

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

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double trans_time{0};

  double * RESTRICT Amem = new double[order*order];
  double * RESTRICT Bmem = new double[order*order];

  matrix A(Amem, order, order);
  matrix B(Bmem, order, order);

  using regular_policy = RAJA::KernelPolicy< RAJA::statement::For<0, thread_exec,
                                             RAJA::statement::For<1, RAJA::simd_exec,
                                             RAJA::statement::Lambda<0> > > >;
  using permute_policy = RAJA::KernelPolicy< RAJA::statement::For<1, thread_exec,
                                             RAJA::statement::For<0, RAJA::simd_exec,
                                             RAJA::statement::Lambda<0> > > >;

  RAJA::RangeSegment range(0, order);
  auto range2d = RAJA::make_tuple(range, range);

  RAJA::kernel<regular_policy>(range2d, [=](int i, int j) {
      A(i,j) = static_cast<double>(i*order+j);
      B(i,j) = 0.0;
  });

  for (int iter = 0; iter<=iterations; ++iter) {

    if (iter==1) trans_time = prk::wtime();

    if (permute) {
        RAJA::kernel<permute_policy>(range2d, [=](int i, int j) {
            B(i,j) += A(j,i);
            A(j,i) += 1.0;
        });
    } else {
        RAJA::kernel<regular_policy>(range2d, [=](int i, int j) {
            B(i,j) += A(j,i);
            A(j,i) += 1.0;
        });
    }
  }
  trans_time = prk::wtime() - trans_time;

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  using reduce_policy = RAJA::KernelPolicy< RAJA::statement::For<0, thread_exec,
                                            RAJA::statement::For<1, RAJA::seq_exec,
                                            RAJA::statement::Lambda<0> > > >;

  double const addit = (iterations+1.) * (0.5*iterations);
  RAJA::ReduceSum<reduce_exec, double> abserr(0.0);
  RAJA::kernel<reduce_policy>(range2d, [=](int i, int j) {
      double const ij = static_cast<double>(i*order+j);
      double const reference = ij*(1.+iterations)+addit;
      abserr += prk::abs(B(j,i) - reference);
  });

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
  return 0;
}


