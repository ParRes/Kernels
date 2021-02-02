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
/// NAME:    transpose
///
/// PURPOSE: This program measures the time for the transpose of a
///          column-major stored matrix into a row-major stored matrix.
///
/// USAGE:   Program input is the matrix order and the number of times to
///          repeat the operation:
///
///          transpose <matrix_size> <# iterations> [tile size]
///
///          An optional parameter specifies the tile size used to divide the
///          individual matrix blocks for improved cache and TLB performance.
///
///          The output consists of diagnostics to make sure the
///          transpose worked and timing statistics.
///
/// HISTORY: Written by  Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, February 2016 and May 2017.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_openmp.h"

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/OpenMP TASKLOOP Matrix transpose: B = A^T" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations, gs;
  int order;
  int tile_size;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <matrix order> [taskloop grainsize] [tile size]";
      }

      // number of times to do the transpose
      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      // order of a the matrix
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

      // taskloop grainsize
      gs = (argc > 4) ? std::atoi(argv[4]) : 32;
      if (gs < 1 || gs > order) {
        throw "ERROR: grainsize";
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
  std::cout << "Matrix order         = " << order << std::endl;
  std::cout << "Tile size            = " << tile_size << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  prk::vector<double> A(order*order);
  prk::vector<double> B(order*order);

  double trans_time{0};

  OMP_PARALLEL()
  OMP_MASTER
  {
    OMP_TASKLOOP( firstprivate(order) shared(A,B) grainsize(gs) )
    for (int i=0;i<order; i++) {
      for (int j=0;j<order;j++) {
        A[i*order+j] = static_cast<double>(i*order+j);
        B[i*order+j] = 0.0;
      }
    }
    OMP_TASKWAIT

    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) trans_time = prk::wtime();

      // transpose the  matrix
      if (tile_size < order) {
        OMP_TASKLOOP_COLLAPSE(2, firstprivate(order) shared(A,B) grainsize(gs) )
        for (int it=0; it<order; it+=tile_size) {
          for (int jt=0; jt<order; jt+=tile_size) {
            for (int i=it; i<std::min(order,it+tile_size); i++) {
              for (int j=jt; j<std::min(order,jt+tile_size); j++) {
                B[i*order+j] += A[j*order+i];
                A[j*order+i] += 1.0;
              }
            }
          }
        }
      } else {
        OMP_TASKLOOP( firstprivate(order) shared(A,B) grainsize(gs) )
        for (int i=0;i<order; i++) {
          for (int j=0;j<order;j++) {
            B[i*order+j] += A[j*order+i];
            A[j*order+i] += 1.0;
          }
        }
      }
      OMP_TASKWAIT
    }
    trans_time = prk::wtime() - trans_time;
  }

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const auto addit = (iterations+1.) * (iterations/2.);
  auto abserr = 0.0;
  OMP_PARALLEL_FOR_REDUCE( +:abserr )
  for (int j=0; j<order; j++) {
    for (int i=0; i<order; i++) {
      const size_t ij = i*order+j;
      const size_t ji = j*order+i;
      const double reference = static_cast<double>(ij)*(1.+iterations)+addit;
      abserr += prk::abs(B[ji] - reference);
    }
  }

#ifdef VERBOSE
  std::cout << "Sum of absolute differences: " << abserr << std::endl;
#endif

  const auto epsilon = 1.0e-8;
  if (abserr < epsilon) {
    std::cout << "Solution validates" << std::endl;
    auto avgtime = trans_time/iterations;
    auto bytes = (size_t)order * (size_t)order * sizeof(double);
    std::cout << "Rate (MB/s): " << 1.0e-6 * (2L*bytes)/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
    std::cout << "ERROR: Aggregate squared error " << abserr
              << " exceeds threshold " << epsilon << std::endl;
    return 1;
  }

  return 0;
}


