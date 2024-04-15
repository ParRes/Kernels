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

#if defined(MKL)
#include <mkl.h>
#ifdef MKL_ILP64
#error Use the MKL library for 32-bit integers!
#endif
#elif defined(ACCELERATE)
// The location of cblas.h is not in the system include path when -framework Accelerate is provided.
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/CBLAS Matrix transpose: B = A^T" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int order;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <matrix order>";
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
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Matrix order         = " << order << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double trans_time{0};

  prk::vector<double> A(order*order);
  prk::vector<double> B(order*order,0.0);
  prk::vector<double> T(order*order);
  double one[1] = {1.0};

  // fill A with the sequence 0 to order^2-1 as doubles
  std::iota(A.begin(), A.end(), 0.0);

  {
    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) trans_time = prk::wtime();

#if defined(ACCELERATE) && defined(ACCELERATE_NEW_LAPACK)
      // B += transpose(A)
      appleblas_dgeadd(CblasRowMajor,
                       CblasTrans, CblasNoTrans,   // opA, opB
                       order, order,               // m, n
                       1.0, &(A[0]), order,        // alpha, A, lda
                       1.0, &(B[0]), order,        // beta, B, ldb
                       &(B[0]), order);            // C, ldc (in-place for B)
#else
      // T = transpose(A)
   #if defined(MKL)
      mkl_domatcopy('R','T', order, order, 1.0, &(A[0]), order, &(T[0]), order);
   #elif defined(OPENBLAS_VERSION)
      cblas_domatcopy(CblasRowMajor,CblasTrans, order, order, 1.0, &(A[0]), order, &(T[0]), order);
   #elif defined(ACCELERATE)
      vDSP_mtransD(&(A[0]), 1, &(T[0]), 1, order, order);
   #else
      #warning No CBLAS transpose extension available!
      for (int i=0;i<order; i++) {
        for (int j=0;j<order;j++) {
          T[i*order+j] = A[j*order+i];
        }
      }
   #endif
      // B += T
      cblas_daxpy(order*order, 1.0, &(T[0]), 1, &(B[0]), 1);
#endif
      // A += 1
      cblas_daxpy(order*order, 1.0, one, 0, &(A[0]), 1);
    }
    trans_time = prk::wtime() - trans_time;
  }

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const auto addit = (iterations+1.) * (iterations/2.);
  double abserr(0);
  // TODO: replace with std::generate, std::accumulate, or similar
  for (int j=0; j<order; j++) {
    for (int i=0; i<order; i++) {
      const int ij = i*order+j;
      const int ji = j*order+i;
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


