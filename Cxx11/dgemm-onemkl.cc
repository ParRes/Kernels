///
/// Copyright (c) 2020, Intel Corporation
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
/// NAME:    dgemm
///
/// PURPOSE: This program tests the efficiency with which a dense matrix
///          dense multiplication is carried out
///
/// USAGE:   The program takes as input the matrix order,
///          the number of times the matrix-matrix multiplication
///          is carried out, and, optionally, a tile size for matrix
///          blocking
///
///          <progname> <# iterations> <matrix order> [<batches>]
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// FUNCTIONS CALLED:
///
///          Other than OpenMP or standard C functions, the following
///          functions are used in this program:
///
/// HISTORY: Written by Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, December, 2017.
///
//////////////////////////////////////////////////////////////////////

#include "prk_sycl.h"
#include "prk_util.h"

#if BETA9 // and older
#include <mkl_blas_sycl.hpp>
#else
#include <oneapi/mkl/blas.hpp>
#endif

using namespace oneapi; // oneapi::mkl -> mkl

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/oneMKL Dense matrix-matrix multiplication: C += A x B" << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int order;
  int batches = 0;
  int input_copy = 0;
  try {
      if (argc < 2) {
        throw "Usage: <# iterations> <matrix order> [<batches>] [<copy input every iteration [0/1]>]";
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

#if 0
      if (argc > 3) {
        batches = std::atoi(argv[3]);
      }

      if (argc > 4) {
        input_copy = std::atoi(argv[4]);
        if (input_copy != 0 && input_copy != 1) {
          throw "ERROR: input_copy was not 0 or 1";
        }
      }
#endif
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Matrix order         = " << order << std::endl;
  if (batches == 0) {
      std::cout << "No batching" << std::endl;
  } else if (batches < 0) {
      std::cout << "Batch size           = " << -batches << " (loop over legacy BLAS)" << std::endl;
  } else if (batches > 0) {
      std::cout << "Batch size           = " <<  batches << " (batched BLAS)" << std::endl;
  }
  std::cout << "Input copy           = " << (input_copy ? "yes" : "no") << std::endl;

  sycl::queue q(sycl::default_selector{});
  prk::SYCL::print_device_platform(q);

  //////////////////////////////////////////////////////////////////////
  // Allocate space for matrices
  //////////////////////////////////////////////////////////////////////

  const int matrices = 1;

  double dgemm_time{0};

  const size_t nelems = (size_t)order * (size_t)order;
  const size_t bytes = nelems * sizeof(double);
  double * h_a = sycl::malloc_host<double>( nelems, q);
  double * h_b = sycl::malloc_host<double>( nelems, q);
  double * h_c = sycl::malloc_host<double>( nelems, q);

  for (int i=0; i<order; ++i) {
    for (int j=0; j<order; ++j) {
       h_a[i*order+j] = i;
       h_b[i*order+j] = i;
       h_c[i*order+j] = 0;
    }
  }

  // copy input from host to device
  double * A = sycl::malloc_device<double>( nelems, q);
  double * B = sycl::malloc_device<double>( nelems, q);
  double * C = sycl::malloc_device<double>( nelems, q);
  q.wait();

  q.memcpy(A, &(h_a[0]), bytes).wait();
  q.memcpy(B, &(h_b[0]), bytes).wait();
  q.memcpy(C, &(h_c[0]), bytes).wait();
  q.wait();

  sycl::free(h_a, q);
  sycl::free(h_b, q);

  {
    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) dgemm_time = prk::wtime();

      const double alpha = 1.0;
      const double beta  = 1.0;

      mkl::blas::gemm(q, mkl::transpose::nontrans, // opA
                         mkl::transpose::nontrans, // opB
                         order, order, order,      // m, n, k
                         alpha,                    // alpha
                         A, order,                 // A, lda
                         B, order,                 // B, ldb
                         beta,                     // beta
                         C, order);                // C, ldc
      q.wait();
    }
    dgemm_time = prk::wtime() - dgemm_time;
  }
  // copy output back to host
  q.memcpy(&(h_c[0]), C, bytes).wait();

  sycl::free(C, q);
  sycl::free(B, q);
  sycl::free(A, q);

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double epsilon = 1.0e-8;
  const double forder = static_cast<double>(order);
  const double reference = 0.25 * prk::pow(forder,3) * prk::pow(forder-1.0,2) * (iterations+1);
  double residuum(0);
  for (int b=0; b<matrices; ++b) {
      const auto checksum = prk::reduce( &(h_c[b*order*order+0]), &(h_c[b*order*order+nelems]), 0.0);
      residuum += std::abs(checksum - reference) / reference;
  }
  residuum /= matrices;

  if (residuum < epsilon) {
#if VERBOSE
    std::cout << "Reference checksum = " << reference << "\n"
              << "Actual checksum = " << checksum << std::endl;
#endif
    std::cout << "Solution validates" << std::endl;
    auto avgtime = dgemm_time/iterations/matrices;
    auto nflops = 2.0 * prk::pow(forder,3);
    std::cout << "Rate (MF/s): " << 1.0e-6 * nflops/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
    std::cout << "Reference checksum = " << reference << "\n"
              << "Residuum           = " << residuum << std::endl;
    return 1;
  }

  sycl::free(h_c, q);

  return 0;
}


