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
///          cblas_dgemm()
///          cblas_dgemm_batch()
///
/// HISTORY: Written by Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, December, 2017.
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

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef PRK_DEBUG
#include <random>
void prk_dgemm_loops(const int order,
               const std::vector<double> & A,
               const std::vector<double> & B,
                     std::vector<double> & C)
{
    for (int i=0; i<order; ++i) {
      for (int j=0; j<order; ++j) {
        for (int k=0; k<order; ++k) {
            C[i*order+j] += A[i*order+k] * B[k*order+j];
        }
      }
    }
}
#endif

void prk_dgemm(const int order,
               const std::vector<double> & A,
               const std::vector<double> & B,
                     std::vector<double> & C)
{
    const int n = order;
    const double alpha = 1.0;
    const double beta  = 1.0;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, alpha, A.data(), n, B.data(), n, beta, C.data(), n);
}

void prk_dgemm(const int order, const int batches,
               const std::vector<std::vector<double>> & A,
               const std::vector<std::vector<double>> & B,
                     std::vector<std::vector<double>> & C)
{
    const int n = order;
    const double alpha = 1.0;
    const double beta  = 1.0;

    for (int b=0; b<batches; ++b) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, alpha, &(A[b][0]), n, &(B[b][0]), n, beta, &(C[b][0]), n);
    }
}

void prk_dgemm(const int order, const int batches, const int nt,
               const std::vector<std::vector<double>> & A,
               const std::vector<std::vector<double>> & B,
                     std::vector<std::vector<double>> & C)
{
    const int n = order;
    const double alpha = 1.0;
    const double beta  = 1.0;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(nt)
#endif
    for (int b=0; b<batches; ++b) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, alpha, A[b].data(), n, B[b].data(), n, beta, C[b].data(), n);
    }
}

void prk_dgemm(const int order, const int batches,
               double** & A,
               double** & B,
               double** & C)
{
    const int n = order;
    const double alpha = 1.0;
    const double beta  = 1.0;

    const int group_count = 1;
    PRK_UNUSED const int group_size[group_count] = { batches };

    const CBLAS_TRANSPOSE transa_array[group_count] = { CblasNoTrans };
    const CBLAS_TRANSPOSE transb_array[group_count] = { CblasNoTrans };

    const int n_array[group_count] = { n };

    const double alpha_array[group_count] = { alpha };
    const double beta_array[group_count]  = { beta };

#ifdef MKL
    cblas_dgemm_batch(CblasRowMajor, transa_array, transb_array,
                      n_array, n_array, n_array,
                      alpha_array,
                      (const double**) A, n_array,
                      (const double**) B, n_array,
                      beta_array,
                      C, n_array,
                      group_count, group_size);
#else // e.g. Accelerate does not have batched BLAS
    for (int b=0; b<batches; ++b) {
        cblas_dgemm(CblasRowMajor,
                    transa_array[0], transb_array[0],
                    n_array[0], n_array[0], n_array[0],
                    alpha_array[0],
                    A[b], n_array[0],
                    B[b], n_array[0],
                    beta_array[0],
                    C[b], n_array[0]);
    }
#endif
}

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/CBLAS Dense matrix-matrix multiplication: C += A x B" << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int order;
  int batches = 0;
  int batch_threads = 1;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <matrix order> [<batches> <batch threads>]";
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

      if (argc > 3) {
        batches = std::atoi(argv[3]);
      }

      if (argc>4) {
        batch_threads = std::atoi(argv[4]);
      } else {
#ifdef _OPENMP
        batch_threads = omp_get_max_threads();
#endif
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Matrix order         = " << order << std::endl;
  if (batches == 0) {
      std::cout << "No batching" << std::endl;
  } else if (batches > 0) {
#ifdef MKL
      std::cout << "Batch size           = " <<  batches << " (batched BLAS)" << std::endl;
#else
      std::cout << "Batch size           = " << std::abs(batches) << " (loop over legacy BLAS sequentially)" << std::endl;
#endif
  } else if (batches < 0) {
      if (batch_threads > 1) {
          std::cout << "Batch size           = " << std::abs(batches) << " (loop over legacy BLAS with " << batch_threads << " threads)" << std::endl;
      } else {
          std::cout << "Batch size           = " << std::abs(batches) << " (loop over legacy BLAS sequentially)" << std::endl;
      }
  }

  //////////////////////////////////////////////////////////////////////
  // Allocate space for matrices
  //////////////////////////////////////////////////////////////////////

  double gemm_time{0};

  const int matrices = (batches==0 ? 1 : abs(batches));

  std::vector<double> const M(order*order,0);
  std::vector<std::vector<double>> A(matrices,M);
  std::vector<std::vector<double>> B(matrices,M);
  std::vector<std::vector<double>> C(matrices,M);
  for (int b=0; b<matrices; ++b) {
    for (int i=0; i<order; ++i) {
      for (int j=0; j<order; ++j) {
         A[b][i*order+j] = i;
         B[b][i*order+j] = i;
         C[b][i*order+j] = 0;
      }
    }
  }

  double ** pA = new double*[matrices];
  double ** pB = new double*[matrices];
  double ** pC = new double*[matrices];

  for (int b=0; b<matrices; ++b) {
     pA[b] = A[b].data();
     pB[b] = B[b].data();
     pC[b] = C[b].data();
  }

  {
    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) gemm_time = prk::wtime();

      if (batches == 0) {
          prk_dgemm(order, A[0], B[0], C[0]);
      } else if (batches < 0) {
          prk_dgemm(order, matrices, batch_threads, A, B, C);
      } else if (batches > 0) {
          prk_dgemm(order, matrices, pA, pB, pC);
      }
    }
    gemm_time = prk::wtime() - gemm_time;
  }

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double epsilon = 1.0e-8;
  const double forder = static_cast<double>(order);
  const double reference = 0.25 * prk::pow(forder,3) * prk::pow(forder-1.0,2) * (iterations+1);
  double residuum(0);
  for (int b=0; b<matrices; ++b) {
      const auto checksum = prk::reduce(C[b].begin(), C[b].end(), 0.0);
      residuum += std::abs(checksum - reference) / reference;
  }
  residuum /= matrices;

  if (residuum < epsilon) {
#if VERBOSE
    std::cout << "Reference checksum = " << reference << "\n"
              << "Residuum           = " << residuum << std::endl;
#endif
    std::cout << "Solution validates" << std::endl;
    auto avgtime = gemm_time/iterations/matrices;
    auto nflops = 2.0 * prk::pow(forder,3);
    prk::print_flop_rate_time("FP64", nflops/avgtime, avgtime);
  } else {
    std::cout << "Reference checksum = " << reference << "\n"
              << "Residuum           = " << residuum << std::endl;
#if VERBOSE
    std::cout << "i, j, A, B, C" << std::endl;
    for (int i=0; i<order; ++i)
      for (int j=0; j<order; ++j)
        std::cout << i << "," << j << " = " << A[0][i*order+j] << ", " << B[0][i*order+j] << ", " << C[0][i*order+j] << "\n";
    std::cout << std::endl;
#endif
    return 1;
  }

  return 0;
}


