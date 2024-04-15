///
/// Copyright (c) 2020, Intel Corporation
/// Copyright (c) 2023, NVIDIA
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
/// NAME:    gemm
///
/// PURPOSE: This program tests the efficiency with which a dense matrix
///          dense multiplication is carried out
///
/// USAGE:   The program takes as input the matrix order,
///          the number of times the matrix-matrix multiplication
///          is carried out, and, optionally, a tile size for matrix
///          blocking
///
///          <progname> <# iterations> <matrix order>
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

#include "prk_util.h"

#if defined(MKL)
    #include <mkl_cblas.h>
    #define PRK_INT  MKL_INT
    #define PRK_F16  MKL_F16
    #define USE_F16 1
    #define PRK_BF16 MKL_BF16
    #define USE_BF16 1
#elif defined(ACCELERATE)
    // The location of cblas.h is not in the system include path when -framework Accelerate is provided.
    #include <Accelerate/Accelerate.h>
    #define PRK_INT int
    #define CBLAS_LAYOUT CBLAS_ORDER
#else
    // assume OpenBLAS for now
    #include <cblas.h>
    #ifdef OPENBLAS_USE64BITINT
        #define PRK_INT long
    #else
        #define PRK_INT int
    #endif
    #define PRK_BF16 bfloat16
#endif

template <typename TAB, typename TC>
void prk_gemm(const CBLAS_LAYOUT Layout,
              const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
              const PRK_INT M, const PRK_INT N, const PRK_INT K,
              const TC alpha,
              const TAB * A, const PRK_INT lda,
              const TAB * B, const PRK_INT ldb,
              const TC beta,
              TC * C, const PRK_INT ldc)
{
    std::cerr << "No valid template match for type T" << std::endl;
    std::abort();
}

#ifdef PRK_F16
template <>
void prk_gemm(const CBLAS_LAYOUT Layout,
              const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
              const PRK_INT M, const PRK_INT N, const PRK_INT K,
              const PRK_F16 alpha,
              const PRK_F16 * A, const PRK_INT lda,
              const PRK_F16 * B, const PRK_INT ldb,
              const PRK_F16 beta,
              PRK_F16 * C, const PRK_INT ldc)
{
    cblas_hgemm(Layout, TransA, TransB,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); 
}
#endif

#ifdef USE_BF16
template <>
void prk_gemm(const CBLAS_LAYOUT Layout,
              const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
              const PRK_INT M, const PRK_INT N, const PRK_INT K,
              const float alpha,
              const PRK_BF16 * A, const PRK_INT lda,
              const PRK_BF16 * B, const PRK_INT ldb,
              const float beta,
              float * C, const PRK_INT ldc)
{
#ifdef MKL
    // MKL
    // cblas_gemm_bf16bf16f32(const CBLAS_LAYOUT Layout,
    //                        const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    //                        const PRK_INT M, const PRK_INT N, const PRK_INT K,
    //                        const float alpha, const PRK_BF16 *A, const PRK_INT lda,
    //                                           const PRK_BF16 *B, const PRK_INT ldb,
    //                        const float beta, float *C, const PRK_INT ldc);
    cblas_gemm_bf16bf16f32(Layout, TransA, TransB,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); 
#else
    // OpenBLAS
    // cblas_sbgemm(OPENBLAS_CONST enum CBLAS_ORDER Order,
    //              OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
    //              OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
    //              OPENBLAS_CONST float alpha, OPENBLAS_CONST bfloat16 *A, OPENBLAS_CONST blasint lda,
    //                                          OPENBLAS_CONST bfloat16 *B, OPENBLAS_CONST blasint ldb,
    //              OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
    cblas_sbgemm(Layout, TransA, TransB,
                 M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}
#endif

template <>
void prk_gemm(const CBLAS_LAYOUT Layout,
              const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
              const PRK_INT M, const PRK_INT N, const PRK_INT K,
              const float alpha,
              const float * A, const PRK_INT lda,
              const float * B, const PRK_INT ldb,
              const float beta,
              float * C, const PRK_INT ldc)
{
    cblas_sgemm(Layout, TransA, TransB,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); 
}

template <>
void prk_gemm(const CBLAS_LAYOUT Layout,
              const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
              const PRK_INT M, const PRK_INT N, const PRK_INT K,
              const double alpha,
              const double * A, const PRK_INT lda,
              const double * B, const PRK_INT ldb,
              const double beta,
              double * C, const PRK_INT ldc)
{
    cblas_dgemm(Layout, TransA, TransB,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); 
}

#ifdef USE_BF16
void run_BF16(int iterations, int order)
{
  double gemm_time{0};

  const size_t nelems = (size_t)order * (size_t)order;

  auto A = new PRK_BF16[nelems];
  auto B = new PRK_BF16[nelems];
  auto C = new float[nelems];

  for (int i=0; i<order; ++i) {
    for (int j=0; j<order; ++j) {
       A[i*order+j] = i;
       B[i*order+j] = i;
       C[i*order+j] = 0;
    }
  }

  {
    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) gemm_time = prk::wtime();

      const float alpha{1};
      const float beta{1};

      prk_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
               order, order, order,      // m, n, k
               alpha,                    // alpha
               A, order,                 // A, lda
               B, order,                 // B, ldb
               beta,                     // beta
               C, order);                // C, ldc
    }
    gemm_time = prk::wtime() - gemm_time;
  }

  delete[] A;
  delete[] B;

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double forder = static_cast<double>(order);
  const double reference = 0.25 * prk::pow(forder,3) * prk::pow(forder-1.0,2) * (iterations+1);
  double checksum{0};
  for (int i=0; i<nelems; ++i) {
      checksum += C[i];
  }
  const double residuum = std::abs(checksum - reference) / reference;
  const double epsilon{1.0e-8};
  if ((residuum < epsilon) || (sizeof(PRK_BF16) < 4)) {
#if VERBOSE
    std::cout << "Reference checksum = " << reference << "\n"
              << "Actual checksum = " << checksum << std::endl;
#endif
    std::cout << "Solution validates" << std::endl;
    auto avgtime = gemm_time/iterations;
    auto nflops = 2.0 * prk::pow(forder,3);
    auto pname = "BF16";
    prk::print_flop_rate_time(pname, nflops/avgtime, avgtime);
  } else {
    std::cout << "Reference checksum = " << reference << "\n"
              << "Residuum           = " << residuum << std::endl;
  }

  delete[] C;
}
#endif

template <typename T>
void run(int iterations, int order)
{
  auto is_fp64 = (typeid(T) == typeid(double));
  auto is_fp32 = (typeid(T) == typeid(float));
  auto is_fp16 =
#ifdef USE_F16
          (typeid(T) == typeid(PRK_F16));
#else
          false;
#endif
  auto is_bf16 =
#ifdef USE_BF16
          (typeid(T) == typeid(PRK_BF16));
#else
          false;
#endif

  double gemm_time{0};

  const size_t nelems = (size_t)order * (size_t)order;

  auto A = new T[nelems];
  auto B = new T[nelems];
  auto C = new T[nelems];

  for (int i=0; i<order; ++i) {
    for (int j=0; j<order; ++j) {
       A[i*order+j] = i;
       B[i*order+j] = i;
       C[i*order+j] = 0;
    }
  }

  {
    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) gemm_time = prk::wtime();

      const T alpha{1};
      const T beta{1};

      prk_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
               order, order, order,      // m, n, k
               alpha,                    // alpha
               A, order,                 // A, lda
               B, order,                 // B, ldb
               beta,                     // beta
               C, order);                // C, ldc
    }
    gemm_time = prk::wtime() - gemm_time;
  }
  // copy output back to host

  delete[] A;
  delete[] B;

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double forder = static_cast<double>(order);
  const double reference = 0.25 * prk::pow(forder,3) * prk::pow(forder-1.0,2) * (iterations+1);
  double checksum{0};
  for (int i=0; i<nelems; ++i) {
      checksum += C[i];
  }
  const double residuum = std::abs(checksum - reference) / reference;
  const double epsilon{1.0e-8};
  if ((residuum < epsilon) || (sizeof(T) < 4)) {
#if VERBOSE
    std::cout << "Reference checksum = " << reference << "\n"
              << "Actual checksum = " << checksum << std::endl;
#endif
    std::cout << "Solution validates" << std::endl;
    auto avgtime = gemm_time/iterations;
    auto nflops = 2.0 * prk::pow(forder,3);
    auto pname = (is_fp64 ? "FP64" :
                  (is_fp32 ? "FP32" :
                   (is_fp16 ? "FP16" :
                    (is_bf16 ? "BF16" : "Unknown FP type"))));
    prk::print_flop_rate_time(pname, nflops/avgtime, avgtime);
  } else {
    std::cout << "Reference checksum = " << reference << "\n"
              << "Residuum           = " << residuum << std::endl;
  }

  delete[] C;
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
  try {
      if (argc < 2) {
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

#ifdef USE_F16
  run<PRK_F16>(iterations, order);
#endif
#ifdef USE_BF16
  run_BF16(iterations, order);
#endif
  run<float>(iterations, order);
  run<double>(iterations, order);

  return 0;
}


