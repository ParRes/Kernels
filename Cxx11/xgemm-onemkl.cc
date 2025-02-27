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

#include "prk_sycl.h"
#include "prk_util.h"

#if BETA9 // and older
#include <mkl_blas_sycl.hpp>
#else
#include <oneapi/mkl/blas.hpp>
#include <oneapi/mkl/bfloat16.hpp>
#endif

using namespace oneapi; // oneapi::mkl -> mkl

template <typename T>
void run(sycl::queue & q, int iterations, int order)
{
  double gemm_time{0};

  const size_t nelems = (size_t)order * (size_t)order;
  const size_t bytes = nelems * sizeof(T);
  auto h_a = sycl::malloc_host<T>( nelems, q);
  auto h_b = sycl::malloc_host<T>( nelems, q);
  auto h_c = sycl::malloc_host<T>( nelems, q);

  for (int i=0; i<order; ++i) {
    for (int j=0; j<order; ++j) {
       h_a[i*order+j] = i;
       h_b[i*order+j] = i;
       h_c[i*order+j] = 0;
    }
  }

  // copy input from host to device
  auto  A = sycl::malloc_device<T>( nelems, q);
  auto  B = sycl::malloc_device<T>( nelems, q);
  auto  C = sycl::malloc_device<T>( nelems, q);
  q.wait();

  q.memcpy(A, &(h_a[0]), bytes).wait();
  q.memcpy(B, &(h_b[0]), bytes).wait();
  q.memcpy(C, &(h_c[0]), bytes).wait();
  q.wait();

  sycl::free(h_a, q);
  sycl::free(h_b, q);

  {
    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) gemm_time = prk::wtime();

      const T alpha{1};
      const T beta{1};

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
    gemm_time = prk::wtime() - gemm_time;
  }
  // copy output back to host
  q.memcpy(&(h_c[0]), C, bytes).wait();

  sycl::free(C, q);
  sycl::free(B, q);
  sycl::free(A, q);

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double forder = static_cast<double>(order);
  const double reference = 0.25 * prk::pow(forder,3) * prk::pow(forder-1.0,2) * (iterations+1);
  double checksum{0};
  for (int i=0; i<nelems; ++i) {
      checksum += h_c[i];
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
    auto is_fp64 = (typeid(T) == typeid(double));
    auto is_fp32 = (typeid(T) == typeid(float));
    auto is_fp16 = (typeid(T) == typeid(sycl::half));
    auto is_bf16 = (typeid(T) == typeid(oneapi::mkl::bfloat16));
    auto pname = (is_fp64 ? "FP64" :
                  (is_fp32 ? "FP32" :
                   (is_fp16 ? "FP16" :
                    (is_bf16 ? "BF16" : "Unknown FP type"))));
    std::cout << pname
              << " Rate (MF/s): " << 1.0e-6 * nflops/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
    std::cout << "Reference checksum = " << reference << "\n"
              << "Residuum           = " << residuum << std::endl;
  }

  sycl::free(h_c, q);
}

template <typename TA, typename TB, typename TC>
void run3(sycl::queue & q, int iterations, int order)
{
  double gemm_time{0};

  const size_t nelems = (size_t)order * (size_t)order;
  auto h_a = sycl::malloc_host<TA>( nelems, q);
  auto h_b = sycl::malloc_host<TB>( nelems, q);
  auto h_c = sycl::malloc_host<TC>( nelems, q);

  for (int i=0; i<order; ++i) {
    for (int j=0; j<order; ++j) {
       h_a[i*order+j] = i;
       h_b[i*order+j] = i;
       h_c[i*order+j] = 0;
    }
  }

  // copy input from host to device
  auto  A = sycl::malloc_device<TA>( nelems, q);
  auto  B = sycl::malloc_device<TB>( nelems, q);
  auto  C = sycl::malloc_device<TC>( nelems, q);
  q.wait();

  q.memcpy(A, &(h_a[0]), nelems * sizeof(TA)).wait();
  q.memcpy(B, &(h_b[0]), nelems * sizeof(TB)).wait();
  q.memcpy(C, &(h_c[0]), nelems * sizeof(TC)).wait();
  q.wait();

  sycl::free(h_a, q);
  sycl::free(h_b, q);

  {
    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) gemm_time = prk::wtime();

      const TA alpha{1};
      const TC beta{1};

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
    gemm_time = prk::wtime() - gemm_time;
  }
  // copy output back to host
  q.memcpy(&(h_c[0]), C, nelems * sizeof(TC)).wait();

  sycl::free(C, q);
  sycl::free(B, q);
  sycl::free(A, q);

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double forder = static_cast<double>(order);
  const double reference = 0.25 * prk::pow(forder,3) * prk::pow(forder-1.0,2) * (iterations+1);
  double checksum{0};
  for (int i=0; i<nelems; ++i) {
      checksum += h_c[i];
  }
  const double residuum = std::abs(checksum - reference) / reference;
  const double epsilon{1.0e-8};
  if ((residuum < epsilon) || (sizeof(TA) < 4)) {
#if VERBOSE
    std::cout << "Reference checksum = " << reference << "\n"
              << "Actual checksum = " << checksum << std::endl;
#endif
    std::cout << "Solution validates" << std::endl;
    auto avgtime = gemm_time/iterations;
    auto nflops = 2.0 * prk::pow(forder,3);

    auto isA_fp64 = (typeid(TA) == typeid(double));
    auto isA_fp32 = (typeid(TA) == typeid(float));
    auto isA_fp16 = (typeid(TA) == typeid(sycl::half));
    auto isA_bf16 = (typeid(TA) == typeid(oneapi::mkl::bfloat16));
    auto pnameA = (isA_fp64 ? "FP64" :
                   (isA_fp32 ? "FP32" :
                    (isA_fp16 ? "FP16" :
                     (isA_bf16 ? "BF16" : "Unknown FP type"))));

    auto isB_fp64 = (typeid(TB) == typeid(double));
    auto isB_fp32 = (typeid(TB) == typeid(float));
    auto isB_fp16 = (typeid(TB) == typeid(sycl::half));
    auto isB_bf16 = (typeid(TB) == typeid(oneapi::mkl::bfloat16));
    auto pnameB = (isB_fp64 ? "FP64" :
                   (isB_fp32 ? "FP32" :
                    (isB_fp16 ? "FP16" :
                     (isB_bf16 ? "BF16" : "Unknown FP type"))));

    auto isC_fp64 = (typeid(TC) == typeid(double));
    auto isC_fp32 = (typeid(TC) == typeid(float));
    auto isC_fp16 = (typeid(TC) == typeid(sycl::half));
    auto isC_bf16 = (typeid(TC) == typeid(oneapi::mkl::bfloat16));
    auto pnameC = (isC_fp64 ? "FP64" :
                   (isC_fp32 ? "FP32" :
                    (isC_fp16 ? "FP16" :
                     (isC_bf16 ? "BF16" : "Unknown FP type"))));

    std::cout << pnameA << "*" << pnameB << "=" << pnameC
              << " Rate (MF/s): " << 1.0e-6 * nflops/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
    std::cout << "Reference checksum = " << reference << "\n"
              << "Residuum           = " << residuum << std::endl;
  }

  sycl::free(h_c, q);
}

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels" << std::endl;
  std::cout << "C++11/oneMKL Dense matrix-matrix multiplication: C += A x B" << std::endl;

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

  //////////////////////////////////////////////////////////////////////
  /// Setup SYCL environment
  //////////////////////////////////////////////////////////////////////

  sycl::queue qs[2] = { sycl::queue{sycl::cpu_selector_v},
                        sycl::queue{sycl::gpu_selector_v} };
  for (auto q : qs) {
      try {
        prk::SYCL::print_device_platform(q);
        bool has_fp64 = prk::SYCL::has_fp64(q);
        run<sycl::half>(q, iterations, order);
        run<oneapi::mkl::bfloat16>(q, iterations, order);
        run3<sycl::half,sycl::half,float>(q, iterations, order);
        run3<oneapi::mkl::bfloat16,oneapi::mkl::bfloat16,float>(q, iterations, order);
        run<float>(q, iterations, order);
        if (has_fp64) {
          run<double>(q, iterations, order);
        } else {
          std::cout << "SYCL device lacks FP64 support." << std::endl;
        }
      }
      catch (sycl::exception & e) {
        std::cout << e.what() << std::endl;
        prk::SYCL::print_exception_details(e);
      }
      catch (std::exception & e) {
        std::cout << e.what() << std::endl;
      }
      catch (const char * e) {
        std::cout << e << std::endl;
      }
  }

  return 0;
}


