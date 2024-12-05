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
#include "prk_cuda.h"

prk::CUDA::info info;

template <typename T>
__global__ void init(int order, T * C)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i<order) && (j<order)) {
      C[i*order+j] = T(0);
    }
}

template <typename T>
__global__ void init(int order, T * A, T * B, T * C)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i<order) && (j<order)) {
      A[i*order+j] = T(i);
      B[i*order+j] = T(i);
      C[i*order+j] = T(0);
    }
}

template <typename TAB, typename TC>
void prk_gemm(const cublasHandle_t & h,
              const int order, const TC alpha, const TC beta,
              const TAB * A, const TAB * B, TC * C)
{
    std::cerr << "No valid template match for type T" << std::endl;
    std::abort();
}

template <>
void prk_gemm(const cublasHandle_t & h,
              const int order, const __half alpha, const __half beta,
              const __half * A, const __half * B, __half * C)
{
    prk::CUDA::check( cublasHgemm(h,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  order, order, order,
                                  &alpha,
                                  A, order,
                                  B, order,
                                  &beta,
                                  C, order) );
}

template <>
void prk_gemm(const cublasHandle_t & h,
              const int order, const float alpha, const float beta,
              const float * A, const float * B, float * C)
{
    prk::CUDA::check( cublasSgemm(h,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  order, order, order,
                                  &alpha,
                                  A, order,
                                  B, order,
                                  &beta,
                                  C, order) );
}

template <>
void prk_gemm(const cublasHandle_t & h,
              const int order, const double alpha, const double beta,
              const double * A, const double * B, double * C)
{
    prk::CUDA::check( cublasDgemm(h,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  order, order, order,
                                  &alpha,
                                  A, order,
                                  B, order,
                                  &beta,
                                  C, order) );
}

template <typename T>
void run(const cublasHandle_t & h, int iterations, int order)
{
  double gemm_time{0};

  const size_t nelems = (size_t)order * (size_t)order;
  auto h_c = prk::CUDA::malloc_host<T>( nelems);

  const int tile_size = 32;
  dim3 dimGrid(prk::divceil(order,tile_size),prk::divceil(order,tile_size),1);
  dim3 dimBlock(tile_size, tile_size, 1);
  info.checkDims(dimBlock, dimGrid);

  auto d_a = prk::CUDA::malloc_device<T>(nelems);
  auto d_b = prk::CUDA::malloc_device<T>(nelems);
  auto d_c = prk::CUDA::malloc_device<T>(nelems);
  init<<<dimGrid, dimBlock>>>(order, d_a, d_b, d_c);
  prk::CUDA::sync();
  {
    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) gemm_time = prk::wtime();

      const T alpha{1};
      const T beta{1};

      prk_gemm(h, order, alpha, beta, d_a, d_b, d_c);
      prk::CUDA::sync();
    }
    gemm_time = prk::wtime() - gemm_time;
  }
  // copy output back to host
  prk::CUDA::copyD2H(h_c, d_c, nelems);

  prk::CUDA::free(d_a);
  prk::CUDA::free(d_b);
  prk::CUDA::free(d_c);

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double forder = static_cast<double>(order);
  const double reference = 0.25 * prk::pow(forder,3) * prk::pow(forder-1.0,2) * (iterations+1);
  double checksum{0};
  for (int i=0; i<nelems; ++i) {
      checksum += double(h_c[i]);
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
    auto is_fp16 = (typeid(T) == typeid(__half));
    auto pname = (is_fp64 ? "FP64" :
                  (is_fp32 ? "FP32" :
                   (is_fp16 ? "FP16" : "Unknown FP type")));
    prk::print_flop_rate_time(pname, nflops/avgtime, avgtime);
  } else {
    std::cout << "Reference checksum = " << reference << "\n"
              << "Residuum           = " << residuum << std::endl;
  }

  prk::CUDA::free_host(h_c);
}

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/CUBLAS Dense matrix-matrix multiplication: C += A x B" << std::endl;

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

  //info.print();

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Matrix order         = " << order << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Setup CUBLAS environment
  //////////////////////////////////////////////////////////////////////

  cublasHandle_t h;
  prk::CUDA::check( cublasCreate(&h) );
  run<__half>(h, iterations, order);
  run<float>(h, iterations, order);
  run<double>(h, iterations, order);
  prk::CUDA::check( cublasDestroy(h) );

  return 0;
}
