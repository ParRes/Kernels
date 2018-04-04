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
///          <progname> <# iterations> <matrix order> [<tile size>]
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// FUNCTIONS CALLED:
///
///          Other than OpenMP or standard C functions, the following
///          functions are used in this program:
///
///          wtime()
///
/// HISTORY: Written by Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, December, 2017.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_cuda.h"

__global__ void init(unsigned order, double * A, double * B, double * C)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i<order) && (j<order)) {
      A[i*order+j] = i;
      B[i*order+j] = i;
      C[i*order+j] = 0;
    }
}

__global__ void init(unsigned order, double * C)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i<order) && (j<order)) {
      C[i*order+j] = 0;
    }
}

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/CUBLAS Dense matrix-matrix multiplication: C += A x B" << std::endl;

  prk::CUDA::info info;
  info.print();

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int order;
  int input_copy = 0;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <matrix order> <copy input every iteration [0/1]>";
      }

      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      order = std::atoi(argv[2]);
      if (order <= 0) {
        throw "ERROR: Matrix Order must be greater than 0";
      } else if (order > std::floor(std::sqrt(INT_MAX))) {
        throw "ERROR: matrix dimension too large - overflow risk";
      }

      if (argc > 3) {
        input_copy = std::atoi(argv[3]);
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Matrix order         = " << order << std::endl;
  std::cout << "Input copy           = " << (input_copy ? "yes" : "no") << std::endl;

  cublasHandle_t h;
  prk::CUDA::check( cublasCreate(&h) );

  int tile_size = 32;
  dim3 dimGrid(prk::divceil(order,tile_size),prk::divceil(order,tile_size),1);
  dim3 dimBlock(tile_size, tile_size, 1);

  info.checkDims(dimBlock, dimGrid);

  //////////////////////////////////////////////////////////////////////
  // Allocate space for matrices
  //////////////////////////////////////////////////////////////////////

  double dgemm_time(0);

  const size_t nelems = (size_t)order * (size_t)order;
  const size_t bytes = nelems * sizeof(double);

  // host buffers
  double * h_a;
  double * h_b;
  double * h_c;
  prk::CUDA::check( cudaMallocHost((void**)&h_a, bytes) );
  prk::CUDA::check( cudaMallocHost((void**)&h_b, bytes) );
  prk::CUDA::check( cudaMallocHost((void**)&h_c, bytes) );

  // device buffers
  double * d_a;
  double * d_b;
  double * d_c;
  prk::CUDA::check( cudaMalloc((void**)&d_a, bytes) );
  prk::CUDA::check( cudaMalloc((void**)&d_b, bytes) );
  prk::CUDA::check( cudaMalloc((void**)&d_c, bytes) );

  if (input_copy) {

    for (int i=0; i<order; ++i) {
      for (int j=0; j<order; ++j) {
         h_a[i*order+j] = i;
         h_b[i*order+j] = i;
      }
    }

    prk::CUDA::check( cudaMemcpy(d_a, &(h_a[0]), bytes, cudaMemcpyHostToDevice) );
    prk::CUDA::check( cudaMemcpy(d_b, &(h_b[0]), bytes, cudaMemcpyHostToDevice) );

    init<<<dimGrid, dimBlock>>>(order, d_c);

  } else {

    init<<<dimGrid, dimBlock>>>(order, d_a, d_b, d_c);

  }

  {
    for (auto iter = 0; iter<=iterations; iter++) {

      if (iter==1) dgemm_time = prk::wtime();

      if (input_copy) {
        prk::CUDA::check( cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice) );
        prk::CUDA::check( cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice) );
      }

      double one(1);
      prk::CUDA::check( cublasDgemm(h,
                                    CUBLAS_OP_N, CUBLAS_OP_N, // opA, opB
                                    order, order, order,      // m, n, k
                                    &one,                     // alpha
                                    d_a, order,               // A, lda
                                    d_b, order,               // B, ldb
                                    &one,                     // beta
                                    d_c, order) );            // C, ldc
      prk::CUDA::check( cudaDeviceSynchronize() );
    }
    dgemm_time = prk::wtime() - dgemm_time;
  }

  // copy output back to host
  prk::CUDA::check( cudaMemcpy(&(h_c[0]), d_c, bytes, cudaMemcpyDeviceToHost) );

  prk::CUDA::check( cudaFree(d_c) );
  prk::CUDA::check( cudaFree(d_b) );
  prk::CUDA::check( cudaFree(d_a) );

  prk::CUDA::check( cudaFreeHost(h_a) );
  prk::CUDA::check( cudaFreeHost(h_b) );

  prk::CUDA::check( cublasDestroy(h) );

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const auto epsilon = 1.0e-8;
  const auto forder = static_cast<double>(order);
  const auto reference = 0.25 * std::pow(forder,3) * std::pow(forder-1.0,2) * (iterations+1);
  const auto checksum = prk_reduce( &(h_c[0]), &(h_c[nelems]), 0.0);
  const auto residuum = std::abs(checksum-reference)/reference;

  if (residuum < epsilon) {
#if VERBOSE
    std::cout << "Reference checksum = " << reference << "\n"
              << "Actual checksum = " << checksum << std::endl;
#endif
    std::cout << "Solution validates" << std::endl;
    auto avgtime = dgemm_time/iterations;
    auto nflops = 2.0 * std::pow(forder,3);
    std::cout << "Rate (MF/s): " << 1.0e-6 * nflops/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
    std::cout << "Reference checksum = " << reference << "\n"
              << "Actual checksum = " << checksum << std::endl;
    return 1;
  }

  prk::CUDA::check( cudaFreeHost(h_c) );

  return 0;
}


