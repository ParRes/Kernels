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
/// * Neither the name of <COPYRIGHT HOLDER> nor the names of its
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
#include "prk_cuda.h"

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/CUDA Matrix transpose: B = A^T" << std::endl;

  //prk::CUDAinfo();

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int order;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <matrix order>";
      }

      // number of times to do the transpose
      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      // order of a the matrix
      order = std::atol(argv[2]);
      if (order <= 0) {
        throw "ERROR: Matrix Order must be greater than 0";
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Matrix order          = " << order << std::endl;
  std::cout << "Number of iterations  = " << iterations << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Setup CUDA environment
  //////////////////////////////////////////////////////////////////////

  cublasHandle_t h;
  cublasCreate(&h);

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

  const size_t nelems = (size_t)order * (size_t)order;
  const size_t bytes = nelems * sizeof(float);
  float * h_a;
  float * h_b;
  prk::CUDAcheck( cudaMallocHost((float**)&h_a, bytes) );
  prk::CUDAcheck( cudaMallocHost((float**)&h_b, bytes) );

  // fill A with the sequence 0 to order^2-1 as floats
  for (auto j=0; j<order; j++) {
    for (auto i=0; i<order; i++) {
      h_a[j*order+i] = order*j+i;
      h_b[j*order+i] = 0.0f;
    }
  }

  // copy input from host to device
  float * d_a;
  float * d_b;
  prk::CUDAcheck( cudaMalloc((float**)&d_a, bytes) );
  prk::CUDAcheck( cudaMalloc((float**)&d_b, bytes) );
  prk::CUDAcheck( cudaMemcpy(d_a, &(h_a[0]), bytes, cudaMemcpyHostToDevice) );
  prk::CUDAcheck( cudaMemcpy(d_b, &(h_b[0]), bytes, cudaMemcpyHostToDevice) );

#if 1
  // We need a vector of ones because CUBLAS saxpy do does
  // correctly implement incx=0.
  float * h_o;
  prk::CUDAcheck( cudaMallocHost((float**)&h_o, bytes) );
  for (auto j=0; j<order; j++) {
    for (auto i=0; i<order; i++) {
      h_o[j*order+i] = 1.0f;
    }
  }
  float * d_o;
  prk::CUDAcheck( cudaMalloc((float**)&d_o, bytes) );
  prk::CUDAcheck( cudaMemcpy(d_o, &(h_o[0]), bytes, cudaMemcpyHostToDevice) );
#endif

#ifdef USE_HOST_BUFFERS
  float p_a = h_a;
  float p_b = h_b;
  float p_o = h_o;
#else
  float * p_a = d_a;
  float * p_b = d_b;
  float * p_o = d_o;
#endif

  auto trans_time = 0.0;

  for (auto iter = 0; iter<=iterations; iter++) {

    if (iter==1) trans_time = prk::wtime();

    float one(1);
    // B += trans(A) i.e. B = trans(A) + B
    cublasSgeam(h,
                CUBLAS_OP_T, CUBLAS_OP_N,   // opA, opB
                order, order,               // m, n
                &one, p_a, order,           // alpha, A, lda
                &one, p_b, order,           // beta, B, ldb
                p_b, order);                // C, ldc (in-place for B)
    // A += 1.0 i.e. A = 1.0 * 1.0 + A
#if 0
    // THIS IS BUGGY
    cublasSaxpy(h,
                order*order,                // n
                &one,                       // alpha
                &one, 0,                    // x, incx
                p_a, 1);                    // y, incy
#else
    // THIS IS CORRECT
    cublasSaxpy(h,
                order*order,                // n
                &one,                       // alpha
                p_o, 1,                     // x, incx
                p_a, 1);                    // y, incy
#endif
    // (Host buffer version)
    // The performance is ~10% better if this is done every iteration,
    // instead of only once before the timer is stopped.
    prk::CUDAcheck( cudaDeviceSynchronize() );
  }
  trans_time = prk::wtime() - trans_time;

  // copy output back to host
  prk::CUDAcheck( cudaMemcpy(&(h_b[0]), d_b, bytes, cudaMemcpyDeviceToHost) );

#if 1
  prk::CUDAcheck( cudaFree(d_o) );
  prk::CUDAcheck( cudaFreeHost(h_o) );
#endif

  prk::CUDAcheck( cudaFree(d_b) );
  prk::CUDAcheck( cudaFree(d_a) );
  prk::CUDAcheck( cudaFreeHost(h_a) );

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  // TODO: replace with std::generate, std::accumulate, or similar
  const auto addit = (iterations+1.) * (iterations/2.);
  auto abserr = 0.0;
  for (auto j=0; j<order; j++) {
    for (auto i=0; i<order; i++) {
      const size_t ij = (size_t)i*(size_t)order+(size_t)j;
      const size_t ji = (size_t)j*(size_t)order+(size_t)i;
      const float reference = static_cast<float>(ij)*(1.+iterations)+addit;
      abserr += std::fabs(h_b[ji] - reference);
    }
  }

#ifdef VERBOSE
  std::cout << "Sum of absolute differences: " << abserr << std::endl;
#endif

  const auto epsilon = 1.0e-8;
  if (abserr < epsilon) {
    std::cout << "Solution validates" << std::endl;
    auto avgtime = trans_time/iterations;
    auto bytes = (size_t)order * (size_t)order * sizeof(float);
    std::cout << "Rate (MB/s): " << 1.0e-6 * (2L*bytes)/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
#ifdef VERBOSE
    for (auto i=0; i<order; i++) {
      for (auto j=0; j<order; j++) {
        std::cout << "(" << i << "," << j << ") = " << h_a[i*order+j] << ", " << h_b[i*order+j] << "\n";
      }
    }
#endif
    std::cout << "ERROR: Aggregate squared error " << abserr
              << " exceeds threshold " << epsilon << std::endl;
    return 1;
  }

  prk::CUDAcheck( cudaFreeHost(h_b) );

  return 0;
}


