///
/// Copyright (c) 2018, Intel Corporation
/// Copyright (c) 2021, NVIDIA
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
/// NAME:    sgemm
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
///          cblasSgemm()
///          hipblasSgemmStridedBatched()
///
/// HISTORY: Written by Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, December, 2017.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_hip.h"

#if 0
__global__ void init(unsigned order, float * A, float * B, float * C)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i<order) && (j<order)) {
      A[i*order+j] = i;
      B[i*order+j] = i;
      C[i*order+j] = 0;
    }
}

__global__ void init(unsigned order, float * C)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i<order) && (j<order)) {
      C[i*order+j] = 0;
    }
}
#endif

__global__ void init(int order, const int matrices, float * A, float * B, float * C)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    for (int b=0; b<matrices; ++b) {
      if ((i<order) && (j<order)) {
        A[b*order*order+i*order+j] = i;
        B[b*order*order+i*order+j] = i;
        C[b*order*order+i*order+j] = 0;
      }
    }
}

__global__ void init(int order, const int matrices, float * C)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    for (int b=0; b<matrices; ++b) {
      if ((i<order) && (j<order)) {
        C[b*order*order+i*order+j] = 0;
      }
    }
}

void prk_sgemm(const hipblasHandle_t & h,
               const int order,
               const int batches,
               float * A,
               float * B,
               float * C)
{
    const float alpha = 1.0;
    const float beta  = 1.0;

    for (int b=0; b<batches; ++b) {
        float * pA = &(A[b*order*order]);
        float * pB = &(B[b*order*order]);
        float * pC = &(C[b*order*order]);
        prk::HIP::check( hipblasSgemm(h,
                                      HIPBLAS_OP_N, HIPBLAS_OP_N, // opA, opB
                                      order, order, order,      // m, n, k
                                      &alpha,                   // alpha
                                      pA, order,                // A, lda
                                      pB, order,                // B, ldb
                                      &beta,                    // beta
                                      pC, order) );             // C, ldc
    }
}

void prk_bgemm(const hipblasHandle_t & h,
               const int order,
               const int batches,
               float * A,
               float * B,
               float * C)
{
    const float alpha = 1.0;
    const float beta  = 1.0;

    prk::HIP::check( hipblasSgemmStridedBatched(h,
                                                HIPBLAS_OP_N, HIPBLAS_OP_N,
                                                order, order, order,
                                                &alpha,
                                                (const float *)A, order, order*order,
                                                (const float *)B, order, order*order,
                                                &beta,
                                                C, order, order*order,
                                                batches) );

    //  hipblasStatus_t hipblassgemmBatched(hipblasHandle_t handle,
    //                                    hipblasOperation_t transa,
    //                                    hipblasOperation_t transb,
    //                                    int m, int n, int k,
    //                                    const float          *alpha,
    //                                    const float          *Aarray[], int lda,
    //                                    const float          *Barray[], int ldb,
    //                                    const float          *beta,
    //                                    float          *Carray[], int ldc,
    //                                    int batchCount)
}

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/HIPBLAS Dense matrix-matrix multiplication: C += A x B" << std::endl;

  prk::HIP::info info;
  //info.print();

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

      if (argc > 3) {
        batches = std::atoi(argv[3]);
      }

      if (argc > 4) {
        input_copy = std::atoi(argv[4]);
        if (input_copy != 0 && input_copy != 1) {
          throw "ERROR: input_copy was not 0 or 1";
        }
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
  } else if (batches < 0) {
      std::cout << "Batch size           = " << -batches << " (loop over legacy BLAS)" << std::endl;
  } else if (batches > 0) {
      std::cout << "Batch size           = " <<  batches << " (batched BLAS)" << std::endl;
  }
  std::cout << "Input copy           = " << (input_copy ? "yes" : "no") << std::endl;

  hipblasHandle_t h;
  prk::HIP::check( hipblasCreate(&h) );

  const int tile_size = 32;
  dim3 dimGrid(prk::divceil(order,tile_size),prk::divceil(order,tile_size),1);
  dim3 dimBlock(tile_size, tile_size, 1);

  info.checkDims(dimBlock, dimGrid);

  //////////////////////////////////////////////////////////////////////
  // Allocate space for matrices
  //////////////////////////////////////////////////////////////////////

  double gemm_time(0);

  const int matrices = (batches==0 ? 1 : abs(batches));
  const size_t nelems = (size_t)order * (size_t)order;

  // host buffers
  auto h_a = prk::HIP::malloc_host<float>(nelems);
  auto h_b = prk::HIP::malloc_host<float>(nelems);
  auto h_c = prk::HIP::malloc_host<float>(matrices*nelems);

  // device buffers
  auto d_a = prk::HIP::malloc_device<float>(matrices*nelems);
  auto d_b = prk::HIP::malloc_device<float>(matrices*nelems);
  auto d_c = prk::HIP::malloc_device<float>(matrices*nelems);

  if (input_copy) {
    for (int i=0; i<order; ++i) {
      for (int j=0; j<order; ++j) {
         h_a[i*order+j] = (float)i;
         h_b[i*order+j] = (float)i;
      }
    }

    for (int b=0; b<matrices; ++b) {
      prk::HIP::copyH2Dasync(&(d_a[b*nelems]), h_a, nelems);
      prk::HIP::copyH2Dasync(&(d_b[b*nelems]), h_b, nelems);
    }
    prk::HIP::sync();

    hipLaunchKernelGGL(init, dim3(dimGrid), dim3(dimBlock), 0, 0, order, matrices, d_c);

  } else {

    hipLaunchKernelGGL(init, dim3(dimGrid), dim3(dimBlock), 0, 0, order, matrices, d_a, d_b, d_c);

  }
  prk::HIP::sync();

  double xfer(0);
  double comp(0);
  {
    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) {
          prk::HIP::sync();
          gemm_time = prk::wtime();
      }

      if (input_copy) {
        double t0 = prk::wtime();
        for (int b=0; b<matrices; ++b) {
          prk::HIP::copyH2Dasync(&(d_a[b*nelems]), h_a, nelems);
          prk::HIP::copyH2Dasync(&(d_b[b*nelems]), h_b, nelems);
        }
        prk::HIP::sync();
        double t1 = prk::wtime();
        if (iter==1) xfer += (t1-t0);
      }

      {
        double t0 = prk::wtime();
        if (batches > 0) {
          prk_bgemm(h, order, matrices, d_a, d_b, d_c);
        } else {
          prk_sgemm(h, order, matrices, d_a, d_b, d_c);
        }
        double t1 = prk::wtime();
        if (iter==1) comp += (t1-t0);
      }
    }
    prk::HIP::sync();
    gemm_time = prk::wtime() - gemm_time;
  }
  std::cout << "xfer, comp = " << xfer << "," << comp << std::endl;

  // copy output back to host
  prk::HIP::copyD2H(h_c, d_c, matrices*nelems);

  prk::HIP::free(d_a);
  prk::HIP::free(d_b);
  prk::HIP::free(d_c);

  prk::HIP::free_host(h_a);
  prk::HIP::free_host(h_b);

  prk::HIP::check( hipblasDestroy(h) );

  prk::HIP::sync();

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double epsilon = 1.0e-8;
  const double forder = static_cast<double>(order);
  const double reference = 0.25 * prk::pow(forder,3) * prk::pow(forder-1.0,2) * (iterations+1);
  double residuum{0};
  for (int b=0; b<matrices; ++b) {
      // this template produces an inaccurate result in single precision
      //const double checksum = prk::reduce( &(h_c[b*order*order+0]), &(h_c[b*order*order+nelems]), 0.0);
      double checksum{0};
      for (int i=0; i<nelems; ++i) {
          checksum += h_c[b*order*order+i];
      }
      residuum += std::abs(checksum-reference)/reference;
  }
  residuum /= matrices;

  if (1 || residuum < epsilon) {
#if VERBOSE
    std::cout << "Reference checksum = " << reference << "\n"
              << "Actual checksum = " << checksum << std::endl;
#endif
    std::cout << "Solution validates" << std::endl;
    auto avgtime = gemm_time/iterations/matrices;
    auto nflops = 2.0 * prk::pow(forder,3);
    std::cout << "Rate (MF/s): " << 1.0e-6 * nflops/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
    std::cout << "Reference checksum = " << reference << "\n"
              << "Residuum           = " << residuum << std::endl;
    return 1;
  }

  prk::HIP::free_host(h_c);

  return 0;
}


