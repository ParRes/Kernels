///
/// Copyright (c) 2018, Intel Corporation
/// Copyright (c) 2024, NVIDIA
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
///          cblasDgemm()
///          cublasDgemmStridedBatched()
///
/// HISTORY: Written by Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, December, 2017.
///          CUDA STF by Cedric Augonnet, October 2024.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_cuda.h"

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/CUBLAS STF Dense matrix-matrix multiplication: C += A x B" << std::endl;

  prk::CUDA::info info;
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

  cublasHandle_t h;
  prk::CUDA::check( cublasCreate(&h) );

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

  const auto epsilon = 1.0e-8;
  const auto forder = static_cast<double>(order);
  const auto reference = 0.25 * prk::pow(forder,3) * prk::pow(forder-1.0,2) * (iterations+1);
  double residuum(0);

  context ctx;

  if (batches > 0) {
      /*
       * BATCHED implementation
       */
      auto a = ctx.logical_data(shape_of<slice<double, 3>>(order, order, matrices));
      auto b = ctx.logical_data(shape_of<slice<double, 3>>(order, order, matrices));
      auto c = ctx.logical_data(shape_of<slice<double, 3>>(order, order, matrices));

      // Initialize all matrices
      ctx.parallel_for(a.shape(), a.write(), b.write(), c.write())->*[] __device__ (size_t i, size_t j, size_t k, auto da, auto db, auto dc)
      {
          da(i, j, k) = (double)i;
          db(i, j, k) = (double)i;
          dc(i, j, k) = 0.0;
      };

      for (int iter = 0; iter<=iterations; iter++) {
          if (iter==1) {
              cudaStreamSynchronize(ctx.task_fence());
              gemm_time = prk::wtime();
          }

          const double alpha = 1.0;
          const double beta  = 1.0;
          ctx.task(a.read(), b.read(), c.rw())->*[&](cudaStream_t stream, auto da, auto db, auto dc) {
              cublasSetStream(h, stream);
              prk::CUDA::check( cublasDgemmStridedBatched(h,
                                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                                          order, order, order,
                                                          &alpha,
                                                          (const double *)da.data_handle(), order, order*order,
                                                          (const double *)db.data_handle(), order, order*order,
                                                          &beta,
                                                          dc.data_handle(), order, order*order,
                                                          batches) );
          };
      }

      cudaStreamSynchronize(ctx.task_fence());
      gemm_time = prk::wtime() - gemm_time;

      ctx.host_launch(c.read())->*[&](auto hc)
      {
          for (size_t k = 0; k < hc.extent(2); k++)
          {
              double checksum = 0.0;

              for (size_t j = 0; j < hc.extent(1); j++)
              for (size_t i = 0; i < hc.extent(0); i++)
              {
                  checksum += hc(i, j, k);
              }

              residuum += std::abs(checksum-reference)/reference;
          }
          residuum /= matrices;
      };
  }
  else {
      ::std::vector<logical_data<slice<double, 2>>> vector_a;
      ::std::vector<logical_data<slice<double, 2>>> vector_b;
      ::std::vector<logical_data<slice<double, 2>>> vector_c;

      // Initialize independant matrices
      for (size_t k = 0; k < matrices; k++) {
          auto ak = ctx.logical_data(shape_of<slice<double, 2>>(order, order));
          auto bk = ctx.logical_data(shape_of<slice<double, 2>>(order, order));
          auto ck = ctx.logical_data(shape_of<slice<double, 2>>(order, order));

          vector_a.push_back(ak);
          vector_b.push_back(bk);
          vector_c.push_back(ck);

          ctx.parallel_for(ak.shape(), ak.write(), bk.write(), ck.write())->*[] __device__ (size_t i, size_t j, auto dak, auto dbk, auto dck)
          {
              dak(i, j) = (double)i;
              dbk(i, j) = (double)i;
              dck(i, j) = 0.0;
          };
      }

      for (int iter = 0; iter<=iterations; iter++) {
          if (iter==1) {
              cudaStreamSynchronize(ctx.task_fence());
              gemm_time = prk::wtime();
          }

          const double alpha = 1.0;
          const double beta  = 1.0;

          for (size_t k = 0; k < matrices; k++)
          {
             ctx.task(vector_a[k].read(), vector_b[k].read(), vector_c[k].rw())->*[&](cudaStream_t stream, auto dA, auto dB, auto dC) {
                 cublasSetStream(h, stream);
                 prk::CUDA::check( cublasDgemm(h,
                                               CUBLAS_OP_N, CUBLAS_OP_N, // opA, opB
                                               order, order, order,      // m, n, k
                                               &alpha,                   // alpha
                                               dA.data_handle(), order,                // A, lda
                                               dB.data_handle(), order,                // B, ldb
                                               &beta,                    // beta
                                               dC.data_handle(), order) );             // C, ldc
             };
          }
      }

      cudaStreamSynchronize(ctx.task_fence());
      gemm_time = prk::wtime() - gemm_time;

      for (size_t k = 0; k < matrices; k++)
      {
          double checksum = 0.0;
          ctx.host_launch(vector_c[k].read())->*[&](auto hck)
          {
             for (size_t j = 0; j < hck.extent(1); j++)
             for (size_t i = 0; i < hck.extent(0); i++)
             {
                 checksum += hck(i, j);
             }
          };

          cudaStreamSynchronize(ctx.task_fence());
          residuum += std::abs(checksum-reference)/reference;

      }
      residuum /= matrices;

  }

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  if (residuum < epsilon) {
#if VERBOSE
    std::cout << "Reference checksum = " << reference << "\n"
              << "Actual checksum = " << checksum << std::endl;
#endif
    std::cout << "Solution validates" << std::endl;
    auto avgtime = gemm_time/iterations/matrices;
    auto nflops = 2.0 * prk::pow(forder,3);
    prk::print_flop_rate_time("FP64", nflops/avgtime, avgtime);
  } else {
    std::cout << "Reference checksum = " << reference << "\n"
              << "Residuum           = " << residuum << std::endl;
    return 1;
  }

  ctx.finalize();

  return 0;
}


