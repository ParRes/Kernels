///
/// Copyright (c) 2017, Intel Corporation
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
///          CUDA STF by Cedric Augonnet, October 2024.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_cuda.h"
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main(int argc, char * argv[])
{
  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11 CUDA STF Dense matrix-matrix multiplication: C += A x B" << std::endl;

  int iterations;
  int order;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <matrix order> [tile size]";
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
  /// Allocate space for matrices
  //////////////////////////////////////////////////////////////////////

  context ctx;

  double dgemm_time{0};

  auto A = ctx.logical_data(shape_of<slice<double, 2>>(order, order));
  auto B = ctx.logical_data(shape_of<slice<double, 2>>(order, order));
  auto C = ctx.logical_data(shape_of<slice<double, 2>>(order, order));

  ctx.parallel_for(A.shape(), A.write(), B.write(), C.write())->*[]__device__(size_t i, size_t j, auto dA, auto dB, auto dC)
  {
      dA(i, j) = (double)i;
      dB(i, j) = (double)i;
      dC(i, j) = 0.0;
  };

  {
    for (int iter = 0; iter<=iterations; iter++) {
      if (iter==1) {
          cudaStreamSynchronize(ctx.task_fence());
          dgemm_time = prk::wtime();
      }

      ctx.parallel_for(C.shape(), A.read(), B.read(), C.rw())->*[]__device__ (size_t i, size_t j, auto dA, auto dB, auto dC)
      {
          double Ctemp(0);
          for (size_t k = 0; k < dC.extent(0); k++) {
              Ctemp += dA(i, k)*dB(k, j);
          }
          dC(i,j) += Ctemp;
      };
    }

    cudaStreamSynchronize(ctx.task_fence());
    dgemm_time = prk::wtime() - dgemm_time;
  }

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const auto forder = static_cast<double>(order);
  const auto reference = 0.25 * prk::pow(forder,3) * prk::pow(forder-1.0,2) * (iterations+1);

  double checksum;
  ctx.host_launch(C.read())->*[&](auto hC)
  {
      for (size_t j = 0; j < hC.extent(1); j++)
      for (size_t i = 0; i < hC.extent(0); i++)
      {
          checksum += hC(i, j);
      }
  };

  cudaStreamSynchronize(ctx.task_fence());

  const auto epsilon = 1.0e-8;
  const auto residuum = prk::abs(checksum-reference)/reference;
  if (residuum < epsilon) {
#if VERBOSE
    std::cout << "Reference checksum = " << reference << "\n"
              << "Actual checksum = " << checksum << std::endl;
#endif
    std::cout << "Solution validates" << std::endl;
    auto avgtime = dgemm_time/iterations;
    auto nflops = 2.0 * prk::pow(forder,3);
    std::cout << "Rate (MF/s): " << 1.0e-6 * nflops/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
    std::cout << "Reference checksum = " << reference << "\n"
              << "Actual checksum = " << checksum << std::endl;
#if VERBOSE
    for (int i=0; i<order; ++i)
      for (int j=0; j<order; ++j)
        std::cout << "A(" << i << "," << j << ") = " << A[i*order+j] << "\n";
    for (int i=0; i<order; ++i)
      for (int j=0; j<order; ++j)
        std::cout << "B(" << i << "," << j << ") = " << B[i*order+j] << "\n";
    for (int i=0; i<order; ++i)
      for (int j=0; j<order; ++j)
        std::cout << "C(" << i << "," << j << ") = " << C[i*order+j] << "\n";
    std::cout << std::endl;
#endif
    return 1;
  }

  ctx.finalize();

  return 0;
}


