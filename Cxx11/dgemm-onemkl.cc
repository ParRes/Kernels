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
///          cblasDgemm()
///          cublasDgemmStridedBatched()
///
/// HISTORY: Written by Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, December, 2017.
///
//////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "prk_util.h"
#include "prk_dpct.h"
#include <cmath>

void init(int order, const int matrices, double * A, double * B, double * C, sycl::nd_item<3> item_ct1)
{
    auto i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
    auto j = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) + item_ct1.get_local_id(1);

    for (int b=0; b<matrices; ++b) {
      if ((i<order) && (j<order)) {
        A[b*order*order+i*order+j] = i;
        B[b*order*order+i*order+j] = i;
        C[b*order*order+i*order+j] = 0;
      }
    }
}

void init(int order, const int matrices, double * C, sycl::nd_item<3> item_ct1)
{
    auto i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
    auto j = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) + item_ct1.get_local_id(1);

    for (int b=0; b<matrices; ++b) {
      if ((i<order) && (j<order)) {
        C[b*order*order+i*order+j] = 0;
      }
    }
}

void prk_dgemm(const sycl::queue &h, const int order, const int batches, double *A, double *B, double *C)
{
    const double alpha = 1.0;
    const double beta  = 1.0;

    for (int b=0; b<batches; ++b) {
        double * pA = &(A[b*order*order]);
        double * pB = &(B[b*order*order]);
        double * pC = &(C[b*order*order]);
        mkl::blas::gemm(h, mkl::transpose::nontrans, // opA
                           mkl::transpose::nontrans, // opB
                           order, order, order,      // m, n, k
                           alpha,                    // alpha
                           pA, order,                // A, lda
                           pB, order,                // B, ldb
                           beta,                     // beta
                           pC, order);               // C, ldc
    }
    dpct::get_current_device().queues_wait_and_throw();
}

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
        } else if (order > std::floor(std::sqrt(INT_MAX))) {
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

  sycl::queue h(dpct::get_default_context(), dpct::get_current_device());

  const int tile_size = 32;
  sycl::range<3> dimGrid(prk::divceil(order, tile_size), prk::divceil(order, tile_size), 1);
  sycl::range<3> dimBlock(tile_size, tile_size, 1);

  //////////////////////////////////////////////////////////////////////
  // Allocate space for matrices
  //////////////////////////////////////////////////////////////////////

  double dgemm_time(0);

  const int matrices = (batches == 0 ? 1 : abs(batches));
  const size_t nelems = (size_t)order * (size_t)order;
  const size_t bytes = nelems * sizeof(double);

  // host buffers
  double * h_a = sycl::malloc_host<double>(nelems, dpct::get_default_context()),
  double * h_b = sycl::malloc_host<double>(nelems, dpct::get_default_context()),
  double * h_c = sycl::malloc_host<double>(nelems, dpct::get_default_context()),

  // device buffers
  double * d_a = sycl::malloc_device<double>( matrices * nelems, dpct::get_current_device(), dpct::get_default_context();
  double * d_b = sycl::malloc_device<double>( matrices * nelems, dpct::get_current_device(), dpct::get_default_context();
  double * d_c = sycl::malloc_device<double>( matrices * nelems, dpct::get_current_device(), dpct::get_default_context();

  if (input_copy) {

    for (int i=0; i<order; ++i) {
      for (int j=0; j<order; ++j) {
         h_a[i*order+j] = i;
         h_b[i*order+j] = i;
      }
    }

    for (int b=0; b<matrices; ++b) {
            dpct::get_default_queue().memcpy( &(d_a[b * order * order]), h_a, bytes);
            dpct::get_default_queue().memcpy( &(d_b[b * order * order]), h_b, bytes);
    }
    dpct::get_current_device().queues_wait_and_throw();

    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto dpct_global_range = dimGrid * dimBlock;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)),
                              sycl::range<3>(dimBlock.get(2), dimBlock.get(1), dimBlock.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
                init(order, matrices, d_c, item_ct1);
            });
    });

  } else {

      dpct::get_default_queue().submit([&](sycl::handler &cgh) {
          auto dpct_global_range = dimGrid * dimBlock;

          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)),
                                sycl::range<3>(dimBlock.get(2), dimBlock.get(1), dimBlock.get(0))),
              [=](sycl::nd_item<3> item_ct1) {
                  init(order, matrices, d_a, d_b, d_c, item_ct1);
              });
      });
  }
  dpct::get_current_device().queues_wait_and_throw();

  double xfer(0);
  double comp(0);
  {
    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) dgemm_time = prk::wtime();

      if (input_copy) {
        double t0 = prk::wtime();
        for (int b=0; b<matrices; ++b) {
            dpct::get_default_queue().memcpy( &(d_a[b * order * order]), h_a, bytes);
            dpct::get_default_queue().memcpy( &(d_b[b * order * order]), h_b, bytes);
        }
        dpct::get_current_device().queues_wait_and_throw();
        double t1 = prk::wtime();
        if (iter==1) xfer += (t1-t0);
      }

      {
        double t0 = prk::wtime();
        prk_dgemm(h, order, matrices, d_a, d_b, d_c);
        double t1 = prk::wtime();
        if (iter==1) comp += (t1-t0);
      }
    }
    dgemm_time = prk::wtime() - dgemm_time;
  }
  std::cout << "xfer, comp = " << xfer << "," << comp << std::endl;

  // copy output back to host
  dpct::get_default_queue().memcpy(&(h_c[0]), d_c, matrices * bytes);
  sycl::free(d_c, dpct::get_default_context());
  sycl::free(d_b, dpct::get_default_context());
  sycl::free(d_a, dpct::get_default_context());
  sycl::free(h_a, dpct::get_default_context());
  sycl::free(h_b, dpct::get_default_context());
  dpct::get_current_device().queues_wait_and_throw();

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const auto epsilon = 1.0e-8;
  const auto forder = static_cast<double>(order);
  const auto reference = 0.25 * std::pow(forder, 3) * std::pow(forder - 1.0, 2) * (iterations + 1);
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
        auto nflops = 2.0 * std::pow(forder, 3);
    std::cout << "Rate (MF/s): " << 1.0e-6 * nflops/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
    std::cout << "Reference checksum = " << reference << "\n"
              << "Residuum           = " << residuum << std::endl;
    return 1;
  }

  sycl::free(h_c, dpct::get_default_context());

  return 0;
}


