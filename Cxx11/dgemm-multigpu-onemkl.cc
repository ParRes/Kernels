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

#if 0
void prk_bgemm(const sycl::queue &h, const int order, const int batches,
               double *A, double *B, double *C)
{
    const double alpha = 1.0;
    const double beta  = 1.0;

    cublasDgemmStridedBatched(
        h, mkl::transpose::nontrans, mkl::transpose::nontrans, order, order,
        order, &alpha, (const double *)A, order, order * order,
        (const double *)B, order, order * order, &beta, C, order, order * order,
        batches);
    dpct::get_current_device().queues_wait_and_throw();

    //  cublasStatus_t cublasDgemmBatched(cublasHandle_t handle,
    //                                    cublasOperation_t transa,
    //                                    cublasOperation_t transb,
    //                                    int m, int n, int k,
    //                                    const double          *alpha,
    //                                    const double          *Aarray[], int lda,
    //                                    const double          *Barray[], int ldb,
    //                                    const double          *beta,
    //                                    double          *Carray[], int ldc,
    //                                    int batchCount)
}
#endif // 0

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/CUBLAS Dense matrix-matrix multiplication: C += A x B" << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int order;
  int batches = 0;
  int use_ngpu = 1;
  try {
      if (argc < 2) {
        throw "Usage: <# iterations> <matrix order> [<batches>] [<use_ngpu>]";
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
        use_ngpu = std::atoi(argv[4]);
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
  }
  else if (batches < 0) {
      std::cout << "Batch size            = " << -batches << " (loop over legacy BLAS)" << std::endl;
  }
#if 0
  else if (batches > 0) {
      std::cout << "Batch size            = " <<  batches << " (batched BLAS)" << std::endl;
  }
#endif // 0
  std::cout << "Number of GPUs to use = " << use_ngpu << std::endl;

  int haz_ngpu = dpct::dev_mgr::instance().device_count();
  std::cout << "Number of GPUs found  = " << haz_ngpu << std::endl;

  if (use_ngpu > haz_ngpu) {
      std::cout << "You cannot use more GPUs (" << use_ngpu << ") than you have (" << haz_ngpu << ")" << std::endl;
  }

  int ngpus = use_ngpu;

  std::vector<cublasHandle_t> contexts(ngpus);
  for (int i=0; i<ngpus; ++i) {
        dpct::dev_mgr::instance().select_device(i);
  }

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
  std::vector<double*> h_c(ngpus,nullptr);
  for (int i=0; i<ngpus; ++i) {
      h_c[i] = sycl::malloc_host<double>(matrices * bytes, dpct::get_default_context());
  }

  // device buffers
  std::vector<double*> d_a(ngpus,nullptr);
  std::vector<double*> d_b(ngpus,nullptr);
  std::vector<double*> d_c(ngpus,nullptr);
  for (int i=0; i<ngpus; ++i) {
        dpct::dev_mgr::instance().select_device(i);
        d_a[i] = sycl::malloc_device<double>(matrices * nelems, dpct::get_current_device(), dpct::get_default_context());
        d_b[i] = sycl::malloc_device<double>(matrices * nelems, dpct::get_current_device(), dpct::get_default_context());
        d_c[i] = sycl::malloc_device<double>(matrices * nelems, dpct::get_current_device(), dpct::get_default_context());
        dpct::get_default_queue().submit([&](sycl::handler &cgh)
        {
            auto dpct_global_range = dimGrid * dimBlock;

            auto d_a_i_ct2 = d_a[i];
            auto d_b_i_ct3 = d_b[i];
            auto d_c_i_ct4 = d_c[i];

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)),
                                  sycl::range<3>(dimBlock.get(2), dimBlock.get(1), dimBlock.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                    init(order, matrices, d_a_i_ct2, d_b_i_ct3, d_c_i_ct4, item_ct1);
                });
        });
  }
  for (int i=0; i<ngpus; ++i) {
        dpct::dev_mgr::instance().select_device(i);
        dpct::get_current_device().queues_wait_and_throw();
  }

  for (int iter = 0; iter<=iterations; iter++) {

    if (iter==1) dgemm_time = prk::wtime();

    for (int i=0; i<ngpus; ++i) {
        dpct::dev_mgr::instance().select_device(i);
        if (batches == 0) {
            prk_dgemm(contexts[i], order, matrices, d_a[i], d_b[i], d_c[i]);
        }
        else if (batches < 0) {
            prk_dgemm(contexts[i], order, matrices, d_a[i], d_b[i], d_c[i]);
        }
#if 0
        else if (batches > 0) {
            prk_bgemm(contexts[i], order, matrices, d_a[i], d_b[i], d_c[i]);
        }
#endif // 0
    }
    for (int i=0; i<ngpus; ++i) {
            dpct::dev_mgr::instance().select_device(i);
            dpct::get_current_device().queues_wait_and_throw();
    }
  }
  dgemm_time = prk::wtime() - dgemm_time;

  // copy output back to host
  for (int i=0; i<ngpus; ++i) {
        dpct::dev_mgr::instance().select_device(i);
        dpct::get_default_queue().memcpy(h_c[i], d_c[i], matrices * bytes);
  }

  for (int i=0; i<ngpus; ++i) {
        dpct::dev_mgr::instance().select_device(i);
        dpct::get_current_device().queues_wait_and_throw();
        sycl::free(d_c[i], dpct::get_default_context());
        sycl::free(d_b[i], dpct::get_default_context());
        sycl::free(d_a[i], dpct::get_default_context());
  }

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double epsilon = 1.0e-8;
  const double forder = static_cast<double>(order);
    const double reference = 0.25 * std::pow(forder, 3) *
                             std::pow(forder - 1.0, 2) * (iterations + 1);

  double residuum(0);
  for (int i=0; i<ngpus; ++i) {
      for (int b=0; b<matrices; ++b) {
          const auto checksum = prk::reduce( &(h_c[i][b*order*order+0]), &(h_c[i][b*order*order+nelems]), 0.0);
            residuum += std::abs(checksum - reference) / reference;
      }
  }
  residuum /= matrices;
  residuum /= ngpus;

  if (residuum < epsilon) {
#if VERBOSE
    std::cout << "Reference checksum = " << reference << "\n"
              << "Actual checksum = " << checksum << std::endl;
#endif
    std::cout << "Solution validates" << std::endl;
    auto avgtime = dgemm_time/iterations/matrices;
        auto nflops = 2.0 * std::pow(forder, 3) * ngpus;
    std::cout << "Rate (MF/s): " << 1.0e-6 * nflops/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
    std::cout << "Reference checksum = " << reference << "\n"
              << "Residuum           = " << residuum << std::endl;
    return 1;
  }

  for (int i=0; i<ngpus; ++i) {
        sycl::free(h_c[i], dpct::get_default_context());
  }

  return 0;
}


