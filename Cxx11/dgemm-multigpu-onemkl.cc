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

#include "prk_util.h"
#include "prk_sycl.h"

#include <mkl_blas_sycl.hpp>
#include <mkl_lapack_sycl.hpp>
#include <mkl_sycl_types.hpp>

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

void prk_dgemm(sycl::queue &q, const int order, const int batches, double *A, double *B, double *C)
{
    const double alpha = 1.0;
    const double beta  = 1.0;

    for (int b=0; b<batches; ++b) {
        double * pA = &(A[b*order*order]);
        double * pB = &(B[b*order*order]);
        double * pC = &(C[b*order*order]);
        mkl::blas::gemm(q, mkl::transpose::nontrans, // opA
                           mkl::transpose::nontrans, // opB
                           order, order, order,      // m, n, k
                           alpha,                    // alpha
                           pA, order,                // A, lda
                           pB, order,                // B, ldb
                           beta,                     // beta
                           pC, order);               // C, ldc
    }
    q.wait_and_throw();
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
      std::cout << "Batch size            = " << -batches << " (loop over BLAS)" << std::endl;
  }
  else if (batches > 0) {
      std::cout << "Batched BLAS not supported." << std::endl;
      std::abort();
      //std::cout << "Batch size            = " <<  batches << " (batched BLAS)" << std::endl;
  }
  std::cout << "Number of GPUs to use = " << use_ngpu << std::endl;

  std::vector<sycl::queue> qs;

  auto platforms = sycl::platform::get_platforms();
  for (auto & p : platforms) {
    std::cout << "Platform: " << p.get_info<sycl::info::platform::name>() << std::endl;
    auto devices = p.get_devices();
    for (auto & d : devices ) {
        std::cout << " Device: " << d.get_info<sycl::info::device::name>() << std::endl;
        if (d.is_gpu()) {
            std::cout << "Device is GPU - adding to vector of queues" << std::endl;
            qs.push_back(sycl::queue(d));
        }
    }
  }

  int haz_ngpu = qs.size();
  std::cout << "Number of GPUs found  = " << haz_ngpu << std::endl;

  if (use_ngpu > haz_ngpu) {
      std::cout << "You cannot use more GPUs (" << use_ngpu << ") than you have (" << haz_ngpu << ")" << std::endl;
  }

  int ngpus = use_ngpu;

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
      h_c[i] = sycl::malloc_host<double>(matrices * bytes, qs[i]);
  }

  // device buffers
  std::vector<double*> d_a(ngpus,nullptr);
  std::vector<double*> d_b(ngpus,nullptr);
  std::vector<double*> d_c(ngpus,nullptr);
  for (int i=0; i<ngpus; ++i) {
      auto q = qs[i];
      d_a[i] = sycl::malloc_device<double>(matrices * nelems, q);
      d_b[i] = sycl::malloc_device<double>(matrices * nelems, q);
      d_c[i] = sycl::malloc_device<double>(matrices * nelems, q);
      q.submit([&](sycl::handler &cgh)
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
      auto q = qs[i];
      q.wait_and_throw();
  }

  for (int iter = 0; iter<=iterations; iter++) {

    if (iter==1) dgemm_time = prk::wtime();

    for (int i=0; i<ngpus; ++i) {
        if (batches == 0) {
            prk_dgemm(qs[i], order, matrices, d_a[i], d_b[i], d_c[i]);
        }
        else if (batches < 0) {
            prk_dgemm(qs[i], order, matrices, d_a[i], d_b[i], d_c[i]);
        }
    }
    for (int i=0; i<ngpus; ++i) {
      auto q = qs[i];
      q.wait_and_throw();
    }
  }
  dgemm_time = prk::wtime() - dgemm_time;

  // copy output back to host
  for (int i=0; i<ngpus; ++i) {
      auto q = qs[i];
      q.memcpy(h_c[i], d_c[i], matrices * bytes);
  }

  for (int i=0; i<ngpus; ++i) {
      auto q = qs[i];
      q.wait_and_throw();
      sycl::free(d_c[i], q);
      sycl::free(d_b[i], q);
      sycl::free(d_a[i], q);
  }

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double epsilon = 1.0e-8;
  const double forder = static_cast<double>(order);
  const double reference = 0.25 * std::pow(forder, 3) * std::pow(forder - 1.0, 2) * (iterations + 1);
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
      sycl::free(h_c[i], qs[i]);
  }

  return 0;
}


