///
/// Copyright (c) 2020, Intel Corporation
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
#endif

using namespace oneapi; // oneapi::mkl -> mkl

void prk_dgemm(sycl::queue &q, const int order, const int batches, prk_float *A, prk_float *B, prk_float *C)
{
    const prk_float alpha = 1.0;
    const prk_float beta  = 1.0;

    for (int b=0; b<batches; ++b) {
        prk_float * pA = &(A[b*order*order]);
        prk_float * pB = &(B[b*order*order]);
        prk_float * pC = &(C[b*order*order]);
        mkl::blas::gemm(q, mkl::transpose::nontrans, // opA
                           mkl::transpose::nontrans, // opB
                           order, order, order,      // m, n, k
                           alpha,                    // alpha
                           pA, order,                // A, lda
                           pB, order,                // B, ldb
                           beta,                     // beta
                           pC, order);               // C, ldc
    }
    q.wait();
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
      } else if (order > prk::get_max_matrix_size()) {
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
#if 0
      std::cout << "Batch size            = " <<  batches << " (batched BLAS)" << std::endl;
#else
      std::cout << "Batched BLAS not supported." << std::endl;
      std::abort();
#endif
  }
  std::cout << "Number of GPUs to use = " << use_ngpu << std::endl;

  std::vector<sycl::queue> qs;

  auto platforms = sycl::platform::get_platforms();
  for (auto & p : platforms) {
    auto pname = p.get_info<sycl::info::platform::name>();
    std::cout << "Platform: " << pname << std::endl;
    if ( pname.find("Level-Zero") != std::string::npos) {
        std::cout << "Level Zero GPU skipped" << std::endl;
        break;
    }
    auto devices = p.get_devices();
    for (auto & d : devices ) {
        std::cout << " Device: " << d.get_info<sycl::info::device::name>() << std::endl;
        if ( d.is_gpu() ) {
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

  //////////////////////////////////////////////////////////////////////
  // Allocate space for matrices
  //////////////////////////////////////////////////////////////////////

  double dgemm_time{0};

  const int matrices = (batches == 0 ? 1 : abs(batches));
  const size_t nelems = (size_t)order * (size_t)order;
  const size_t bytes = nelems * sizeof(prk_float);

  // host buffers
  std::vector<prk_float*> h_c(ngpus,nullptr);
  for (int i=0; i<ngpus; ++i) {
      h_c[i] = sycl::malloc_host<prk_float>(matrices * bytes, qs[i]);
  }

  // device buffers
  std::vector<prk_float*> d_a(ngpus, nullptr);
  std::vector<prk_float*> d_b(ngpus, nullptr);
  std::vector<prk_float*> d_c(ngpus, nullptr);
  for (int i=0; i<ngpus; ++i) {
      auto q = qs[i];
      prk_float * p_a = d_a[i] = sycl::malloc_device<prk_float>(matrices * nelems, q);
      prk_float * p_b = d_b[i] = sycl::malloc_device<prk_float>(matrices * nelems, q);
      prk_float * p_c = d_c[i] = sycl::malloc_device<prk_float>(matrices * nelems, q);
      q.submit([&](sycl::handler &cgh)
      {
          cgh.parallel_for( sycl::range<2>{(size_t)order,(size_t)order},
                            [=] (sycl::id<2> it) {
            auto i = it[0];
            auto j = it[1];
            for (int b=0; b<matrices; ++b) {
                p_a[b*order*order+i*order+j] = i;
                p_b[b*order*order+i*order+j] = i;
                p_c[b*order*order+i*order+j] = 0;
            }
          });
      });
  }
  for (int i=0; i<ngpus; ++i) {
      auto q = qs[i];
      q.wait();
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
      q.wait();
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
      q.wait();
      sycl::free(d_c[i], q);
      sycl::free(d_b[i], q);
      sycl::free(d_a[i], q);
  }

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double epsilon = 1.0e-8;
  const double forder = static_cast<double>(order);
  const double reference = 0.25 * prk::pow(forder,3) * prk::pow(forder-1.0,2) * (iterations+1);
  double residuum(0);
  for (int i=0; i<ngpus; ++i) {
      for (int b=0; b<matrices; ++b) {
          const double checksum = prk::reduce( &(h_c[i][b*order*order+0]), &(h_c[i][b*order*order+nelems]), (prk_float)0);
          residuum += prk::abs(checksum - reference) / reference;
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
    auto nflops = 2.0 * prk::pow(forder, 3) * ngpus;
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


