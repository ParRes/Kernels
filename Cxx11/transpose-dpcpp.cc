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
/// NAME:    transpose
///
/// PURPOSE: This program measures the time for the transpose of a
///          column-major stored matrix into a row-major stored matrix.
///
/// USAGE:   Program input is the matrix order and the number of times to
///          repeat the operation:
///
///          transpose <matrix_size> <# iterations>
///
///          The output consists of diagnostics to make sure the
///          transpose worked and timing statistics.
///
/// HISTORY: Written by  Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, February 2016 and May 2017.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_sycl.h"

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/DPCT Matrix transpose: B = A^T" << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  size_t order;
  try {
      if (argc < 3) {
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

  std::cout << "Number of iterations  = " << iterations << std::endl;
  std::cout << "Matrix order          = " << order << std::endl;

  sycl::queue q(sycl::default_selector{});
  prk::SYCL::print_device_platform(q);

  //////////////////////////////////////////////////////////////////////
  // Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

  const size_t nelems = (size_t)order * (size_t)order;
  const size_t bytes = nelems * sizeof(double);
  double * h_a = syclx::malloc_host<double>( nelems, q);
  double * h_b = syclx::malloc_host<double>( nelems, q);

  // fill A with the sequence 0 to order^2-1
  for (int j=0; j<order; j++) {
    for (int i=0; i<order; i++) {
      h_a[j*order+i] = static_cast<double>(order*j+i);
      h_b[j*order+i] = static_cast<double>(0);
    }
  }

  // copy input from host to device
  double * A = syclx::malloc_device<double>( nelems, q);
  double * B = syclx::malloc_device<double>( nelems, q);
  q.memcpy(A, &(h_a[0]), bytes).wait();
  q.memcpy(B, &(h_b[0]), bytes).wait();

  auto trans_time = 0.0;

  for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) trans_time = prk::wtime();

      q.submit([&](sycl::handler& h) {

        h.parallel_for( sycl::range<2>{order,order}, [=] (sycl::id<2> it) {
#if USE_2D_INDEXING
          sycl::id<2> ij{it[0],it[1]};
          sycl::id<2> ji{it[1],it[0]};
          B[ij] += A[ji];
          A[ji] += (T)1;
#else
          B[it[0] * order + it[1]] += A[it[1] * order + it[0]];
          A[it[1] * order + it[0]] += 1.0;
#endif
        });
      });
      q.wait();
  }
  trans_time = prk::wtime() - trans_time;

  // copy output back to host
  q.memcpy(&(h_b[0]), B, bytes).wait();

  syclx::free(B, q);
  syclx::free(A, q);

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double addit = (iterations+1.) * (iterations/2.);
  double abserr(0);
  for (int j=0; j<order; j++) {
    for (int i=0; i<order; i++) {
      const size_t ij = (size_t)i*(size_t)order+(size_t)j;
      const size_t ji = (size_t)j*(size_t)order+(size_t)i;
      const double reference = static_cast<double>(ij)*(1.+iterations)+addit;
      abserr += prk::abs(h_b[ji] - reference);
    }
  }

  syclx::free(h_b, q);
  syclx::free(h_a, q);

  const auto epsilon = 1.0e-8;
  if (abserr < epsilon) {
    std::cout << "Solution validates" << std::endl;
    auto avgtime = trans_time/iterations;
    auto bytes = (size_t)order * (size_t)order * sizeof(double);
    std::cout << "Rate (MB/s): " << 1.0e-6 * (2L*bytes)/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
    std::cout << "ERROR: Aggregate squared error " << abserr
              << " exceeds threshold " << epsilon << std::endl;
    return 1;
  }

  return 0;
}


