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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "prk_util.h"
#include "prk_dpct.h"

void transpose(unsigned order, double * A, double * B, sycl::nd_item<3> item_ct1)
{
    auto i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
    auto j = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) + item_ct1.get_local_id(1);

    if ((i<order) && (j<order)) {
        B[i*order+j] += A[j*order+i];
        A[j*order+i] += (double)1;
    }
}

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/DPCT Matrix transpose: B = A^T" << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int order, tile_size;
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
        } else if (order > std::floor(std::sqrt(INT_MAX))) {
        throw "ERROR: matrix dimension too large - overflow risk";
      }

      // default tile size for tiling of local transpose
      tile_size = 32;
      if (argc > 3) {
          tile_size = std::atoi(argv[3]);
          if (tile_size <= 0) tile_size = order;
          if (tile_size > order) tile_size = order;
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations  = " << iterations << std::endl;
  std::cout << "Matrix order          = " << order << std::endl;
  std::cout << "Tile size             = " << tile_size << std::endl;

  sycl::range<3> dimGrid(prk::divceil(order, tile_size), prk::divceil(order, tile_size), 1);
  sycl::range<3> dimBlock(tile_size, tile_size, 1);

  //////////////////////////////////////////////////////////////////////
  // Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

  const size_t nelems = (size_t)order * (size_t)order;
  const size_t bytes = nelems * sizeof(double);
  double * h_a = sycl::malloc_host<double>( nelems, dpct::get_default_context());
  double * h_b = sycl::malloc_host<double>( nelems, dpct::get_default_context());

  // fill A with the sequence 0 to order^2-1
  for (int j=0; j<order; j++) {
    for (int i=0; i<order; i++) {
      h_a[j*order+i] = static_cast<double>(order*j+i);
      h_b[j*order+i] = static_cast<double>(0);
    }
  }

  // copy input from host to device
  double * d_a = sycl::malloc_device<double>( nelems, dpct::get_current_device(), dpct::get_default_context());
  double * d_b = sycl::malloc_device<double>( nelems, dpct::get_current_device(), dpct::get_default_context());
  dpct::get_default_queue().memcpy(d_a, &(h_a[0]), bytes).wait();
  dpct::get_default_queue().memcpy(d_b, &(h_b[0]), bytes).wait();

  auto trans_time = 0.0;

  for (int iter = 0; iter<=iterations; iter++) {

    if (iter==1) trans_time = prk::wtime();

        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            auto dpct_global_range = dimGrid * dimBlock;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                 dpct_global_range.get(1),
                                                 dpct_global_range.get(0)),
                                  sycl::range<3>(dimBlock.get(2),
                                                 dimBlock.get(1),
                                                 dimBlock.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                    transpose(order, d_a, d_b, item_ct1);
                });
        });
        dpct::get_current_device().queues_wait_and_throw();
  }
  trans_time = prk::wtime() - trans_time;

  // copy output back to host
  dpct::get_default_queue().memcpy(&(h_b[0]), d_b, bytes).wait();

#ifdef VERBOSE
  // copy input back to host - debug only
  cudaMemcpy(&(h_a[0]), d_a, bytes, cudaMemcpyDeviceToHost);
#endif

  sycl::free(d_b, dpct::get_default_context());
  sycl::free(d_a, dpct::get_default_context());

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
            abserr += std::fabs(h_b[ji] - reference);
    }
  }

#ifdef VERBOSE
  std::cout << "Sum of absolute differences: " << abserr << std::endl;
#endif

  sycl::free(h_b, dpct::get_default_context();
  sycl::free(h_a, dpct::get_default_context();

  const auto epsilon = 1.0e-8;
  if (abserr < epsilon) {
    std::cout << "Solution validates" << std::endl;
    auto avgtime = trans_time/iterations;
    auto bytes = (size_t)order * (size_t)order * sizeof(double);
    std::cout << "Rate (MB/s): " << 1.0e-6 * (2L*bytes)/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
#ifdef VERBOSE
    for (int i=0; i<order; i++) {
      for (int j=0; j<order; j++) {
        std::cout << "(" << i << "," << j << ") = " << h_a[i*order+j] << ", " << h_b[i*order+j] << "\n";
      }
    }
#endif
    std::cout << "ERROR: Aggregate squared error " << abserr
              << " exceeds threshold " << epsilon << std::endl;
    return 1;
  }

  return 0;
}


