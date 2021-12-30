///
/// Copyright (c) 2017, Intel Corporation
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
///
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
///
//////////////////////////////////////////////////////////////////////

#include "prk_sycl.h"
#include "prk_util.h"

#include "prk_dgemm_codeplay.h"

class dgemm;

void prk_dgemm(sycl::queue & q,
               size_t order,
               sycl::buffer<double> & d_A,
               sycl::buffer<double> & d_B,
               sycl::buffer<double> & d_C)
{
    q.submit([&](sycl::handler& h) {

      auto A = d_A.get_access<sycl::access::mode::read>(h);
      auto B = d_B.get_access<sycl::access::mode::read>(h);
      auto C = d_C.get_access<sycl::access::mode::read_write>(h);

      h.parallel_for<class dgemm>( sycl::range<2>{order,order}, [=] (sycl::id<2> it) {

          const int i = it[0];
          const int j = it[1];

          double Ctemp(0);
          for (size_t k = 0; k < order; ++k) {
              Ctemp += A[i*order+k] * B[k*order+j];
          }
          C[i*order+j] += Ctemp;
      });
    });
    q.wait();
}

int main(int argc, char * argv[])
{
  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/SYCL Dense matrix-matrix multiplication: C += A x B" << std::endl;

  int iterations;
  int order;
  int tile_size;
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

      tile_size = (argc>3) ? std::atoi(argv[3]) : 32;
      if (tile_size <= 0) tile_size = order;

  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  sycl::queue q(sycl::default_selector{});
  prk::SYCL::print_device_platform(q);

  if (tile_size < order) {
      // CodePlay tiled implementation is not general and requires order=2^N and tiles fit in work group
      auto max_work_group_size = q.get_device().get_info<sycl::info::device::max_work_group_size>();
      int max_tile_size = std::sqrt(max_work_group_size);
      if (tile_size > max_tile_size ) {
          std::cout << "Tile size (" << tile_size << ") is larger than allowed (" << max_tile_size << ")" << std::endl;
          //std::cout << "max_work_group_size  = " << max_work_group_size << std::endl;
          //std::cout << "max_tile_size        = " << max_tile_size << std::endl;
          tile_size = max_tile_size;
      }
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Matrix order         = " << order << std::endl;
  if (tile_size < order) {
      std::cout << "Tile size            = " << tile_size << std::endl;
      if ( order != (int)std::exp2( (int)std::log2( order )) ) {
          std::cout << "Tiled implementation requires that order be a power of 2!" << std::endl;
          std::abort();
      }
  } else {
      std::cout << "Untiled" << std::endl;
  }

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for matrices
  //////////////////////////////////////////////////////////////////////

  double dgemm_time{0};

  std::vector<double> h_A(order*order);
  std::vector<double> h_B(order*order);
  std::vector<double> h_C(order*order,0.0);
  for (int i=0; i<order; ++i) {
    for (int j=0; j<order; ++j) {
       h_A[i*order+j] = i;
       h_B[i*order+j] = i;
    }
  }

  {
    sycl::buffer<double,1> d_A { h_A.data(), sycl::range<1>(h_A.size()) };
    sycl::buffer<double,1> d_B { h_B.data(), sycl::range<1>(h_B.size()) };
    sycl::buffer<double,1> d_C { h_C.data(), sycl::range<1>(h_C.size()) };

    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) dgemm_time = prk::wtime();

      if (tile_size < order) {
          prk_dgemm<double>(q, order, tile_size, d_A, d_B, d_C);
      } else {
          prk_dgemm(q, order, d_A, d_B, d_C);
      }
    }
    dgemm_time = prk::wtime() - dgemm_time;
  }

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const auto forder = static_cast<double>(order);
  const auto reference = 0.25 * prk::pow(forder,3) * prk::pow(forder-1.0,2) * (iterations+1);
  const auto checksum = prk::reduce(h_C.begin(), h_C.end(), 0.0);

  const auto epsilon = 1.0e-8;
  const auto residuum = std::abs(checksum-reference)/reference;
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

  return 0;
}


