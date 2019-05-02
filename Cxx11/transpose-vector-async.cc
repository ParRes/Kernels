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

#include "prk_util.h"

// These headers are busted with NVCC and GCC 5.4.0
// The <future> header is busted with Cray C++ 8.6.1.
#if !defined(__NVCC__) && !defined(_CRAYC)
#include <thread>
#include <future>
#endif

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/Threads Matrix transpose: B = A^T" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int order;
  int tile_size;
  int block_size;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <matrix order> <block size> [tile size]";
      }

      // number of times to do the transpose
      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      // order of a the matrix
      order = std::atoi(argv[2]);
      if (order <= 0) {
        throw "ERROR: Matrix Order must be greater than 0";
      }

      block_size = std::atoi(argv[3]);
      if (block_size <= 0) {
        throw "ERROR: block size must be greater than 0";
      }

      // default tile size for tiling of local transpose
      tile_size = (argc>4) ? std::atoi(argv[4]) : 32;
      // a negative tile size means no tiling of the local transpose
      if (tile_size <= 0) tile_size = block_size;
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  int num_futures = order/block_size;
  if (order % block_size) num_futures++;
  num_futures *= num_futures;

  std::cout << "Number of futures     = " << num_futures << std::endl;
  std::cout << "Number of iterations  = " << iterations << std::endl;
  std::cout << "Matrix order          = " << order << std::endl;
  std::cout << "Block size            = " << block_size << std::endl;
  std::cout << "Tile size             = " << tile_size << std::endl;

  if (num_futures > 300) {
      std::cout << "These settings may lead to resource exhaustion.\n"
                << "Please use a larger block size.\n";
      return 1;
  }

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  prk::vector<double> A(order*order);
  prk::vector<double> B(order*order,0.0);

  // fill A with the sequence 0 to order^2-1 as doubles
  std::iota(A.begin(), A.end(), 0.0);

  auto trans_time = 0.0;

  std::vector<std::future<void>> pool;

  for (auto iter = 0; iter<=iterations; iter++) {

    if (iter==1) trans_time = prk::wtime();

    for (auto ib=0; ib<order; ib+=block_size) {
      for (auto jb=0; jb<order; jb+=block_size) {
        pool.push_back(std::async(std::launch::async, [=,&A,&B] {
          for (auto it=ib; it<std::min(order,ib+block_size); it+=tile_size) {
            for (auto jt=jb; jt<std::min(order,jb+block_size); jt+=tile_size) {
              for (auto i=it; i<std::min(ib+block_size,it+tile_size); i++) {
                for (auto j=jt; j<std::min(jb+block_size,jt+tile_size); j++) {
                  B[i*order+j] += A[j*order+i];
                  A[j*order+i] += 1.0;
                }
              }
            }
          }
        } ));
      }
    }
    std::for_each(pool.begin(), pool.end(), [](std::future<void> & f) { f.wait(); });
    pool.clear();
  }
  trans_time = prk::wtime() - trans_time;

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  // TODO: replace with std::generate, std::accumulate, or similar
  const auto addit = (iterations+1.) * (iterations/2.);
  auto abserr = 0.0;
  for (auto j=0; j<order; j++) {
    for (auto i=0; i<order; i++) {
      const int ij = i*order+j;
      const int ji = j*order+i;
      const double reference = static_cast<double>(ij)*(1.+iterations)+addit;
      abserr += std::fabs(B[ji] - reference);
    }
  }

#ifdef VERBOSE
  std::cout << "Sum of absolute differences: " << abserr << std::endl;
#endif

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


