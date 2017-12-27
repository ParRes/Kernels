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
/// NAME:    dgemm
/// 
/// PURPOSE: This program tests the efficiency with which a dense matrix
///          dense multiplication is carried out
///   
/// USAGE:   The program takes as input the matrix
///          order, the number of times the matrix-matrix multiplication 
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

#include "prk_util.h"

int main(int argc, char * argv[])
{
  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11 Dense matrix-matrix multiplication: C += A x B" << std::endl;

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
      }

      tile_size = (argc>3) ? std::atoi(argv[3]) : 32;
      if (tile_size <= 0) tile_size = order;

  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Matrix order         = " << order << std::endl;
  std::cout << "Tile size            = " << tile_size << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for matrices
  //////////////////////////////////////////////////////////////////////

  double dgemm_time(0);

  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> C;
  A.resize(order*order);
  B.resize(order*order);
  C.resize(order*order,0.0);
  for (auto i=0; i<order; ++i) {
    for (auto j=0; j<order; ++j) {
       A[i*order+j] = i;
       B[i*order+j] = i;
    }
  }

  {
    for (auto iter = 0; iter<iterations; iter++) {

      if (iter==1) dgemm_time = prk::wtime();

      if (0) { //tile_size < order) {
        for (auto it=0; it<order; it+=tile_size) {
          for (auto jt=0; jt<order; jt+=tile_size) {
            for (auto i=it; i<std::min(order,it+tile_size); i++) {
              for (auto j=jt; j<std::min(order,jt+tile_size); j++) {
                B[i*order+j] += A[j*order+i];
                A[j*order+i] += 1.0;
              }
            }
          }
        }
      } else {
        for (auto i=0; i<order; ++i) {
          for (auto j=0; j<order; ++j) {
            for (auto k=0; k<order; ++k) {
                C[i*order+j] += A[i*order+k] * B[k*order+j];
            }
          }
        }
      }
    }
    dgemm_time = prk::wtime() - dgemm_time;
  }

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double forder = static_cast<double>(order);
  const double reference = 0.25 * std::pow(forder,3) * std::pow(forder-1.0,2) * iterations;

  double checksum(0);
  // TODO: replace with std::generate, std::accumulate, or similar
  for (auto i=0; i<order; i++) {
    for (auto j=0; j<order; j++) {
      checksum += C[i*order+j];
    }
  }

  const auto epsilon = 1.0e-8;
  if (std::abs(checksum-reference) < epsilon) {
    std::cout << "Solution validates" << std::endl;
    auto avgtime = dgemm_time/iterations;
    auto nflops = 2.0 * std::pow(forder,3);
    std::cout << "Rate (MB/s): " << 1.0e-6 * nflops/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else { 
    std::cout << "Reference checksum = " << reference << "\n"
              << "Actual checksum = " << checksum << std::endl;
#if VERBOSE
    for (auto i=0; i<order; ++i)
      for (auto j=0; j<order; ++j)
        std::cout << "A(" << i << "," << j << ") = " << A[i*order+j] << "\n";
    for (auto i=0; i<order; ++i)
      for (auto j=0; j<order; ++j)
        std::cout << "B(" << i << "," << j << ") = " << B[i*order+j] << "\n";
    for (auto i=0; i<order; ++i)
      for (auto j=0; j<order; ++j)
        std::cout << "C(" << i << "," << j << ") = " << C[i*order+j] << "\n";
    std::cout << std::endl;
#endif
    return 1;
  }

  return 0;
}


