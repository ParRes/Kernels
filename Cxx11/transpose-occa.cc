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
///          Converted to C++11 by Jeff Hammond, January 2018.
///
//////////////////////////////////////////////////////////////////////

#include "occa.hpp"

#include "prk_util.h"

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/OCCA Matrix transpose: B = A^T" << std::endl;

  char* dc = std::getenv("OCCA_DEVICE");
  if (dc==NULL) {
      std::cout << "By default, OCCA executes in serial.\n";
      std::cout << "Set OCCA_DEVICE as follows for parallel execution\n";
      std::cout << " OCCA_DEVICE=\"mode = OpenMP\"\n";
      std::cout << " OCCA_DEVICE=\"mode = OpenCL, platformID = 0, deviceID = 0\" (CPU)\n";
      std::cout << " OCCA_DEVICE=\"mode = OpenCL, platformID = 1, deviceID = 0\" (GPU)\n";
      std::cout << " OCCA_DEVICE=\"mode = CUDA', deviceID = 0\"\n";
  }
  std::string ds = (dc==NULL) ? "mode = Serial" : dc;
  occa::device device(ds);

  //////////////////////////////////////////////////////////////////////
  // Read and test input parameters
  //////////////////////////////////////////////////////////////////////

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
      } else if (order > std::floor(std::sqrt(INT_MAX))) {
        throw "ERROR: matrix dimension too large - overflow risk";
      }

      // default tile size for tiling of local transpose
      tile_size = (argc>3) ? std::atoi(argv[3]) : 32;
      // a negative tile size means no tiling of the local transpose
      if (tile_size <= 0) tile_size = order;
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Matrix order         = " << order << std::endl;
  std::cout << "Tile size            = " << tile_size << std::endl;
  std::cout << "OCCA mode            = " << "\"" << ds << "\"" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  auto trans_time = 0.0;

  double * h_A = new double[order*order];
  double * h_B = new double[order*order];
  for (auto i=0;i<order; i++) {
    for (auto j=0;j<order;j++) {
      h_A[i*order+j] = static_cast<double>(i*order+j);
      h_B[i*order+j] = 0.0;
    }
  }

  occa::memory d_A = device.malloc(order * order * sizeof(double), h_A);
  occa::memory d_B = device.malloc(order * order * sizeof(double), h_B);

  d_A.copyFrom(h_A);
  d_B.copyFrom(h_B);

  occa::kernel transpose = device.buildKernel("transpose.okl", "transpose");

  {
    for (auto iter = 0; iter<=iterations; iter++) {
      if (iter==1) trans_time = prk::wtime();
      transpose(order, d_A, d_B);
      device.finish();
    }
    trans_time = prk::wtime() - trans_time;
  }

  d_B.copyTo(h_B);

  d_A.free();
  d_B.free();
  transpose.free();
  device.free();

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const auto addit = (iterations+1.) * (iterations/2.);
  auto abserr = 0.0;
  for (auto j=0; j<order; j++) {
    for (auto i=0; i<order; i++) {
      const int ij = i*order+j;
      const int ji = j*order+i;
      const double reference = static_cast<double>(ij)*(1.+iterations)+addit;
      abserr += std::fabs(h_B[ji] - reference);
    }
  }

  delete[] h_A;
  delete[] h_B;

#ifdef VERBOSE
  std::cout << "Sum of absolute differences: " << abserr << std::endl;
#endif

  const auto epsilon = 1.0e-8;
  if (abserr < epsilon) {
    std::cout << "Solution validates" << std::endl;
    auto avgtime = trans_time/iterations;
    auto bytes = order * order * sizeof(double);
    std::cout << "Rate (MB/s): " << 1.0e-6 * (2L*bytes)/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
    std::cout << "ERROR: Aggregate squared error " << abserr
              << " exceeds threshold " << epsilon << std::endl;
    return 1;
  }

  return 0;
}


