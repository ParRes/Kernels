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
#include "prk_opencl.hpp"

int main(int argc, char * argv[])
{
  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/OpenCL Matrix transpose: B = A^T" << std::endl;

  int iterations;
  int order;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <matrix order>";
      }

      // number of times to do the transpose
      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      // order of a the matrix
      order = std::atol(argv[2]);
      if (order <= 0) {
        throw "ERROR: Matrix Order must be greater than 0";
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Matrix order          = " << order << std::endl;
  std::cout << "Number of iterations  = " << iterations << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Setup OpenCL environment
  //////////////////////////////////////////////////////////////////////

  // FIXME: allow other options here
  cl::Context context(CL_DEVICE_TYPE_DEFAULT);

  cl::Program program(context, prk::loadProgram("transpose.cl"), true);

  cl::CommandQueue queue(context);

  auto kernel = cl::make_kernel<int, cl::Buffer, cl::Buffer>(program, "transpose");

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

  std::vector<float> h_a;
  std::vector<float> h_b;
  size_t nelems = (size_t)order * (size_t)order;
  h_a.resize(nelems);
  h_b.resize(nelems,0.0);
  // fill A with the sequence 0 to order^2-1 as floats
  std::iota(h_a.begin(), h_a.end(), 0.0);

  // copy input from host to device
  cl::Buffer d_a = cl::Buffer(context, begin(h_a), end(h_a), true);
  cl::Buffer d_b = cl::Buffer(context, begin(h_b), end(h_b), true);

  auto trans_time = 0.0;

  for (auto iter = 0; iter<=iterations; iter++) {

    if (iter==1) trans_time = prk::wtime();

    // transpose the  matrix
    kernel(cl::EnqueueArgs(queue, cl::NDRange(order,order)), order, d_a, d_b);
    queue.finish();

  }
  trans_time = prk::wtime() - trans_time;

  // copy output back to host
  cl::copy(queue, d_b, begin(h_b), end(h_b));

#ifdef VERBOSE
  // copy input back to host - debug only
  cl::copy(queue, d_a, begin(h_a), end(h_a));
#endif

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  // TODO: replace with std::generate, std::accumulate, or similar
  const auto addit = (iterations+1.) * (iterations/2.);
  auto abserr = 0.0;
  for (auto j=0; j<order; j++) {
    for (auto i=0; i<order; i++) {
      const size_t ij = (size_t)i*(size_t)order+(size_t)j;
      const size_t ji = (size_t)j*(size_t)order+(size_t)i;
      const float reference = static_cast<float>(ij)*(1.+iterations)+addit;
      abserr += std::fabs(h_b[ji] - reference);
    }
  }

#ifdef VERBOSE
  std::cout << "Sum of absolute differences: " << abserr << std::endl;
#endif

  const auto epsilon = 1.0e-8;
  if (abserr < epsilon) {
    std::cout << "Solution validates" << std::endl;
    auto avgtime = trans_time/iterations;
    auto bytes = (size_t)order * (size_t)order * sizeof(float);
    std::cout << "Rate (MB/s): " << 1.0e-6 * (2L*bytes)/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
#ifdef VERBOSE
    for (auto i=0; i<order; i++) {
      for (auto j=0; j<order; j++) {
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


