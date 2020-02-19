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
/// NAME:    nstream
///
/// PURPOSE: To compute memory bandwidth when adding a vector of a given
///          number of double precision values to the scalar multiple of
///          another vector of the same length, and storing the result in
///          a third vector.
///
/// USAGE:   The program takes as input the number
///          of iterations to loop over the triad vectors, the length of the
///          vectors, and the offset between vectors
///
///          <progname> <# iterations> <vector length> <offset>
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// NOTES:   Bandwidth is determined as the number of words read, plus the
///          number of words written, times the size of the words, divided
///          by the execution time. For a vector length of N, the total
///          number of words read and written is 4*N*sizeof(double).
///
/// HISTORY: This code is loosely based on the Stream benchmark by John
///          McCalpin, but does not follow all the Stream rules. Hence,
///          reported results should not be associated with Stream in
///          external publications
///
///          Converted to C++11 by Jeff Hammond, November 2017.
///
//////////////////////////////////////////////////////////////////////

#include "occa.hpp"

#include "prk_util.h"

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/OCCA STREAM triad: A = B + scalar * C" << std::endl;

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
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations, offset;
  int length;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <vector length> [<offset>]";
      }

      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      length = std::atoi(argv[2]);
      if (length <= 0) {
        throw "ERROR: vector length must be positive";
      }

      offset = (argc>3) ? std::atoi(argv[3]) : 0;
      if (length <= 0) {
        throw "ERROR: offset must be nonnegative";
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Vector length        = " << length << std::endl;
  std::cout << "Offset               = " << offset << std::endl;
  std::cout << "OCCA mode            = " << "\"" << ds << "\"" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  auto nstream_time = 0.0;

  double * h_A = new double[length];
  double * h_B = new double[length];
  double * h_C = new double[length];
  for (size_t i=0; i<length; ++i) {
      h_A[i] = 0.0;
      h_B[i] = 2.0;
      h_C[i] = 2.0;
  }

  double scalar(3);

  occa::memory d_A = device.malloc(length * sizeof(double), h_A);
  occa::memory d_B = device.malloc(length * sizeof(double), h_B);
  occa::memory d_C = device.malloc(length * sizeof(double), h_C);

  d_A.copyFrom(h_A);
  d_B.copyFrom(h_B);
  d_C.copyFrom(h_C);

  occa::kernel nstream = device.buildKernel("nstream.okl", "nstream");

  {
    for (auto iter = 0; iter<=iterations; iter++) {
      if (iter==1) nstream_time = prk::wtime();
      nstream(length, scalar, d_A, d_B, d_C);
      device.finish();
    }
    nstream_time = prk::wtime() - nstream_time;
  }

  d_A.copyTo(h_A);

  d_A.free();
  d_B.free();
  d_C.free();
  nstream.free();
  device.free();

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  double ar(0);
  double br(2);
  double cr(2);
  double ref(0);
  for (auto i=0; i<=iterations; i++) {
      ar += br + scalar * cr;
  }

  ar *= length;

  double asum(0);
  for (auto i=0; i<length; i++) {
      asum += std::fabs(h_A[i]);
  }

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  double epsilon=1.e-8;
  if (std::fabs(ar-asum)/asum > epsilon) {
      std::cout << "Failed Validation on output array\n"
                << std::setprecision(16)
                << "       Expected checksum: " << ar << "\n"
                << "       Observed checksum: " << asum << std::endl;
      std::cout << "ERROR: solution did not validate" << std::endl;
      return 1;
  } else {
      std::cout << "Solution validates" << std::endl;
      double avgtime = nstream_time/iterations;
      double nbytes = 4.0 * length * sizeof(double);
      std::cout << "Rate (MB/s): " << 1.e-6*nbytes/avgtime
                << " Avg time (s): " << avgtime << std::endl;
  }

  return 0;
}


