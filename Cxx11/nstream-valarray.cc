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
/// NAME:    nstream
///
/// PURPOSE: To compute memory bandwidth when adding a vector of a given
///          number of double precision values to the scalar multiple of
///          another vector of the same length, and storing the result in
///          a third vector.
///
/// USAGE:   The program takes as input the number
///          of iterations to loop over the triad vectors and
///          the length of the vectors.
///
///          <progname> <# iterations> <vector length>
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

#include "prk_util.h"
#include <valarray>

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11 STREAM triad: A = B + scalar * C" << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  size_t length;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <vector length>";
      }

      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      length = std::atol(argv[2]);
      if (length <= 0) {
        throw "ERROR: vector length must be positive";
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Vector length        = " << length << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double nstream_time{0};

  std::valarray<double> A(0.0,length);
  std::valarray<double> B(2.0,length);
  std::valarray<double> C(2.0,length);

  double scalar = 3.0;

  {
    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) nstream_time = prk::wtime();

      A += B + scalar * C;
    }
    nstream_time = prk::wtime() - nstream_time;
  }

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  double ar(0);
  double br(2);
  double cr(2);
  for (int i=0; i<=iterations; i++) {
      ar += br + scalar * cr;
  }

  ar *= length;

  double asum(0);
  for (size_t i=0; i<length; i++) {
      asum += prk::abs(A[i]);
  }

  double epsilon=1.e-8;
  if (prk::abs(ar-asum)/asum > epsilon) {
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


