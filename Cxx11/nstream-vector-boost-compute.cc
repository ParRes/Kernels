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
///
/// HISTORY: This code is loosely based on the Stream benchmark by John
///          McCalpin, but does not follow all the Stream rules. Hence,
///          reported results should not be associated with Stream in
///          external publications
///
///          Converted to C++11 by Jeff Hammond, November 2017.
///
//////////////////////////////////////////////////////////////////////

#include "boost/compute.hpp"

#include "prk_util.h"

namespace compute = boost::compute;

using boost::compute::_1;

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/Boost.Compute STREAM triad: A = B + scalar * C" << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations, offset;
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

      offset = (argc>3) ? std::atoi(argv[3]) : 0;
      if (length <= 0) {
        throw "ERROR: offset must be nonnegative";
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  auto device = compute::system::default_device();

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Vector length        = " << length << std::endl;
  std::cout << "Offset               = " << offset << std::endl;
  std::cout << "Boost.Compute device = " << device.name() << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  auto nstream_time = 0.0;

  std::vector<float> h_A(length);

  const float scalar(3);

  compute::context context(device);
  compute::command_queue queue(context, device);
  {
    compute::vector<float> d_A(length, context);
    compute::vector<float> d_B(length, context);
    compute::vector<float> d_C(length, context);

    compute::fill(d_A.begin(), d_A.end(), 0, queue);
    compute::fill(d_B.begin(), d_B.end(), 2, queue);
    compute::fill(d_C.begin(), d_C.end(), 2, queue);
    queue.finish();

    for (auto iter = 0; iter<=iterations; iter++) {

      if (iter==1) nstream_time = prk::wtime();

      // Aout and Ain are necessary because A += .. does not work
      auto Aout = compute::lambda::get<0>(boost::compute::_1);
      auto Ain  = compute::lambda::get<1>(boost::compute::_1);
      auto B    = compute::lambda::get<2>(boost::compute::_1);
      auto C    = compute::lambda::get<3>(boost::compute::_1);

      auto begin = compute::make_zip_iterator( boost::make_tuple( d_A.begin(), d_A.begin(), d_B.begin(), d_C.begin()));
      auto end   = compute::make_zip_iterator( boost::make_tuple( d_A.end(),   d_A.end(),   d_B.end(),   d_C.end()));

      compute::for_each(begin, end,
                        compute::lambda::make_tuple
                        (
                            Aout = Ain + B + scalar * C
                        ),
                        queue
                       );
      queue.finish();
    }

    nstream_time = prk::wtime() - nstream_time;

    compute::copy(d_A.begin(), d_A.end(), h_A.begin(), queue);
    queue.finish();
  }
  compute::system::finish();

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  double ar(0);
  double br(2);
  double cr(2);
  for (auto i=0; i<=iterations; i++) {
      ar += br + scalar * cr;
  }

  ar *= length;

  double asum(0);
  for (size_t i=0; i<length; i++) {
      asum += std::fabs(h_A[i]);
  }

  double epsilon(1.e-8);
  if (std::fabs(ar-asum)/asum > epsilon) {
      std::cout << "Failed Validation on output array\n"
                << "       Expected checksum: " << ar << "\n"
                << "       Observed checksum: " << asum << std::endl;
      std::cout << "ERROR: solution did not validate" << std::endl;
      return 1;
  } else {
      std::cout << "Solution validates" << std::endl;
      double avgtime = nstream_time/iterations;
      double nbytes = 4.0 * length * sizeof(float);
      std::cout << "Rate (MB/s): " << 1.e-6*nbytes/avgtime
                << " Avg time (s): " << avgtime << std::endl;
  }

  return 0;
}


