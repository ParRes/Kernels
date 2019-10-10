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

#include "prk_util.h"
#include "prk_opencl.h"

template <typename T>
void run(cl::Context context, int iterations, size_t length)
{
  auto precision = (sizeof(T)==8) ? 64 : 32;

  cl::Program program(context, prk::opencl::loadProgram("nstream.cl"), true);

  auto function = (precision==64) ? "nstream64" : "nstream32";

  cl_int err;
  auto kernel = cl::make_kernel<int, T, cl::Buffer, cl::Buffer, cl::Buffer>(program, function, &err);
  if(err != CL_SUCCESS){
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
  }

  cl::CommandQueue queue(context);

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for the input and nstream matrix
  //////////////////////////////////////////////////////////////////////

  auto nstream_time = 0.0;

  std::vector<T> h_a(length, T(0));
  std::vector<T> h_b(length, T(2));
  std::vector<T> h_c(length, T(2));

  // copy input from host to device
  cl::Buffer d_a = cl::Buffer(context, begin(h_a), end(h_a), false);
  cl::Buffer d_b = cl::Buffer(context, begin(h_b), end(h_b), true);
  cl::Buffer d_c = cl::Buffer(context, begin(h_c), end(h_c), true);

  double scalar = 3.0;

  for (auto iter = 0; iter<=iterations; iter++) {

    if (iter==1) nstream_time = prk::wtime();

    // nstream the  matrix
    kernel(cl::EnqueueArgs(queue, cl::NDRange(length)), length, scalar, d_a, d_b, d_c);
    queue.finish();

  }
  nstream_time = prk::wtime() - nstream_time;

  // copy output back to host
  cl::copy(queue, d_a, begin(h_a), end(h_a));

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  double ar(0);
  T br(2);
  T cr(2);
  for (auto i=0; i<=iterations; i++) {
      ar += br + scalar * cr;
  }

  ar *= length;

  double asum(0);
  for (size_t i=0; i<length; i++) {
      asum += std::fabs(h_a[i]);
  }

  const double epsilon = (precision==64) ? 1.0e-8 : 1.0e-4;
  if (std::fabs(ar-asum)/asum > epsilon) {
      std::cout << "Failed Validation on output array\n"
                << std::setprecision(16)
                << "       Expected checksum: " << ar << "\n"
                << "       Observed checksum: " << asum << std::endl;
      std::cout << "ERROR: solution did not validate" << std::endl;
  } else {
      std::cout << "Solution validates" << std::endl;
      double avgtime = nstream_time/iterations;
      double nbytes = 4.0 * length * sizeof(T);
      std::cout << precision << "B "
                << "Rate (MB/s): " << 1.e-6*nbytes/avgtime
                << " Avg time (s): " << avgtime << std::endl;
  }
}

int main(int argc, char* argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/OpenCL STREAM triad: A = B + scalar * C" << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations, offset, length;
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

  //////////////////////////////////////////////////////////////////////
  /// Setup OpenCL environment
  //////////////////////////////////////////////////////////////////////

  prk::opencl::listPlatforms();

  cl_int err = CL_SUCCESS;

  cl::Context cpu(CL_DEVICE_TYPE_CPU, NULL, NULL, NULL, &err);
  if ( err == CL_SUCCESS && prk::opencl::available(cpu) )
  {
    const int precision = prk::opencl::precision(cpu);

    std::cout << "CPU Precision        = " << precision << "-bit" << std::endl;

    if (precision==64) {
        run<double>(cpu, iterations, length);
    }
    run<float>(cpu, iterations, length);
  } else {
    std::cerr << "No CPU" << std::endl;
  }

  cl::Context gpu(CL_DEVICE_TYPE_GPU, NULL, NULL, NULL, &err);
  if ( err == CL_SUCCESS && prk::opencl::available(gpu) )
  {
    const int precision = prk::opencl::precision(gpu);

    std::cout << "GPU Precision        = " << precision << "-bit" << std::endl;

    if (precision==64) {
        run<double>(gpu, iterations, length);
    }
    run<float>(gpu, iterations, length);
  } else {
    std::cerr << "No GPU" << std::endl;
  }

  cl::Context acc(CL_DEVICE_TYPE_ACCELERATOR, NULL, NULL, NULL, &err);
  if ( err == CL_SUCCESS && prk::opencl::available(acc) )
  {

    const int precision = prk::opencl::precision(acc);

    std::cout << "ACC Precision        = " << precision << "-bit" << std::endl;

    if (precision==64) {
        run<double>(acc, iterations, length);
    }
    run<float>(acc, iterations, length);
  } else {
    std::cerr << "No ACC" << std::endl;
  }

  return 0;
}
