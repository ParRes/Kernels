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
#include "prk_cuda.h"

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/CUBLAS STREAM triad: A = B + scalar * C" << std::endl;

  prk::CUDA::info info;
  info.print();

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

  cublasHandle_t h;
  //prk::CUDA::check( cublasInit() );
  prk::CUDA::check( cublasCreate(&h) );

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double nstream_time(0);

  const size_t bytes = length * sizeof(double);

  double * h_A;
  double * h_B;
  double * h_C;
  prk::CUDA::check( cudaMallocHost((void**)&h_A, bytes) );
  prk::CUDA::check( cudaMallocHost((void**)&h_B, bytes) );
  prk::CUDA::check( cudaMallocHost((void**)&h_C, bytes) );

  for (size_t i=0; i<length; ++i) {
    h_A[i] = 0;
    h_B[i] = 2;
    h_C[i] = 2;
  }

  double * d_A;
  double * d_B;
  double * d_C;
  prk::CUDA::check( cudaMalloc((void**)&d_A, bytes) );
  prk::CUDA::check( cudaMalloc((void**)&d_B, bytes) );
  prk::CUDA::check( cudaMalloc((void**)&d_C, bytes) );

  prk::CUDA::check( cudaMemcpy(d_A, &(h_A[0]), bytes, cudaMemcpyHostToDevice) );
  prk::CUDA::check( cudaMemcpy(d_B, &(h_B[0]), bytes, cudaMemcpyHostToDevice) );
  prk::CUDA::check( cudaMemcpy(d_C, &(h_C[0]), bytes, cudaMemcpyHostToDevice) );

  double scalar(3);
  {
    for (auto iter = 0; iter<=iterations; iter++) {

      if (iter==1) nstream_time = prk::wtime();

      double one(1);
      prk::CUDA::check( cublasDaxpy(h, length,
                                    &one,        // alpha
                                    d_B, 1,      // x, incx
                                    d_A, 1) );   // y, incy
      prk::CUDA::check( cublasDaxpy(h, length,
                                    &scalar,     // alpha
                                    d_C, 1,      // x, incx
                                    d_A, 1) );   // y, incy

      prk::CUDA::check( cudaDeviceSynchronize() );
    }
    nstream_time = prk::wtime() - nstream_time;
  }

  prk::CUDA::check( cudaMemcpy(&(h_A[0]), d_A, bytes, cudaMemcpyDeviceToHost) );

  prk::CUDA::check( cudaFree(d_C) );
  prk::CUDA::check( cudaFree(d_B) );
  prk::CUDA::check( cudaFree(d_A) );

  prk::CUDA::check( cudaFreeHost(h_B) );
  prk::CUDA::check( cudaFreeHost(h_C) );

  prk::CUDA::check( cublasDestroy(h) );
  //prk::CUDA::check( cublasShutdown() );

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

  prk::CUDA::check( cudaFreeHost(h_A) );

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


