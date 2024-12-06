///
/// Copyright (c) 2017, Intel Corporation
/// Copyright (c) 2024, NVIDIA
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
///          of iterations to loop over the triad vectors and the length
///          of the vectors.
///
///          <progname> <# iterations> <vector length> ...
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

__global__ void nstream(const unsigned n, const double scalar, double * A, const double * B, const double * C)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[i] += B[i] + scalar * C[i];
    }
}

__global__ void nstream2(const unsigned n, const double scalar, double * A, const double * B, const double * C)
{
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        A[i] += B[i] + scalar * C[i];
    }
}

__global__ void fault_pages(const unsigned n, double * A, double * B, double * C)
{
    //const unsigned inc = 4096/sizeof(double);
    //for (unsigned int i = 0; i < n; i += inc) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        A[i] = (double)0;
        B[i] = (double)2;
        C[i] = (double)2;
    }
}

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/CUDA STREAM triad: A = B + scalar * C" << std::endl;

  prk::CUDA::info info;
  //info.print();

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int length;
  bool system_memory,  grid_stride, ordered_fault, prefetch;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <vector length> [<use_system_memory> <grid_stride> <ordered_fault> <prefetch>]";
      }

      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      length = std::atoi(argv[2]);
      if (length <= 0) {
        throw "ERROR: vector length must be positive";
      }

      system_memory = (argc>3) ? prk::parse_boolean(std::string(argv[3])) : false;
      grid_stride   = (argc>4) ? prk::parse_boolean(std::string(argv[4])) : false;
      ordered_fault = (argc>5) ? prk::parse_boolean(std::string(argv[5])) : false;
      prefetch      = (argc>6) ? prk::parse_boolean(std::string(argv[6])) : false;
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Vector length        = " << length << std::endl;
  std::cout << "Memory allocator     = " << (system_memory ? "system (malloc)" : "cudaMallocManaged") << std::endl;
  std::cout << "Grid stride          = " << (grid_stride   ? "yes" : "no") << std::endl;
  std::cout << "Ordered fault        = " << (ordered_fault ? "yes" : "no") << std::endl;
  std::cout << "Prefetch             = " << (prefetch ? "yes" : "no") << std::endl;

  const int blockSize = 256;
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid(prk::divceil(length,blockSize), 1, 1);

  info.checkDims(dimBlock, dimGrid);

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double nstream_time(0);

  double * A;
  double * B;
  double * C;

  if (system_memory) {
      A = new double[length];
      B = new double[length];
      C = new double[length];
  } else {
      A = prk::CUDA::malloc_managed<double>(length);
      B = prk::CUDA::malloc_managed<double>(length);
      C = prk::CUDA::malloc_managed<double>(length);
  }

  // initialize on CPU to ensure pages are faulted there
  for (int i=0; i<length; ++i) {
    A[i] = static_cast<double>(0);
    B[i] = static_cast<double>(2);
    C[i] = static_cast<double>(2);
  }

  if (ordered_fault) {
      fault_pages<<<1,1>>>(static_cast<unsigned>(length), A, B, C);
      prk::CUDA::sync();
  }

  if (prefetch) {
      prk::CUDA::prefetch(A, length);
      prk::CUDA::prefetch(B, length);
      prk::CUDA::prefetch(C, length);
  }

  double scalar(3);
  {
    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) {
          prk::CUDA::sync();
          nstream_time = prk::wtime();
      }

      if (grid_stride) {
          nstream2<<<dimGrid, dimBlock>>>(static_cast<unsigned>(length), scalar, A, B, C);
      } else {
          nstream<<<dimGrid, dimBlock>>>(static_cast<unsigned>(length), scalar, A, B, C);
      }
      prk::CUDA::sync();
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
  for (int i=0; i<length; i++) {
      asum += prk::abs(A[i]);
  }

  if (system_memory) {
      free(A);
      free(B);
      free(C);
  } else {
      prk::CUDA::free(A);
      prk::CUDA::free(B);
      prk::CUDA::free(C);
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


