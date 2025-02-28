///
/// Copyright (c) 2020, Intel Corporation
/// Copyright (c) 2021, NVIDIA
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
#include "prk_cuda.h"
#include "prk_nccl.h"

__global__ void nstream(const unsigned n, const double scalar, double * A, const double * B, const double * C)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
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

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels" << std::endl;
  std::cout << "C++11/CUDA STREAM triad: A = B + scalar * C" << std::endl;

  prk::CUDA::info info;
  info.print();

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  size_t length, block_size=256;
  bool grid_stride = false;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <vector length> [<block_size>] [<grid_stride>]";
      }

      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      length = std::atol(argv[2]);
      if (length <= 0) {
        throw "ERROR: vector length must be positive";
      } else if (length >= UINT_MAX) {
        throw "ERROR: vector length must be less than UINT_MAX";
      }

      if (argc>3) {
         block_size = std::atoi(argv[3]);
      }

      if (argc>4) {
        grid_stride = prk::parse_boolean(std::string(argv[4]));
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Vector length        = " << length << std::endl;
  std::cout << "Block size           = " << block_size << std::endl;
  std::cout << "Grid stride          = " << (grid_stride   ? "yes" : "no") << std::endl;

  dim3 dimBlock(block_size, 1, 1);
  dim3 dimGrid(prk::divceil(length,block_size), 1, 1);

  info.checkDims(dimBlock, dimGrid);

  int num_gpus = info.num_gpus();
  std::vector<ncclComm_t> nccl_comm_world(num_gpus);
  std::cerr << "before ncclCommInitAll: " << num_gpus << " GPUs" << std::endl;
#if 1
  prk::check( ncclCommInitAll(nccl_comm_world.data(), num_gpus, nullptr) );
#else
  {
      ncclUniqueId Id;
      prk::CUDA::check( ncclGetUniqueId(&Id) );
      prk::CUDA::check( ncclGroupStart() );
      for (int i=0; i<num_gpus; i++) {
        info.set_gpu(i);
        std::cerr << "before ncclCommInitRank: " << i << std::endl;
        prk::CUDA::check( ncclCommInitRank(&nccl_comm_world[i], num_gpus, Id, i) );
        std::cerr << "after ncclCommInitRank: " << i << std::endl;
      }
      prk::CUDA::check( ncclGroupEnd() );
  }
#endif
  std::cerr << "after ncclCommInitAll" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double nstream_time(0);

  double * h_A = prk::CUDA::malloc_host<double>(length*num_gpus);
  double * h_B = prk::CUDA::malloc_host<double>(length);
  double * h_C = prk::CUDA::malloc_host<double>(length);

  for (size_t i=0; i<length; ++i) {
    h_A[i] = 0;
    h_B[i] = 2;
    h_C[i] = 2;
  }

  std::vector<double*> d_A(num_gpus,nullptr);
  std::vector<double*> d_B(num_gpus,nullptr);
  std::vector<double*> d_C(num_gpus,nullptr);

  for (int i=0; i<num_gpus; i++) {
      info.set_gpu(i);
      d_A[i] = prk::CUDA::malloc_async<double>(length);
      d_B[i] = prk::CUDA::malloc_async<double>(length);
      d_C[i] = prk::CUDA::malloc_async<double>(length);
      prk::CUDA::copyH2Dasync(d_A[i], h_A, length);
      prk::CUDA::copyH2Dasync(d_B[i], h_B, length);
      prk::CUDA::copyH2Dasync(d_C[i], h_C, length);
      prk::CUDA::sync();
  }

  double scalar(3);
  {
    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) {
          for (int i=0; i<num_gpus; i++) {
              info.set_gpu(i);
              prk::CUDA::sync();
          }
          nstream_time = prk::wtime();
      }

      for (int i=0; i<num_gpus; i++) {
          info.set_gpu(i);
          if (grid_stride) {
              nstream2<<<dimGrid, dimBlock>>>(static_cast<unsigned>(length), scalar, d_A[i], d_B[i], d_C[i]);
          } else {
              nstream<<<dimGrid, dimBlock>>>(static_cast<unsigned>(length), scalar, d_A[i], d_B[i], d_C[i]);
          }
      }
      for (int i=0; i<num_gpus; i++) {
          info.set_gpu(i);
          prk::CUDA::sync();
      }
    }
    nstream_time = prk::wtime() - nstream_time;
  }

  for (int i=0; i<num_gpus; i++) {
      info.set_gpu(i);
      prk::CUDA::copyD2H(&h_A[i*length], d_A[i], length);
      prk::CUDA::free(d_A[i]);
      prk::CUDA::free(d_B[i]);
      prk::CUDA::free(d_C[i]);
      prk::check( ncclCommFinalize(nccl_comm_world[i]) );
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
  ar *= num_gpus;

  double asum(0);
  for (int i=0; i<length*num_gpus; i++) {
      asum += prk::abs(h_A[i]);
  }

  prk::CUDA::free_host(h_A);
  prk::CUDA::free_host(h_B);
  prk::CUDA::free_host(h_C);

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
      double nbytes = 4.0 * length * sizeof(double) * num_gpus;
      std::cout << "Rate (MB/s): " << 1.e-6*nbytes/avgtime
                << " Avg time (s): " << avgtime << std::endl;
  }

  return 0;
}


