///
/// Copyright (c) 2013, Intel Corporation
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
#include "prk_cuda.h"

// The kernel was derived from https://github.com/parallel-forall/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu

const int tile_dim = 32;
const int block_rows = 8;

__global__ void transposeNoBankConflict(int order, double * A, double * B)
{
    __shared__ double tile[tile_dim][tile_dim+1];

    auto x = blockIdx.x * tile_dim + threadIdx.x;
    auto y = blockIdx.y * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j += block_rows) {
       tile[threadIdx.y+j][threadIdx.x] = A[(y+j)*order + x];
       A[(y+j)*order + x] += (double)1;
    }

    __syncthreads();

    x = blockIdx.y * tile_dim + threadIdx.x;
    y = blockIdx.x * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j+= block_rows) {
        B[(y+j)*order + x] += tile[threadIdx.x][threadIdx.y + j];
    }
}

__global__ void transposeCoalesced(int order, double * A, double * B)
{
    __shared__ double tile[tile_dim][tile_dim];

    auto x = blockIdx.x * tile_dim + threadIdx.x;
    auto y = blockIdx.y * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j += block_rows) {
       tile[threadIdx.y+j][threadIdx.x] = A[(y+j)*order + x];
       A[(y+j)*order + x] += (double)1;
    }

    __syncthreads();

    x = blockIdx.y * tile_dim + threadIdx.x;
    y = blockIdx.x * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j+= block_rows) {
        B[(y+j)*order + x] += tile[threadIdx.x][threadIdx.y + j];
    }
}

__global__ void transposeNaive(int order, double * A, double * B)
{
    auto x = blockIdx.x * tile_dim + threadIdx.x;
    auto y = blockIdx.y * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j+= block_rows) {
        B[x*order + (y+j)] += A[(y+j)*order + x];
        A[(y+j)*order + x] += (double)1;
    }
}

const std::array<std::string,3> vnames = {"naive", "coalesced", "no bank conflicts"};

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/CUDA Matrix transpose: B = A^T" << std::endl;

  prk::CUDA::info info;
  info.print();

  //////////////////////////////////////////////////////////////////////
  // Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int order, variant;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <matrix order> [variant (0/1/2)]";
      }

      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      order = std::atoi(argv[2]);
      if (order <= 0) {
        throw "ERROR: Matrix Order must be greater than 0";
      } else if (order > prk::get_max_matrix_size()) {
        throw "ERROR: matrix dimension too large - overflow risk";
      } else if (order % tile_dim) {
        throw "ERROR: matrix dimension not divisible by tile size";
      }

      variant = 2; // transposeNoBankConflicts
      if (argc > 3) {
          variant = std::atoi(argv[3]);
      }
      if (variant < 0 || variant > 2) {
          throw "Please select a valid variant (0: naive 1: coalesced, 2: no bank conflicts)";
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations  = " << iterations << std::endl;
  std::cout << "Matrix order          = " << order << std::endl;
  std::cout << "Variant               = " << vnames[variant] << std::endl;

  dim3 dimGrid(order/tile_dim, order/tile_dim, 1);
  dim3 dimBlock(tile_dim, block_rows, 1);

  info.checkDims(dimBlock, dimGrid);

  //////////////////////////////////////////////////////////////////////
  // Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

  const size_t nelems = (size_t)order * (size_t)order;

  double * h_a = prk::CUDA::malloc_host<double>(nelems);
  double * h_b = prk::CUDA::malloc_host<double>(nelems);

  // fill A with the sequence 0 to order^2-1
  for (int j=0; j<order; j++) {
    for (int i=0; i<order; i++) {
      h_a[j*order+i] = static_cast<double>(order*j+i);
      h_b[j*order+i] = static_cast<double>(0);
    }
  }

  // copy input from host to device
  double * d_a = prk::CUDA::malloc_device<double>(nelems);
  double * d_b = prk::CUDA::malloc_device<double>(nelems);

  prk::CUDA::copyH2D(d_a, h_a, nelems);
  prk::CUDA::copyH2D(d_b, h_b, nelems);

  double trans_time{0};

  for (int iter = 0; iter<=iterations; iter++) {

    if (iter==1) {
        prk::CUDA::sync();
        trans_time = prk::wtime();
    }

    if (variant==0) {
        transposeNaive<<<dimGrid, dimBlock>>>(order, d_a, d_b);
    } else if (variant==1) {
        transposeCoalesced<<<dimGrid, dimBlock>>>(order, d_a, d_b);
    } else if (variant==2) {
        transposeNoBankConflict<<<dimGrid, dimBlock>>>(order, d_a, d_b);
    }
    prk::CUDA::sync();
  }
  trans_time = prk::wtime() - trans_time;

  prk::CUDA::copyD2H(h_b, d_b, nelems);

#ifdef VERBOSE
  prk::CUDA::copyD2H(h_a, d_a, nelems);
#endif

  prk::CUDA::free(d_a);
  prk::CUDA::free(d_b);

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double addit = (iterations+1.) * (iterations/2.);
  double abserr(0);
  for (int j=0; j<order; j++) {
    for (int i=0; i<order; i++) {
      const size_t ij = (size_t)i*(size_t)order+(size_t)j;
      const size_t ji = (size_t)j*(size_t)order+(size_t)i;
      const double reference = static_cast<double>(ij)*(1.+iterations)+addit;
      abserr += prk::abs(h_b[ji] - reference);
    }
  }

#ifdef VERBOSE
  std::cout << "Sum of absolute differences: " << abserr << std::endl;
#endif

  const double epsilon = 1.0e-8;
  if (abserr < epsilon) {
    std::cout << "Solution validates" << std::endl;
    auto avgtime = trans_time/iterations;
    auto bytes = (size_t)order * (size_t)order * sizeof(double);
    std::cout << "Rate (MB/s): " << 1.0e-6 * (2L*bytes)/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
#ifdef VERBOSE
    for (int i=0; i<order; i++) {
      for (int j=0; j<order; j++) {
        std::cout << "(" << i << "," << j << ") = " << h_a[i*order+j] << ", " << h_b[i*order+j] << "\n";
      }
    }
#endif
    std::cout << "ERROR: Aggregate squared error " << abserr
              << " exceeds threshold " << epsilon << std::endl;
    return 1;
  }

  prk::CUDA::free_host(h_a);
  prk::CUDA::free_host(h_b);

  return 0;
}


