///
/// Copyright (c) 2018, Intel Corporation
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
/// NAME:    dgemm
///
/// PURPOSE: This program tests the efficiency with which a dense matrix
///          dense multiplication is carried out
///
/// USAGE:   The program takes as input the matrix order,
///          the number of times the matrix-matrix multiplication
///          is carried out, and, optionally, a tile size for matrix
///          blocking
///
///          <progname> <# iterations> <matrix order>
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// FUNCTIONS CALLED:
///
///          Other than OpenMP or standard C functions, the following
///          functions are used in this program:
///
///          cublasDgemm()
///
/// HISTORY: Written by Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, December, 2017.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_cuda.h"
#include "prk_mpi.h"

__global__ void init(int order, double * A, double * B, double * C)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i<order) && (j<order)) {
      A[i*order+j] = i;
      B[i*order+j] = i;
      C[i*order+j] = 0;
    }
}

__global__ void init(int order, double * C)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i<order) && (j<order)) {
      C[i*order+j] = 0;
    }
}

int main(int argc, char * argv[])
{
  {
    prk::MPI::state mpi(&argc,&argv);

    int np = prk::MPI::size();
    int me = prk::MPI::rank();

    prk::CUDA::info cuda;

    if (me == 0) {
      std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
      std::cout << "MPI/C++11/CUBLAS Dense matrix-matrix multiplication: C += A x B" << std::endl;
      //cuda.print();
    }

    int ngpu = cuda.num_gpus();

    if (ngpu != np) {
        std::cout << "Please run with one MPI process per GPU (single-node only)" << std::endl;
        return (np-ngpu);
    }

    // assign a GPU per MPI process
    cuda.set_gpu(me);

    //////////////////////////////////////////////////////////////////////
    /// Read and test input parameters
    //////////////////////////////////////////////////////////////////////

    int iterations;
    int order;
    try {
        if (argc < 2) {
          throw "Usage: <# iterations> <matrix order>";
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
        }
    }
    catch (const char * e) {
      std::cout << e << std::endl;
      return 1;
    }

    if (me == 0) {
      std::cout << "Number of iterations = " << iterations << std::endl;
      std::cout << "Matrix order         = " << order << std::endl;
    }

    cublasHandle_t h;
    prk::CUDA::check( cublasCreate(&h) );

    const int tile_size = 32;
    dim3 dimGrid(prk::divceil(order,tile_size),prk::divceil(order,tile_size),1);
    dim3 dimBlock(tile_size, tile_size, 1);

    cuda.checkDims(dimBlock, dimGrid);

    //////////////////////////////////////////////////////////////////////
    // Allocate space for matrices
    //////////////////////////////////////////////////////////////////////

    double dgemm_time{0};

    const size_t nelems = (size_t)order * (size_t)order;
    const size_t bytes = nelems * sizeof(double);

    // host buffers
    double * h_c;
    prk::CUDA::check( cudaMallocHost((void**)&h_c, bytes) );

    // device buffers
    double * d_a;
    double * d_b;
    double * d_c;
    d_a = prk::CUDA::malloc_device<double>(order*order);
    d_b = prk::CUDA::malloc_device<double>(order*order);
    prk::CUDA::check( cudaMalloc((void**)&d_c, bytes) );

    init<<<dimGrid, dimBlock>>>(order, d_a, d_b, d_c);

    {
      for (int iter = 0; iter<=iterations; iter++) {

        if (iter==1) {
            prk::MPI::barrier();
            dgemm_time = prk::wtime();
        }

        double alpha = 1.0;
        double beta  = 1.0;
        prk::CUDA::check( cublasDgemm(h,
                                      CUBLAS_OP_N, CUBLAS_OP_N, // opA, opB
                                      order, order, order,      // m, n, k
                                      &alpha,                   // alpha
                                      d_a, order,               // A, lda
                                      d_b, order,               // B, ldb
                                      &beta,                    // beta
                                      d_c, order) );            // C, ldc

        prk::CUDA::check( cudaDeviceSynchronize() );
      }
      prk::MPI::barrier();
      dgemm_time = prk::wtime() - dgemm_time;
    }

    // copy output back to host
    prk::CUDA::check( cudaMemcpyAsync(&(h_c[0]), d_c, bytes, cudaMemcpyDeviceToHost) );

    prk::CUDA::check( cudaFree(d_c) );
    prk::CUDA::check( cudaFree(d_b) );
    prk::CUDA::check( cudaFree(d_a) );

    prk::CUDA::check( cublasDestroy(h) );

    prk::CUDA::check( cudaDeviceSynchronize() );

    //////////////////////////////////////////////////////////////////////
    /// Analyze and output results
    //////////////////////////////////////////////////////////////////////

    const double epsilon = 1.0e-8;
    const double forder = static_cast<double>(order);
    const double reference = 0.25 * prk::pow(forder,3) * prk::pow(forder-1.0,2) * (iterations+1);
    double residuum{0};
    const auto checksum = prk::reduce( &(h_c[0]), &(h_c[nelems]), 0.0);
    residuum += std::abs(checksum-reference)/reference;

    // take the global max to make sure everyone passes...
    residuum = prk::MPI::max(residuum);

#ifndef VERBOSE
    if (residuum >= epsilon)
#endif
    {
      for (int r=0; r<np; ++r) {
        prk::MPI::barrier();
        if (r==me) {
          std::cout << "Reference checksum = " << reference << "\n"
                    << "Actual checksum = " << residuum << std::endl;
        }
      }
    }

    if (residuum < epsilon) {
      prk::MPI::barrier();
      if (me==0) {
        std::cout << "Solution validates" << std::endl;
      }
      auto time = dgemm_time/iterations;
      auto nflops = 2.0 * prk::pow(forder,3);
      auto rate = 1.0e-6 * nflops/time;

      double minrate = prk::MPI::min(rate);
      double maxrate = prk::MPI::max(rate);
      double avgrate = prk::MPI::avg(rate);

      double mintime = prk::MPI::min(time);
      double maxtime = prk::MPI::max(time);
      double avgtime = prk::MPI::avg(time);

      if (me==0) {
        std::cout << "MIN Rate (MF/s): " << minrate << " Avg time (s): " << maxtime << std::endl;
        std::cout << "MAX Rate (MF/s): " << maxrate << " Avg time (s): " << mintime << std::endl;
        std::cout << "AVG Rate (MF/s): " << avgrate << " Avg time (s): " << avgtime << std::endl;
      }
    }

    prk::CUDA::check( cudaFreeHost(h_c) );

  } // prk::MPI:state goes out of scope here

  return 0;
}


