///
/// Copyright (c) 2020, Intel Corporation
/// Copyright (c) 2025, NVIDIA
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
#include "prk_nvshmem.h"
#include "prk_cuda.h"
#include "transpose-kernel.h"

const std::array<std::string,3> vnames = {"naive", "coalesced", "no bank conflicts"};

int main(int argc, char * argv[])
{
  {
    prk::NVSHMEM::state nvshmem(&argc,&argv);

    int np = prk::NVSHMEM::size();
    int me = prk::NVSHMEM::rank();

    prk::CUDA::info info;
    //if (me == 0) info.print();

    //////////////////////////////////////////////////////////////////////
    /// Read and test input parameters
    //////////////////////////////////////////////////////////////////////

    int iterations = -1, variant = -1;
    size_t order = 0, block_order = 0;
    bool on_device = true;

    if (me == 0) {
      std::cout << "Parallel Research Kernels" << std::endl;
      std::cout << "C++11/NVSHMEM Matrix transpose: B = A^T" << std::endl;
    }

    // do this on every PE to avoid needing a host broadcast
    {
      try {
        if (argc < 3) {
          throw "Usage: <# iterations> <matrix order> [variant (0-2)] [on_device]";
        }

        iterations  = std::atoi(argv[1]);
        if (iterations < 1) {
          throw "ERROR: iterations must be >= 1";
        }

        order = std::atol(argv[2]);
        if (order <= 0) {
          throw "ERROR: Matrix Order must be greater than 0";
        }
        else if (order % np != 0) {
          throw "ERROR: Matrix order must be an integer multiple of the number of MPI processes";
        }

        variant = 2; // transposeNoBankConflicts
        if (argc > 3) {
            variant = std::atoi(argv[3]);
        }
        if (variant < 0 || variant > 2) {
            throw "Please select a valid variant (0: naive 1: coalesced, 2: no bank conflicts)";
        }

        block_order = order / np;
        if (block_order % tile_dim) {
          throw "ERROR: Block Order must be an integer multiple of the tile dimension (32)";
        }

        if (argc > 4) {
            on_device = (0 != std::atoi(argv[4]));
        }
      }
      catch (const char * e) {
        std::cout << e << std::endl;
        prk::NVSHMEM::abort(1);
        return 1;
      }
    }
     
    if (me == 0) {
      std::cout << "Number of PEs        = " << np << std::endl;
      std::cout << "Number of iterations = " << iterations << std::endl;
      std::cout << "Matrix order         = " << order << std::endl;
      std::cout << "Variant              = " << vnames[variant] << std::endl;
      std::cout << "Device-initiated     = " << (on_device ? "true" : "false") << std::endl;
    }

    // for B += T.T
    dim3 dimGrid(block_order/tile_dim, block_order/tile_dim, 1);
    dim3 dimBlock(tile_dim, block_rows, 1);
    info.checkDims(dimBlock, dimGrid);

    // for A += 1
    const int threads_per_block = 256;
    const int blocks_per_grid = (order * block_order + threads_per_block - 1) / threads_per_block;

    //std::cout << "@" << me << " order=" << order << " block_order=" << block_order << std::endl;

    const int num_gpus = info.num_gpus();
    info.set_gpu(me % num_gpus);

    //////////////////////////////////////////////////////////////////////
    // Allocate space for the input and transpose matrix
    //////////////////////////////////////////////////////////////////////

    double trans_time{0};

    const size_t nelems = order * block_order;

    double * h_A = prk::CUDA::malloc_host<double>(nelems);
    double * h_B = prk::CUDA::malloc_host<double>(nelems);

    // fill A with the sequence 0 to order^2-1 as doubles
    for (size_t i=0; i<order; i++) {
        for (size_t j=0; j<block_order; j++) {
            h_A[i*block_order + j] = me * block_order + i * order + j;
            h_B[i*block_order + j] = 0;
        }
    }

    //A[order][block_order]
    double * A = prk::NVSHMEM::allocate<double>(nelems);
    double * B = prk::CUDA::malloc_device<double>(nelems);

    prk::CUDA::copyH2D(A, h_A, nelems);
    prk::CUDA::copyH2D(B, h_B, nelems);
    prk::NVSHMEM::barrier(true);

    {
      for (int iter = 0; iter<=iterations; iter++) {

        if (iter==1) {
            prk::NVSHMEM::barrier(false); // sync PEs not memory
            prk::CUDA::sync();
            trans_time = prk::wtime();
        }

        if (on_device) {
#if 1
            // we do this and barrier outside of the kernel because this kernel supports gridsize <= 792 (at least on H100)
            // and that is too small for the transpose algorithm we are doing, which requires e.g. a 32x32x1 grid for a
            // 4096x4096 matrix
            transpose_nvshmem_ptr<<<dimGrid, dimBlock>>>(variant, block_order*block_order, me, np,
                                                         block_order, A, B);
#else
            size_t block_size = block_order*block_order;
            void* args[] = {&variant, &block_size, &me, &np, &block_order, &A, &B};
            size_t sm_size = 32*33*sizeof(double);

            int gridsize = -1;
            prk::check( (cudaError_t)nvshmemx_collective_launch_query_gridsize((const void*)transpose_nvshmem_ptr,
                                                                               dimBlock, args, sm_size, &gridsize) );
            std::cout << "transpose_nvshmem_ptr: NVSHMEM gridsize = " << gridsize << std::endl;
            std::cout << "dimGrid = " << dimGrid.x << "," << dimGrid.y << "," << dimGrid.z << std::endl;

            prk::check( (cudaError_t)nvshmemx_collective_launch((const void*)transpose_nvshmem_ptr, 
                                                                dimGrid, dimBlock,
                                                                args, sm_size, (cudaStream_t)0) );
#endif
        } else {
            // transpose the matrix
            for (int r=0; r<np; r++) {
                const int recv_from = (me + r) % np;
                size_t offset = block_order * block_order * me;
                const double * T = (double*)nvshmem_ptr(A + offset, recv_from);
                offset = block_order * block_order * recv_from;
                if (variant==0) {
                    transposeNaive<<<dimGrid, dimBlock>>>(block_order, T, B + offset);
                } else if (variant==1) {
                    transposeCoalesced<<<dimGrid, dimBlock>>>(block_order, T, B + offset);
                } else if (variant==2) {
                    transposeNoBankConflict<<<dimGrid, dimBlock>>>(block_order, T, B + offset);
                }
            }
        }
        prk::NVSHMEM::barrier(false);
        //prk::CUDA::sync();

        // increment A
        cuda_increment<<<blocks_per_grid, threads_per_block>>>(order * block_order, A);
        prk::NVSHMEM::barrier(false);
        //prk::CUDA::sync();
      }
      //prk::NVSHMEM::barrier(false);
      prk::CUDA::sync();
      trans_time = prk::wtime() - trans_time;
    }

    prk::CUDA::copyD2H(h_B, B, nelems);

    prk::NVSHMEM::free(A);
    prk::CUDA::free(B);

    prk::CUDA::free_host(h_A);

    //////////////////////////////////////////////////////////////////////
    /// Analyze and output results
    //////////////////////////////////////////////////////////////////////

    const double addit = (iterations+1.0) * (iterations*0.5);
    double abserr(0);
    for (size_t i=0; i<order; i++) {
      for (size_t j=0; j<block_order; j++) {
        const double temp = (order*(me*block_order+j)+(i)) * (1+iterations) + addit;
        abserr += prk::abs(h_B[i*block_order+j] - temp);
      }
    }
    //abserr = prk::NVSHMEM::sum(abserr);

    const auto epsilon = 1.0e-8;
    if (abserr > epsilon) {
        throw std::runtime_error("validation failed at PE " + std::to_string(me) );
    }
    prk::NVSHMEM::barrier();

    prk::CUDA::free_host(h_B);

#ifdef VERBOSE
    if (me == 0) std::cout << "Sum of absolute differences: " << abserr << std::endl;
#endif

    if (me == 0) {
      if (abserr < epsilon) {
        std::cout << "Solution validates" << std::endl;
        auto avgtime = trans_time/iterations;
        auto bytes = (size_t)order * (size_t)order * sizeof(double);
        std::cout << "Rate (MB/s): " << 1.0e-6 * (2L*bytes)/avgtime
                  << " Avg time (s): " << avgtime << std::endl;
      } else {
        std::cout << "ERROR: Aggregate squared error " << abserr
                  << " exceeds threshold " << epsilon << std::endl;
        return 1;
      }
    }
  }
  return 0;
}
