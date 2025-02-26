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
#include "prk_mpi.h"
#include "transpose-kernel.h"

int main(int argc, char * argv[])
{
  {
    prk::MPI::state mpi(&argc,&argv);

    int np = prk::MPI::size();
    int me = prk::MPI::rank();

    //////////////////////////////////////////////////////////////////////
    /// Read and test input parameters
    //////////////////////////////////////////////////////////////////////

    int iterations;
    size_t order, block_order, tile_size;

    if (me == 0) {
      std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
      std::cout << "C++11/MPI Matrix transpose: B = A^T" << std::endl;

      try {
        if (argc < 3) {
          throw "Usage: <# iterations> <matrix order> [tile size]";
        }
     
        iterations  = std::atoi(argv[1]);
        if (iterations < 1) {
          throw "ERROR: iterations must be >= 1";
        }
     
        order = std::atol(argv[2]);
        if (order <= 0) {
          throw "ERROR: Matrix Order must be greater than 0";
        // } else if (order > prk::get_max_matrix_size()) {
        //   throw "ERROR: matrix dimension too large - overflow risk";
        }

        if (order % np != 0) {
          throw "ERROR: Matrix order must be an integer multiple of the number of MPI processes";
        }

        // default tile size for tiling of local transpose
        tile_size = (argc>3) ? std::atoi(argv[3]) : 32;
        // a negative tile size means no tiling of the local transpose
        if (tile_size <= 0) tile_size = order;
      }
      catch (const char * e) {
        std::cout << e << std::endl;
        prk::MPI::abort(1);
        return 1;
      }
     
      std::cout << "Number of iterations = " << iterations << std::endl;
      std::cout << "Matrix order         = " << order << std::endl;
      std::cout << "Tile size            = " << tile_size << std::endl;
    }

    prk::MPI::bcast(&iterations);
    prk::MPI::bcast(&order);
    prk::MPI::bcast(&tile_size);
    
    block_order = order / np;

    //std::cout << "@" << me << " order=" << order << " block_order=" << block_order << std::endl;

    //////////////////////////////////////////////////////////////////////
    // Allocate space for the input and transpose matrix
    //////////////////////////////////////////////////////////////////////

    double trans_time{0};

    //A[order][block_order]
    prk::vector<double> A(order * block_order, 0.0);
    prk::vector<double> B(order * block_order, 0.0);
    prk::vector<double> T(order * block_order, 0.0);

    // fill A with the sequence 0 to order^2-1 as doubles
    for (size_t i=0; i<order; i++) {
        for (size_t j=0; j<block_order; j++) {
            A[i*block_order + j] = me * block_order + i * order + j;
        }
    }
    prk::MPI::barrier();

    //prk::MPI::print_matrix(A, order, block_order, "A@" + std::to_string(me));

    {
      for (int iter = 0; iter<=iterations; iter++) {

        if (iter==1) {
            trans_time = prk::wtime();
            prk::MPI::barrier();
        }

        prk::MPI::alltoall(A.data(), block_order*block_order, T.data(), block_order*block_order);

        // transpose the  matrix  
        for (int r=0; r<np; r++) {
          const size_t offset = block_order * block_order * r;
          transpose_block(B.data() + offset, T.data() + offset, block_order, tile_size); 
        }
        // increment A
        std::transform(A.begin(), A.end(), A.begin(), [](auto a) { return a + 1; });
      }
      trans_time = prk::wtime() - trans_time;
    }

    //prk::MPI::print_matrix(A, order, block_order, "A@" + std::to_string(me));
    //prk::MPI::print_matrix(B, order, block_order, "B@" + std::to_string(me));

    //////////////////////////////////////////////////////////////////////
    /// Analyze and output results
    //////////////////////////////////////////////////////////////////////

    const double addit = (iterations+1.0) * (iterations*0.5);
    double abserr(0);
    for (size_t i=0; i<order; i++) {
      for (size_t j=0; j<block_order; j++) {
        const double temp = (order*(me*block_order+j)+(i)) * (1+iterations) + addit;
        abserr += prk::abs(B[i*block_order+j] - temp);
      }
    }
    abserr = prk::MPI::sum(abserr);

#ifdef VERBOSE
    std::cout << "Sum of absolute differences: " << abserr << std::endl;
#endif

    if (me == 0) {
      const auto epsilon = 1.0e-8;
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
