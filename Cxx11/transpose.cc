/*
Copyright (c) 2013, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
* Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products
      derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/*******************************************************************

NAME:    transpose

PURPOSE: This program measures the time for the transpose of a
         column-major stored matrix into a row-major stored matrix.

USAGE:   Program input is the matrix order and the number of times to
         repeat the operation:

         transpose <matrix_size> <# iterations> [tile size]

         An optional parameter specifies the tile size used to divide the
         individual matrix blocks for improved cache and TLB performance.

         The output consists of diagnostics to make sure the
         transpose worked and timing statistics.


FUNCTIONS CALLED:

         Other than standard C functions, the following
         functions are used in this program:

         wtime()          portable wall-timer interface.

HISTORY: Written by  Rob Van der Wijngaart, February 2009.
         Modernized by Jeff Hammond, February 2016.
*******************************************************************/

#include "prk_util.h"

#ifdef PRK_LOOP_INDEX_TYPE
typedef PRK_LOOP_INDEX_TYPE prk_index_t;
#else
typedef size_t prk_index_t;
#endif

int main(int argc, char * argv[])
{
  /*********************************************************************
  ** read and test input parameters
  *********************************************************************/

  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "Serial Matrix transpose: B = A^T" << std::endl;

  if (argc != 4 && argc != 3) {
    std::cout << "Usage: " << argv[0] << " <# iterations> <matrix order> [tile size]" << std::endl;
    exit(EXIT_FAILURE);
  }

  int iterations  = std::atoi(argv[1]); /* number of times to do the transpose */
  if (iterations < 1) {
    std::cout << "ERROR: iterations must be >= 1 : " << iterations << std::endl;
    exit(EXIT_FAILURE);
  }

  prk_index_t order = std::atoi(argv[2]); /* order of a the matrix */
  if (order <= 0) {
    std::cout << "ERROR: Matrix Order must be greater than 0 : " << order << std::endl;
    exit(EXIT_FAILURE);
  }

  prk_index_t tile_size = 32; /* default tile size for tiling of local transpose */
  if (argc == 4) {
      tile_size = std::atoi(argv[3]);
  }
  /* a non-positive tile size means no tiling of the local transpose */
  if (tile_size <= 0) {
      tile_size = order;
  }

  /*********************************************************************
  ** Allocate space for the input and transpose matrix
  *********************************************************************/

  double * A;
  double * B;
  try {
      A = new double[order*order];
      B = new double[order*order];
  } catch (std::bad_alloc& err) {
      std::cout << "memory allocation failed" << err.what() << std::endl;
  }

  std::cout << "Matrix order          = " << order << std::endl;
  if (tile_size < order) {
      std::cout << "Tile size             = " << tile_size << std::endl;
  } else {
      std::cout << "Untiled" << std::endl;
  }
  std::cout << "Number of iterations  = " << iterations << std::endl;

  double trans_time = 0.0;

  {
      for (prk_index_t j=0; j<order; j++) {
        for (prk_index_t i=0; i<order; i++) {
          const double val = static_cast<size_t>(order)*static_cast<size_t>(j)+static_cast<size_t>(i);
          A[j*order+i] = val;
          B[j*order+i] = 0.0;
        }
      }

      for (int iter = 0; iter<=iterations; iter++) {
        /* start timer after a warmup iteration */
        if (iter==1) {
          trans_time = prk::wtime();
        }
        /* transpose the  matrix */
        if (tile_size < order) {
          for (prk_index_t it=0; it<order; it+=tile_size) {
            for (prk_index_t jt=0; jt<order; jt+=tile_size) {
              for (prk_index_t i=it; i<std::min(order,it+tile_size); i++) {
                for (prk_index_t j=jt; j<std::min(order,jt+tile_size); j++) {
                  B[i*order+j] += A[j*order+i];
                  A[j*order+i] += 1.0;
                }
              }
            }
          }
        } else {
          for (prk_index_t i=0;i<order; i++) {
            for (prk_index_t j=0;j<order;j++) {
              B[i*order+j] += A[j*order+i];
              A[j*order+i] += 1.0;
            }
          }
        }
      }
      { trans_time = prk::wtime() - trans_time; }

  }

  /*********************************************************************
  ** Analyze and output results.
  *********************************************************************/

  double abserr = 0.0;
  const double addit = ((double)(iterations+1) * (double) (iterations))/2.0;
  for (prk_index_t j=0;j<order;j++) {
    for (prk_index_t i=0;i<order; i++) {
      const size_t offset_ij = (size_t)i*(size_t)order+(size_t)j;
      const double reference = static_cast<double>(offset_ij)*static_cast<double>(iterations+1)+addit;
      abserr += std::fabs(B[j*order+i] - reference);
    }
  }

  delete [] B;
  delete [] A;

#ifdef VERBOSE
  std::cout << "Sum of absolute differences: " << abserr << std::endl;
#endif

  const double epsilon = 1.e-8;
  if (abserr < epsilon) {
    std::cout << "Solution validates" << std::endl;
    double avgtime = trans_time/iterations;
    size_t bytes = (size_t)order * (size_t)order * sizeof(double);
    std::cout << "Rate (MB/s): " << 1.0E-06 * (2L*bytes)/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
    std::cout << "ERROR: Aggregate squared error " << abserr
              << " exceeds threshold " << epsilon << std::endl;
    exit(EXIT_FAILURE);
  }

  return 0;
}


