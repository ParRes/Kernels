///
/// Copyright (c) 2020, Intel Corporation
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
///          <progname> <# iterations> <matrix order> [<tile size>]
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// FUNCTIONS CALLED:
///
///          Other than OpenMP or standard C functions, the following
///          functions are used in this program:
///
///          wtime()
///
/// HISTORY: Written by Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, December, 2017.
///          Converted to C11   by Jeff Hammond, October, 2020.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_openmp.h"

void prk_dgemm(const int order,
               const double * A,
               const double * B,
                     double * C)
{
    OMP_TARGET( teams distribute parallel for simd is_device_ptr(A,B,C) )
    for (int i=0; i<order; ++i) {
      for (int k=0; k<order; ++k) {
        for (int j=0; j<order; ++j) {
            C[i*order+j] += A[i*order+k] * B[k*order+j];
        }
      }
    }
}

void prk_dgemm_tiled(const int order, const int tile_size,
                     const double * A,
                     const double * B,
                           double * C)
{
    OMP_TARGET( teams distribute collapse(3) is_device_ptr(A,B,C) )
    for (int it=0; it<order; it+=tile_size) {
      for (int kt=0; kt<order; kt+=tile_size) {
        for (int jt=0; jt<order; jt+=tile_size) {
          // ICC will not hoist these on its own...
          int iend = MIN(order,it+tile_size);
          int jend = MIN(order,jt+tile_size);
          int kend = MIN(order,kt+tile_size);
          OMP( parallel for simd )
          for (int i=it; i<iend; ++i) {
            for (int k=kt; k<kend; ++k) {
              for (int j=jt; j<jend; ++j) {
                C[i*order+j] += A[i*order+k] * B[k*order+j];
              }
            }
          }
        }
      }
    }
}

int main(int argc, char * argv[])
{
  printf("Parallel Research Kernels version %d\n", PRKVERSION );
  printf("C11/OpenMP TARGET Dense matrix-matrix multiplication: C += A x B\n");

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  if (argc < 3) {
    printf("Usage: <# iterations> <matrix order> [tile size]\n");
    return 1;
  }

  int iterations = atoi(argv[1]);
  if (iterations < 1) {
    printf("ERROR: iterations must be >= 1\n");
    return 1;
  }

  int order = atoi(argv[2]);
  if (order <= 0) {
    printf("ERROR: Matrix Order must be greater than 0\n");
    return 1;
  }

  // default tile size for tiling of local transpose
  int tile_size = (argc>3) ? atoi(argv[3]) : 32;
  // a negative tile size means no tiling of the local transpose
  if (tile_size <= 0) tile_size = order;

  printf("Number of threads (max)   = %d\n", omp_get_max_threads());
  printf("Number of iterations  = %d\n", iterations);
  printf("Matrix order          = %d\n", order);
  printf("Tile size             = %d\n", tile_size);

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for matrices
  //////////////////////////////////////////////////////////////////////

  double dgemm_time = 0.0;

  size_t bytes = order*order*sizeof(double);
  double * restrict A = prk_malloc(bytes);
  double * restrict B = prk_malloc(bytes);
  double * restrict C = prk_malloc(bytes);

  for (int i=0;i<order; i++) {
    for (int j=0;j<order;j++) {
      A[i*order+j] = (double)j;
      B[i*order+j] = (double)j;
      C[i*order+j] = 0.0;
    }
  }

  OMP_TARGET( data map(to: A[0:order*order], B[0:order*order]) map(tofrom: C[0:order*order]) )
  {
    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) dgemm_time = omp_get_wtime();

      if (tile_size < order) {
          prk_dgemm_tiled(order, tile_size, A, B, C);
      } else {
          prk_dgemm(order, A, B, C);
      }
    }
    dgemm_time = omp_get_wtime() - dgemm_time;
  }

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double forder = (double)order;
  const double reference = 0.25 * pow(forder,3) * pow(forder-1.0,2) * (iterations+1.0);
  double checksum = 0.0;
  OMP_PARALLEL_FOR_REDUCE( +:checksum )
  for (int j=0; j<order; j++) {
    for (int i=0; i<order; i++) {
      checksum += C[i*order+j];
    }
  }

  prk_free(A);
  prk_free(B);

#ifdef VERBOSE
  printf("Reference checksum = %lf, checksum = %lf\n", ref_checksum, checksum);
#endif

  const double epsilon = 1.0e-8;
  const double residuum = fabs(checksum-reference)/reference;
  if (residuum < epsilon) {
    printf("Solution validates\n");
    const double avgtime = dgemm_time/iterations;
    double nflops = 2.0*forder*forder*forder;
    printf("Rate (MFlops/s): %lf  Avg time (s): %lf\n", 1.0E-06 * nflops/avgtime, avgtime);
  } else {
    printf("ERROR: Checksum = %lf, Reference checksum = %lf\n", checksum, reference);
    return 1;
  }

  return 0;
}


