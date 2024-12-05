///
/// Copyright (c) 2013, Intel Corporation
/// Copyright (c) 2023, NVIDIA
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
///          C11-ification by Jeff Hammond, June 2017.
///
//////////////////////////////////////////////////////////////////////

#include <openacc.h>
#include "prk_util.h"

int main(int argc, char * argv[])
{
  printf("Parallel Research Kernels version %d\n", PRKVERSION );
  printf("C11/OpenACC Matrix transpose: B = A^T\n");

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

  printf("Number of iterations  = %d\n", iterations);
  printf("Matrix order          = %d\n", order);

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

  double trans_time = 0.0;

  size_t bytes = order*order*sizeof(double);

  double (* const restrict A)[order] = (double (*)[order]) acc_malloc(bytes);
  if (A == NULL) {
    printf(" Error allocating space for transposed matrix\n");
    exit(EXIT_FAILURE);
  }
  double (* const restrict B)[order] = (double (*)[order]) acc_malloc(bytes);
  if (B == NULL) {
    printf(" Error allocating space for input matrix\n");
    exit(EXIT_FAILURE);
  }

  {
#ifdef LOOP
    #pragma acc parallel loop deviceptr(A,B)
#else
    #pragma acc kernels deviceptr(A,B)
#endif
    for (int j=0;j<order;j++) {
      for (int i=0;i<order; i++) {
        A[j][i] = (double)(i*order+j);
        B[j][i] = 0.0;
      }
    }

    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) trans_time = prk_wtime();

#ifdef LOOP
      #pragma acc parallel loop tile(*,*) deviceptr(A,B)
#else
      #pragma acc kernels deviceptr(A,B)
#endif
      for (int i=0;i<order; i++) {
        for (int j=0;j<order;j++) {
          B[i][j] += A[j][i];
          A[j][i] += 1.0;
        }
      }
    }
    trans_time = prk_wtime() - trans_time;
  }

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double addit = (iterations+1.) * (iterations/2.);
  double abserr = 0.0;
  #pragma acc parallel loop reduction( +:abserr ) deviceptr(B)
  for (int j=0; j<order; j++) {
    for (int i=0; i<order; i++) {
      const size_t ji = j*order+i;
      const double reference = (double)(ji)*(1.+iterations)+addit;
      abserr += fabs(B[j][i] - reference);
    }
  }

  acc_free(A);
  acc_free(B);

#ifdef VERBOSE
  printf("Sum of absolute differences: %lf\n", abserr);
#endif

  const double epsilon = 1.0e-8;
  if (abserr < epsilon) {
    printf("Solution validates\n");
    const double avgtime = trans_time/iterations;
    printf("Rate (MB/s): %lf Avg time (s): %lf\n", 2.0e-6 * bytes/avgtime, avgtime );
  } else {
    printf("ERROR: Aggregate squared error %lf exceeds threshold %lf\n", abserr, epsilon );
    return 1;
  }

  return 0;
}


