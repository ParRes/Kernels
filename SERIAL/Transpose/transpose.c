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

#include <par-res-kern_general.h>

int main(int argc, char ** argv)
{
  /*********************************************************************
  ** read and test input parameters
  *********************************************************************/

  printf("Parallel Research Kernels version %s\n", PRKVERSION);
  printf("Serial Matrix transpose: B = A^T\n");

  if (argc != 4 && argc != 3) {
    printf("Usage: %s <# iterations> <matrix order> [tile size]\n", *argv);
    exit(EXIT_FAILURE);
  }

  int iterations  = atoi(*++argv); /* number of times to do the transpose */
  if (iterations < 1) {
    printf("ERROR: iterations must be >= 1 : %d \n", iterations);
    exit(EXIT_FAILURE);
  }

  int order = atol(*++argv); /* order of a the matrix */
  if (order < 0) {
    printf("ERROR: Matrix Order must be greater than 0 : %d \n", order);
    exit(EXIT_FAILURE);
  }

  int tile_size = 32; /* default tile size for tiling of local transpose */
  if (argc == 4) {
      tile_size = atoi(*++argv);
  }
  /* a non-positive tile size means no tiling of the local transpose */
  if (tile_size <=0) {
      tile_size = order;
  }

  /*********************************************************************
  ** Allocate space for the input and transpose matrix
  *********************************************************************/

  size_t bytes = (size_t)order * (size_t)order * sizeof(double);

  double (* const restrict A)[order] = (double (*)[order]) prk_malloc(bytes);
  if (A == NULL){
    printf(" Error allocating space for transposed matrix\n");
    exit(EXIT_FAILURE);
  }
  double (* const restrict B)[order] = (double (*)[order]) prk_malloc(bytes);
  if (B == NULL){
    printf(" Error allocating space for input matrix\n");
    exit(EXIT_FAILURE);
  }

  printf("Matrix order          = %d\n", order);
  if (tile_size < order) {
      printf("Tile size             = %d\n", tile_size);
  } else {
      printf("Untiled\n");
  }
  printf("Number of iterations  = %d\n", iterations);

  /*  Fill the original matrix, set transpose to known garbage value. */

  /* Fill the original column matrix. */
  {
      for (int j=0;j<order; j++) {
        for (int i=0;i<order; i++) {
          A[j][i] = (double) ((size_t)order*(size_t)j+(size_t)i);
        }
      }
  }

  /*  Set the transpose matrix to a known garbage value. */
  {
      for (int j=0;j<order; j++) {
        for (int i=0;i<order; i++) {
          B[j][i] = 0.0;
        }
      }
  }

  double trans_time = 0.0;

  for (int iter = 0; iter<=iterations; iter++){

    /* start timer after a warmup iteration */
    if (iter==1) trans_time = wtime();

    /* Transpose the  matrix */
    if (tile_size < order) {
      for (int it=0; it<order; it+=tile_size) {
        for (int jt=0; jt<order; jt+=tile_size) {
          for (int i=it; i<MIN(order,it+tile_size); i++) {
            for (int j=jt; j<MIN(order,jt+tile_size); j++) {
              B[i][j] += A[j][i];
              A[j][i] += 1.0;
            }
          }
        }
      }
    } else {
      for (int i=0;i<order; i++) {
        for (int j=0;j<order;j++) {
          B[i][j] += A[j][i];
          A[j][i] += 1.0;
        }
      }
    }
  }  /* end of iter loop  */

  trans_time = wtime() - trans_time;

  /*********************************************************************
  ** Analyze and output results.
  *********************************************************************/

  double addit = ((double)(iterations+1) * (double) (iterations))/2.0;
  double abserr = 0.0;
  for (int j=0;j<order;j++) {
    for (int i=0;i<order; i++) {
      abserr += ABS(B[j][i] - ((double)(order*(size_t)i+(size_t)j)*(iterations+1L)+addit));
    }
  }

  prk_free(B);
  prk_free(A);

#ifdef VERBOSE
  printf("Sum of absolute differences: %f\n",abserr);
#endif

  const double epsilon = 1.e-8;
  if (abserr < epsilon) {
    printf("Solution validates\n");
    double avgtime = trans_time/iterations;
    printf("Rate (MB/s): %lf Avg time (s): %lf\n", 1.0E-06 * (2L*bytes)/avgtime, avgtime);
#ifdef VERBOSE
    printf("Squared errors: %f \n", abserr);
#endif
    exit(EXIT_SUCCESS);
  }
  else {
    printf("ERROR: Aggregate squared error %lf exceeds threshold %e\n", abserr, epsilon);
    exit(EXIT_FAILURE);
  }

  return 0;
}


