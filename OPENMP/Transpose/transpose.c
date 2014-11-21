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

PURPOSE: This OpenMP program measures the time for the transpose of a 
         column-major stored matrix into a row-major stored matrix.
  
USAGE:   Program input is three command line arguments that give the
         matrix order, the number of times to repeat the operation 
         (iterations), and the number of threads to use:

         transpose <# threads> <matrix_size> <# iterations> [tile size]

         An optional parameter specifies the tile size used to divide the
         individual matrix blocks for improved cache and TLB performance. 
  
         The output consists of diagnostics to make sure the 
         transpose worked and timing statistics.


FUNCTIONS CALLED:

         Other than OpenMP or standard C functions, the following 
         functions are used in this program:

         wtime()          portable wall-timer interface.
         bail_out()
         test_results()   Verify that the transpose worked

HISTORY: Written by Tim Mattson, April 1999.  
         Updated by Rob Van der Wijngaart, December 2005.
  
*******************************************************************/

#include <par-res-kern_general.h>
#include <par-res-kern_omp.h>

#define A(i,j)    A[i+order*(j)]
#define B(i,j)    B[i+order*(j)]
static double test_results (int , double*);

int main(int argc, char ** argv) {

  int    order;         /* order of a the matrix                           */
  int    tile_size=32;  /* default tile size for tiling of local transpose */
  int    iterations;    /* number of times to do the transpose             */
  int    i, j, it, jt, iter;  /* dummies                                   */
  double bytes;         /* combined size of matrices                       */
  double * RESTRICT A;  /* buffer to hold original matrix                  */
  double * RESTRICT B;  /* buffer to hold transposed matrix                */
  double abserr;        /* absolute error                                  */
  double epsilon=1.e-8; /* error tolerance                                 */
  double transpose_time,/* timing parameters                               */
         avgtime;
  int    nthread_input, 
         nthread;
  int    num_error=0;     /* flag that signals that requested and 
                             obtained numbers of threads are the same      */

  /*********************************************************************
  ** read and test input parameters
  *********************************************************************/

  if (argc != 4 && argc != 5){
    printf("Usage: %s <# threads> <# iterations> <matrix order> [tile size]\n",
           *argv);
    exit(EXIT_FAILURE);
  }

  /* Take number of threads to request from command line */
  nthread_input = atoi(*++argv); 

  if ((nthread_input < 1) || (nthread_input > MAX_THREADS)) {
    printf("ERROR: Invalid number of threads: %d\n", nthread_input);
    exit(EXIT_FAILURE);
  }

  omp_set_num_threads(nthread_input);

  iterations  = atoi(*++argv); 
  if (iterations < 1){
    printf("ERROR: iterations must be >= 1 : %d \n",iterations);
    exit(EXIT_FAILURE);
  }

  order = atoi(*++argv); 
  if (order < 0){
    printf("ERROR: Matrix Order must be greater than 0 : %d \n", order);
    exit(EXIT_FAILURE);
  }

  if (argc == 5) tile_size = atoi(*++argv);
  /* a non-positive tile size means no tiling of the local transpose */
  if (tile_size <=0) tile_size = order;

  /*********************************************************************
  ** Allocate space for the input and transpose matrix
  *********************************************************************/

  A   = (double *)malloc(order*order*sizeof(double));
  if (A == NULL){
    printf(" Error allocating space for input matrix\n");
    exit(EXIT_FAILURE);
  }
  B  = (double *)malloc(order*order*sizeof(double));
  if (B == NULL){
    printf(" Error allocating space for transposed matrix\n");
    exit(EXIT_FAILURE);
  }

  bytes = 2.0 * sizeof(double) * order * order;

  #pragma omp parallel private (iter)
  {  

  #pragma omp master
  {
  nthread = omp_get_num_threads();

  printf("OpenMP Matrix transpose: B = A^T\n");
  if (nthread != nthread_input) {
    num_error = 1;
    printf("ERROR: number of requested threads %d does not equal ",
           nthread_input);
    printf("number of spawned threads %d\n", nthread);
  } 
  else {
    printf("Number of threads     = %i;\n",nthread_input);
    printf("Matrix order          = %d\n", order);
    if (tile_size < order) printf("Tile size             = %d\n", tile_size);
    else                   printf("Untiled\n");
    printf("Number of iterations  = %d\n", iterations);
  }
  }
  bail_out(num_error);

  /*  Fill the original matrix, set transpose to known garbage value. */

  #pragma omp for private (i)
  for (j=0;j<order;j++) {
    for (i=0;i<order; i++) {
      A(i,j) = (double) (order*j + i);
      B(i,j) = -1.0;
    }
  }

  for (iter = 0; iter<=iterations; iter++){

    /* start timer after a warmup iteration                                        */
    if (iter == 1) { 
      #pragma omp barrier
      #pragma omp master
      {
        transpose_time = wtime();
      }
    }

    /* Transpose the  matrix; only use tiling if the tile size is smaller 
       than the matrix */
    if (tile_size < order) {
      #pragma omp for private (i, it, jt)
      for (j=0; j<order; j+=tile_size) 
        for (i=0; i<order; i+=tile_size) 
          for (it=i; it<MIN(order,i+tile_size); it++)
            for (jt=j; jt<MIN(order,j+tile_size);jt++){ 
                B(jt,it) = A(it,jt);
            } 
    }
    else {
      #pragma omp for private (i)
      for (j=0;j<order;j++) 
        for (i=0;i<order; i++) {
          B(j,i) = A(i,j);
        }
    }	

  }  /* end of iter loop  */

  #pragma omp barrier
  #pragma omp master
  {
    transpose_time = wtime() - transpose_time;
  }

  } /* end of OpenMP parallel region */

  abserr =  test_results (order, B);

  /*********************************************************************
  ** Analyze and output results.
  *********************************************************************/

  if (abserr < epsilon) {
    printf("Solution validates\n");
    avgtime = transpose_time/iterations;
    printf("Rate (MB/s): %lf Avg time (s): %lf\n",
           1.0E-06 * bytes/avgtime, avgtime);
#ifdef VERBOSE
    printf("Squared errors: %f \n", abserr);
#endif
    exit(EXIT_SUCCESS);
  }
  else {
    printf("ERROR: Aggregate squared error %lf exceeds threshold %e\n",
           abserr, epsilon);
    exit(EXIT_FAILURE);
  }

}  /* end of main */



/* function that computes the error committed during the transposition */

double test_results (int order, double *B) {

  double abserr=0.0;
  int i,j;

  #pragma omp parallel for private(i) reduction(+:abserr)
  for (j=0;j<order;j++) {
    for (i=0;i<order; i++) {
      abserr += ABS(B(i,j) - (i*order + j));
    }
  }

#ifdef VERBOSE
  #pragma omp master 
  {
  printf(" Squared sum of differences: %f\n",abserr);
  }
#endif   
  return abserr;
}
