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

NAME:    reduce

PURPOSE: This program tests the efficiency with which a pair of 
         vectors can be added in elementwise fashion. 
  
USAGE:   The program takes as input the length of the vectors and the 
         number of times the reduction is repeated, 

         <progname> <# iterations> <vector length> [<algorithm>]
  
         The output consists of diagnostics to make sure the 
         algorithm worked, and of timing statistics.

FUNCTIONS CALLED:

         Other than standard C functions, the following 
         functions are used in this program:

         wtime()

HISTORY: Written by Rob Van der Wijngaart, February 2009.
  
*******************************************************************/

#include <par-res-kern_general.h>

/* MEMWORDS is the total number of words needed. */
#ifndef MEMWORDS
  #define MEMWORDS  10000000
#endif

#define VEC0(i)        vector[i]
#define VEC1(i)        vector[vector_length+i]

int main(int argc, char ** argv)
{
  int    vector_length;   /* length of vectors to be aggregated            */
  int    total_length;    /* bytes needed to store reduction vectors       */
  double reduce_time,     /* timing parameters                             */
         avgtime = 0.0, 
         maxtime = 0.0, 
         mintime = 366.0*24.0*3600.0; /* set the minimum time to a large 
                             value; one leap year should be enough           */
  double epsilon=1.e-8;   /* error tolerance                                 */
  int    i, iter;         /* dummies                                         */
  double element_value;   /* reference element value for final vector        */
  int    iterations;      /* number of times the reduction is carried out    */
  static double           /* use static so it goes on the heap, not stack    */
  RESTRICT vector[MEMWORDS];/* we would like to allocate "vector" dynamically, 
                             but need to be able to flush the thing in some 
                             versions of the reduction algorithm -> static   */

/*****************************************************************************
** process and test input parameters    
******************************************************************************/

  if (argc != 3){
    printf("Usage:     %s <# iterations> <vector length>\n", *argv);
    return(EXIT_FAILURE);
  }

  iterations = atoi(*++argv);
  if (iterations < 1){
    printf("ERROR: Iterations must be positive : %d \n", iterations);
    exit(EXIT_FAILURE);
  }

  vector_length  = atoi(*++argv);
  if (vector_length < 1){
    printf("ERROR: vector length must be >= 1 : %d \n",vector_length);
    exit(EXIT_FAILURE);
  }
  /*  make sure we stay within the memory allocated for vector               */
  total_length = 2*vector_length;
  if (total_length/2 != vector_length || total_length > MEMWORDS) {
    printf("Vector length of %d too large; ", vector_length);
    printf("increase MEMWORDS in Makefile or reduce vector length\n");
    exit(EXIT_FAILURE);
  }

  printf("Serial Vector Reduction\n");
  printf("Vector length                  = %d\n", vector_length);
  printf("Number of iterations           = %d\n", iterations);

  for (iter=0; iter<iterations; iter++) {

    /* initialize the arrays, assuming first-touch memory placement          */
    for (i=0; i<vector_length; i++) {
      VEC0(i) = (double)(1);
      VEC1(i) = (double)(2);
    }
   
    reduce_time = wtime();
    /* do actual reduction                                                   */

    /* first do the "local" part, which is the same for all algorithms       */
    for (i=0; i<vector_length; i++) {
      VEC0(i) += VEC1(i);
    }

    reduce_time = wtime() - reduce_time;
#ifdef VERBOSE
    printf("\nFinished with reduction, using %lf seconds \n", reduce_time);
#endif
    if (iter>0 || iterations==1) { /* skip the first iteration               */
      avgtime = avgtime + reduce_time;
      mintime = MIN(mintime, reduce_time);
      maxtime = MAX(maxtime, reduce_time);
    }

  } /* end of iter loop                                                      */

  /* verify correctness */
  element_value = (2.0+1.0);

  for (i=0; i<vector_length; i++) {
    if (ABS(VEC0(i) - element_value) >= epsilon) {
       printf("First error at i=%d; value: %lf; reference value: %lf\n",
              i, VEC0(i), element_value);
       exit(EXIT_FAILURE);
    }
  }

  printf("Solution validates\n");
#ifdef VERBOSE
  printf("Element verification value: %lf\n", element_value);
#endif
  avgtime = avgtime/(double)(MAX(iterations-1,1));
  printf("Rate (MFlops/s): %lf,  Avg time (s): %lf,  Min time (s): %lf",
         1.0E-06 * (2.0-1.0)*vector_length/mintime, avgtime, mintime);
  printf(", Max time (s): %lf\n", maxtime);

  exit(EXIT_SUCCESS);
}
