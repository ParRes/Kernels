///
/// Copyright (c) 2019, Intel Corporation
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
/// NAME:    nstream
///
/// PURPOSE: To compute memory bandwidth when adding a vector of a given
///          number of double precision values to the scalar multiple of
///          another vector of the same length, and storing the result in
///          a third vector.
///
/// USAGE:   The program takes as input the number
///          of iterations to loop over the triad vectors, the length of the
///          vectors, and the offset between vectors
///
///          <progname> <# iterations> <vector length> <offset>
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// NOTES:   Bandwidth is determined as the number of words read, plus the
///          number of words written, times the size of the words, divided
///          by the execution time. For a vector length of N, the total
///          number of words read and written is 4*N*sizeof(double).
///
///
/// HISTORY: This code is loosely based on the Stream benchmark by John
///          McCalpin, but does not follow all the Stream rules. Hence,
///          reported results should not be associated with Stream in
///          external publications
///
///          Converted to C++11 by Jeff Hammond, November 2017.
///          Converted to C11 by Jeff Hammond, February 2019.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"

int main(int argc, char * argv[])
{
  printf("Parallel Research Kernels version %.2f\n", PRKVERSION );
#ifdef _OPENMP
  printf("C11/OpenMP TASKLOOP STREAM triad: A = B + scalar * C\n");
#else
  printf("C11/Serial STREAM triad: A = B + scalar * C\n");
#endif

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  if (argc < 3) {
    printf("Usage: <# iterations> <vector length> [<taskloop grainsize>]\n");
    return 1;
  }

  // number of times to do the transpose
  int iterations = atoi(argv[1]);
  if (iterations < 1) {
    printf("ERROR: iterations must be >= 1\n");
    return 1;
  }

  // length of a the matrix
  size_t length = atol(argv[2]);
  if (length <= 0) {
    printf("ERROR: Matrix length must be greater than 0\n");
    return 1;
  }

  // taskloop grainsize
  int gs = (argc > 3) ? atoi(argv[3]) : 1024;


#ifdef _OPENMP
  printf("Number of threads    = %d\n", omp_get_max_threads());
  printf("Taskloop grainsize    = %d\n", gs);
#endif
  printf("Number of iterations = %d\n", iterations);
  printf("Vector length        = %zu\n", length);
  //printf("Offset               = %d\n", offset);

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double nstream_time = 0.0;

  size_t bytes = length*sizeof(double);
  double * restrict A = prk_malloc(bytes);
  double * restrict B = prk_malloc(bytes);
  double * restrict C = prk_malloc(bytes);

  double scalar = 3.0;

  OMP_PARALLEL()
  OMP_MASTER
  {
    OMP_TASKLOOP( firstprivate(length) shared(A,B,C) grainsize(gs) )
    for (size_t i=0; i<length; i++) {
      A[i] = 0.0;
      B[i] = 2.0;
      C[i] = 2.0;
    }
    OMP_TASKWAIT

    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) nstream_time = prk_wtime();

      OMP_TASKLOOP( firstprivate(length) shared(A,B,C) grainsize(gs) )
      for (size_t i=0; i<length; i++) {
          A[i] += B[i] + scalar * C[i];
      }
      OMP_TASKWAIT
    }
    nstream_time = prk_wtime() - nstream_time;
  }

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  double ar = 0.0;
  double br = 2.0;
  double cr = 2.0;
  for (int i=0; i<=iterations; i++) {
      ar += br + scalar * cr;
  }

  ar *= length;

  double asum = 0.0;
  OMP_PARALLEL_FOR_REDUCE( +:asum )
  for (size_t i=0; i<length; i++) {
      asum += fabs(A[i]);
  }

  double epsilon=1.e-8;
  if (fabs(ar-asum)/asum > epsilon) {
      printf("Failed Validation on output array\n"
             "       Expected checksum: %lf\n"
             "       Observed checksum: %lf\n"
             "ERROR: solution did not validate\n", ar, asum);
      return 1;
  } else {
      printf("Solution validates\n");
      double avgtime = nstream_time/iterations;
      double nbytes = 4.0 * length * sizeof(double);
      printf("Rate (MB/s): %lf Avg time (s): %lf\n", 1.e-6*nbytes/avgtime, avgtime);
  }

  return 0;
}


