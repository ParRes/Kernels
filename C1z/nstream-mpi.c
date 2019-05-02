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

#include <mpi.h>

int main(int argc, char * argv[])
{
  int me, np;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (me==0) {
      printf("Parallel Research Kernels version %.2f\n", PRKVERSION );
      printf("C11/MPI STREAM triad: A = B + scalar * C\n");
  }

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  if (argc < 3) {
    if (me==0) printf("Usage: <# iterations> <vector length>\n");
    MPI_Finalize();
    return 1;
  }

  // number of times to do the transpose
  int iterations = atoi(argv[1]);
  if (iterations < 1) {
    if (me==0) printf("ERROR: iterations must be >= 1\n");
    MPI_Finalize();
    return 1;
  }

  // length of a the matrix
  size_t length = atol(argv[2]);
  if (length <= 0) {
    if (me==0) printf("ERROR: Matrix length must be greater than 0\n");
    MPI_Finalize();
    return 1;
  }

  if (me==0) {
      printf("Number of processes  = %d\n", np);
      printf("Number of iterations = %d\n", iterations);
      printf("Vector length        = %zu\n", length);
      //printf("Offset               = %d\n", offset);
  }

  size_t local_length;
  if (length % np == 0) {
      local_length = length / np;
  } else {
      double x = (double)length / np;
      size_t y = (size_t)ceil(x);
      if (me != (np-1)) {
          local_length = y;
      } else {
          local_length = length - y*(np-1);
      }
  }
  //printf("Vector length (%4d) = %zu\n", me, local_length);
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double nstream_time = 0.0;

  double * restrict A;
  double * restrict B;
  double * restrict C;

  MPI_Win wA, wB, wC;

  size_t bytes = local_length*sizeof(double);

  MPI_Win_allocate_shared(bytes, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, (void**)&A, &wA);
  MPI_Win_allocate_shared(bytes, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, (void**)&B, &wB);
  MPI_Win_allocate_shared(bytes, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, (void**)&C, &wC);

  double scalar = 3.0;

  for (size_t i=0; i<local_length; i++) {
    A[i] = 0.0;
    B[i] = 2.0;
    C[i] = 2.0;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  for (int iter = 0; iter<=iterations; iter++) {

    if (iter==1) {
        MPI_Barrier(MPI_COMM_WORLD);
        nstream_time = MPI_Wtime();
    }

    for (size_t i=0; i<local_length; i++) {
        A[i] += B[i] + scalar * C[i];
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  nstream_time = MPI_Wtime() - nstream_time;

  MPI_Allreduce(MPI_IN_PLACE, &nstream_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  double ar = 0.0;
  double br = 2.0;
  double cr = 2.0;
  for (int i=0; i<=iterations; i++) {
      ar += br + scalar * cr;
  }

  ar *= local_length;

  double asum = 0.0;
  for (size_t i=0; i<local_length; i++) {
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
      if (me==0) printf("Solution validates\n");
      double avgtime = nstream_time/iterations;
      double nbytes = 4.0 * length * sizeof(double);
      if (me==0) printf("Rate (MB/s): %lf Avg time (s): %lf\n", 1.e-6*nbytes/avgtime, avgtime);
  }

  MPI_Win_free(&wA);
  MPI_Win_free(&wB);
  MPI_Win_free(&wC);

  MPI_Finalize();

  return 0;
}


