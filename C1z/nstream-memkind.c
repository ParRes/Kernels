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

#include <memkind.h>
#ifndef MEMKIND_PMEM_MIN_SIZE
# define MEMKIND_PMEM_MIN_SIZE (1024 * 1024 * 16)
#endif

int main(int argc, char * argv[])
{
  printf("Parallel Research Kernels version %.2f\n", PRKVERSION );
#ifdef _OPENMP
  printf("C11/OpenMP STREAM triad: A = B + scalar * C\n");
#else
  printf("C11 STREAM triad: A = B + scalar * C\n");
#endif

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  if (argc < 3) {
    printf("Usage: <# iterations> <vector length>\n");
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

#ifdef _OPENMP
  printf("Number of threads    = %d\n", omp_get_max_threads());
#endif
  printf("Number of iterations = %d\n", iterations);
  printf("Vector length        = %zu\n", length);
  //printf("Offset               = %d\n", offset);

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double nstream_time = 0.0;

  size_t bytes = length*sizeof(double);

  char * pool_path = getenv("PRK_MEMKIND_POOL_PATH");
  if (pool_path == NULL) {
      pool_path = "/pmem";
  }
  printf("MEMKIND pool path = %s\n", pool_path);
  struct memkind * memkind_handle;
  int err = memkind_create_pmem(pool_path, 0, &memkind_handle);
  if (err) {
    printf("MEMKIND failed to create a memory pool! (err=%d, errno=%d)\n", err, errno);
  }

  size_t usable_size = 0;

  double * restrict A = memkind_malloc(memkind_handle, bytes);
  if (A==NULL) {
    printf("MEMKIND failed to allocate A! (errno=%d)\n", errno);
  }
  usable_size = memkind_malloc_usable_size(memkind_handle, A);
  printf("A usage size = %zu\n", usable_size);

  double * restrict B = memkind_malloc(memkind_handle, bytes);
  if (B==NULL) {
    printf("MEMKIND failed to allocate B! (errno=%d)\n", errno);
  }
  usable_size = memkind_malloc_usable_size(memkind_handle, B);
  printf("B usage size = %zu\n", usable_size);

  double * restrict C = memkind_malloc(memkind_handle, bytes);
  if (C==NULL) {
    printf("MEMKIND failed to allocate C! (errno=%d)\n", errno);
  }
  usable_size = memkind_malloc_usable_size(memkind_handle, C);
  printf("C usage size = %zu\n", usable_size);

  double scalar = 3.0;

  OMP_PARALLEL()
  {
    OMP_FOR_SIMD()
    for (size_t i=0; i<length; i++) {
      A[i] = 0.0;
      B[i] = 2.0;
      C[i] = 2.0;
    }

    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) {
          OMP_BARRIER
          OMP_MASTER
          nstream_time = prk_wtime();
      }

      OMP_FOR_SIMD()
      for (size_t i=0; i<length; i++) {
          A[i] += B[i] + scalar * C[i];
      }
    }
    OMP_BARRIER
    OMP_MASTER
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

  memkind_free(memkind_handle, A);
  memkind_free(memkind_handle, B);
  memkind_free(memkind_handle, C);

  err = memkind_destroy_kind(memkind_handle);
  if (err) {
      printf("MEMKIND failed to create destroy a memory pool! (err=%d, errno=%d)\n", err, errno);
  }

  return 0;
}


