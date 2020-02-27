///
/// Copyright (c) 2013, Intel Corporation
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

#include "prk_util.h"

void initialize(const int order, double A[], double B[]);
void transpose(const int order, double A[], double B[]);
void transpose_tiled(const int order, double A[], double B[], const int tile_size);

int main(int argc, char * argv[])
{
  printf("Parallel Research Kernels version %.2f\n", PRKVERSION );
#ifdef _OPENMP
  printf("ISPC + OpenMP Matrix transpose: B = A^T\n");
#else
  printf("ISPC (single threaded) Matrix transpose: B = A^T\n");
#endif

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  if (argc < 3) {
    printf("Usage: <# iterations> <matrix order> [tile size]\n");
    return 1;
  }

  // number of times to do the transpose
  int iterations = atoi(argv[1]);
  if (iterations < 1) {
    printf("ERROR: iterations must be >= 1\n");
    return 1;
  }

  // order of a the matrix
  int order = atoi(argv[2]);
  if (order <= 0) {
    printf("ERROR: Matrix Order must be greater than 0\n");
    return 1;
  }

  // default tile size for tiling of local transpose
  int tile_size = (argc>4) ? atoi(argv[3]) : 32;
  // a negative tile size means no tiling of the local transpose
  if (tile_size <= 0) tile_size = order;

  //printf("ISPC threads          = %d\n", ispc_num_threads());
  printf("Number of iterations  = %d\n", iterations);
  printf("Matrix order          = %d\n", order);
  printf("Tile size             = %d\n", tile_size);

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

  double trans_time = 0.0;

  size_t bytes = order*order*sizeof(double);
  double * restrict A = prk_malloc(bytes);
  double * restrict B = prk_malloc(bytes);

  initialize(order,A,B);

  for (int iter = 0; iter<=iterations; iter++) {
    if (iter==1) trans_time = prk_wtime();
    if (tile_size<order) {
      transpose_tiled(order,A,B,tile_size);
    } else {
      transpose(order,A,B);
    }
  }
  trans_time = prk_wtime() - trans_time;

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const double addit = (iterations+1.) * (iterations/2.);
  double abserr = 0.0;
  for (int j=0; j<order; j++) {
    for (int i=0; i<order; i++) {
      const size_t ij = i*order+j;
      const size_t ji = j*order+i;
      const double reference = (double)(ij)*(1.+iterations)+addit;
      abserr += fabs(B[ji] - reference);
    }
  }

  prk_free(A);
  prk_free(B);

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

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Signature of ispc-generated 'task' functions
typedef void (*TaskFuncType)(void *data,
    int threadIndex,
    int threadCount,
    int taskIndex,
    int taskCount,
    int taskIndex0,
    int taskIndex1,
    int taskIndex2,
    int taskCount0,
    int taskCount1,
    int taskCount2);

void ISPCLaunch(void **taskGroupPtr,
                void *_func,
                void *data,
                int count0,
                int count1,
                int count2)
{
  const int count = count0 * count1 * count2;
  TaskFuncType func = (TaskFuncType)_func;

  OMP_PARALLEL()
  {
#ifdef _OPENMP
    const int threadIndex = omp_get_thread_num();
    const int threadCount = omp_get_num_threads();
#else
    const int threadIndex = 0;
    const int threadCount = 1;
#endif

    OMP_FOR(schedule(runtime))
    for (int i = 0; i < count; i++) {
      int taskIndex0 = i % count0;
      int taskIndex1 = (i / count0) % count1;
      int taskIndex2 = i / (count0 * count1);

      func(data,
           threadIndex,
           threadCount,
           i,
           count,
           taskIndex0,
           taskIndex1,
           taskIndex2,
           count0,
           count1,
           count2);
    }
  }
}

void ISPCSync(void *h)
{
  free(h);
}

void *ISPCAlloc(void **taskGroupPtr, int64_t size, int32_t alignment)
{
  *taskGroupPtr = aligned_alloc(alignment, size);
  return *taskGroupPtr;
}
