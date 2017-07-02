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
/// NAME:    Pipeline
///
/// PURPOSE: This program tests the efficiency with which point-to-point
///          synchronization can be carried out. It does so by executing
///          a pipelined algorithm on an n*n grid. The first array dimension
///          is distributed among the threads (stripwise decomposition).
///
/// USAGE:   The program takes as input the
///          dimensions of the grid, and the number of iterations on the grid
///
///                <progname> <iterations> <n> <n>
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// FUNCTIONS CALLED:
///
///          Other than standard C functions, the following
///          functions are used in this program:
///
///          wtime()
///
/// HISTORY: - Written by Rob Van der Wijngaart, February 2009.
///          - C99-ification by Jeff Hammond, February 2016.
///          - C11-ification by Jeff Hammond, June 2017.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"

int main(int argc, char * argv[])
{
  printf("Parallel Research Kernels version %.2f\n", PRKVERSION);
  printf("C11/OpenMP pipeline execution on 2D grid\n");

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  if (argc < 3) {
    printf("Usage: <# iterations> <array dimension>\n");
    return 1;
  }

  // number of times to run the pipeline algorithm
  int iterations = atoi(argv[1]);
  if (iterations < 1) {
    printf("ERROR: iterations must be >= 1\n");
    return 1;
  }

  // grid dimensions
  int n = atol(argv[2]);
  if (n < 1) {
    printf("ERROR: grid dimension must be positive: %d\n", n);
    return 1;
  }

  printf("Number of threads (max)   = %d\n", omp_get_max_threads());
  printf("Number of iterations      = %d\n", iterations);
  printf("Grid sizes                = %d,%d\n", n, n);

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double pipeline_time = 0.0; // silence compiler warning

  size_t bytes = n*n*sizeof(double);
  double * restrict grid = prk_malloc(bytes);

  _Pragma("omp parallel")
  {
    PRAGMA_OMP_FOR_SIMD
    for (int i=0; i<n; i++) {
      for (int j=0; j<n; j++) {
        grid[i*n+j] = 0.0;
      }
    }

    // set boundary values (bottom and left side of grid)
    _Pragma("omp master")
    {
      for (int j=0; j<n; j++) {
        grid[0*n+j] = (double)j;
      }
      for (int i=0; i<n; i++) {
        grid[i*n+0] = (double)i;
      }
    }
    _Pragma("omp barrier")

    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) {
          _Pragma("omp barrier")
          _Pragma("omp master")
          pipeline_time = prk_wtime();
      }

      for (int j=1; j<n; j++) {
        PRAGMA_OMP_FOR_SIMD
        for (int i=1; i<=j; i++) {
          const int x = i;
          const int y = j-i+1;
          grid[x*n+y] = grid[(x-1)*n+y] + grid[x*n+(y-1)] - grid[(x-1)*n+(y-1)];
        }
      }
      for (int j=n-2; j>=1; j--) {
        PRAGMA_OMP_FOR_SIMD
        for (int i=1; i<=j; i++) {
          const int x = n+i-j-1;
          const int y = n-i;
          grid[x*n+y] = grid[(x-1)*n+y] + grid[x*n+(y-1)] - grid[(x-1)*n+(y-1)];
        }
      }
      _Pragma("omp master")
      grid[0*n+0] = -grid[(n-1)*n+(n-1)];
    }
    _Pragma("omp barrier")
    _Pragma("omp master")
    pipeline_time = prk_wtime() - pipeline_time;
  }

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  const double epsilon = 1.e-8;
  const double corner_val = ((iterations+1.)*(n+n-2.));
  if ( (fabs(grid[(n-1)*n+(n-1)] - corner_val)/corner_val) > epsilon) {
    printf("ERROR: checksum %lf does not match verification value %lf\n", grid[(n-1)*n+(n-1)], corner_val);
    return 1;
  }

  prk_free(grid);

#ifdef VERBOSE
  printf("Solution validates; verification value = %lf\n", corner_val );
#else
  printf("Solution validates\n" );
#endif
  double avgtime = pipeline_time/iterations;
  printf("Rate (MFlops/s): %lf Avg time (s): %lf\n", 2.0e-6 * ( (n-1)*(n-1) )/avgtime, avgtime );

  return 0;
}
