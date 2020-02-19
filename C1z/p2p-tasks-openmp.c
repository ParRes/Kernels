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
///          a pipelined algorithm on an m*n grid. The first array dimension
///          is distributed among the threads (stripwise decomposition).
///
/// USAGE:   The program takes as input the
///          dimensions of the grid, and the number of iterations on the grid
///
///                <progname> <iterations> <m> <n>
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
#include "p2p-kernel.h"

int main(int argc, char * argv[])
{
  printf("Parallel Research Kernels version %.2f\n", PRKVERSION);
#ifdef _OPENMP
  printf("C11/OpenMP TASKS pipeline execution on 2D grid\n");
#else
  printf("C11/Serial pipeline execution on 2D grid\n");
#endif

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  if (argc < 4) {
    printf("Usage: <# iterations> <first array dimension> <second array dimension>"
           " [<first chunk dimension> <second chunk dimension>]\n");
    return 1;
  }

  // number of times to run the pipeline algorithm
  int iterations = atoi(argv[1]);
  if (iterations < 1) {
    printf("ERROR: iterations must be >= 1\n");
    return 1;
  }

  // grid dimensions
  int m = atol(argv[2]);
  int n = atol(argv[3]);
  if (m < 1 || n < 1) {
    printf("ERROR: grid dimensions must be positive: %d,%d\n", m, n);
    return 1;
  }

  // grid chunk dimensions
  int mc = (argc > 4) ? atol(argv[4]) : m;
  int nc = (argc > 5) ? atol(argv[5]) : n;
  if (mc < 1 || mc > m || nc < 1 || nc > n) {
    printf("WARNING: grid chunk dimensions invalid: %d,%d (ignoring)\n", mc, nc);
    mc = m;
    nc = n;
  }

#ifdef _OPENMP
  printf("Number of threads (max)   = %d\n", omp_get_max_threads());
#endif
  printf("Number of iterations      = %d\n", iterations);
  printf("Grid sizes                = %d,%d\n", m, n);
  printf("Grid chunk sizes          = %d,%d\n", mc, nc);

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double pipeline_time = 0.0; // silence compiler warning

  size_t bytes = m*n*sizeof(double);
  double * restrict grid = prk_malloc(bytes);

  OMP_PARALLEL()
  OMP_MASTER
  {
    OMP_TASKLOOP( firstprivate(n) shared(grid) )
    for (int i=0; i<m; i++) {
      for (int j=0; j<n; j++) {
        grid[i*n+j] = 0.0;
      }
    }
    OMP_TASKWAIT

    for (int j=0; j<n; j++) {
      grid[0*n+j] = (double)j;
    }
    for (int i=0; i<m; i++) {
      grid[i*n+0] = (double)i;
    }

    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) pipeline_time = prk_wtime();

      for (int i=1; i<m; i+=mc) {
        for (int j=1; j<n; j+=nc) {
          OMP_TASK( depend(in:grid[(i-mc)*n+j],grid[i*n+(j-nc)]) depend(out:grid[i*n+j]) )
          sweep_tile(i, MIN(m,i+mc), j, MIN(n,j+nc), n, grid);
        }
      }
      OMP_TASKWAIT
      grid[0*n+0] = -grid[(m-1)*n+(n-1)];
    }
    pipeline_time = prk_wtime() - pipeline_time;
  }

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  const double epsilon = 1.e-8;
  const double corner_val = ((iterations+1.)*(n+m-2.));
  if ( (fabs(grid[(m-1)*n+(n-1)] - corner_val)/corner_val) > epsilon) {
    printf("ERROR: checksum %lf does not match verification value %lf\n", grid[(m-1)*n+(n-1)], corner_val);
    return 1;
  }

  prk_free(grid);

#ifdef VERBOSE
  printf("Solution validates; verification value = %lf\n", corner_val );
#else
  printf("Solution validates\n" );
#endif
  double avgtime = pipeline_time/iterations;
  printf("Rate (MFlops/s): %lf Avg time (s): %lf\n", 2.0e-6 * ( (m-1)*(n-1) )/avgtime, avgtime );

  return 0;
}
