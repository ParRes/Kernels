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
///          a pipelined algorithm on an n^2 grid. The first array dimension
///          is distributed among the threads (stripwise decomposition).
///
/// USAGE:   The program takes as input the
///          dimensions of the grid, and the number of iterations on the grid
///
///                <progname> <iterations> <n>
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

int main(int argc, char* argv[])
{
  printf("Parallel Research Kernels version %.2f\n", PRKVERSION);
#ifdef _OPENMP
  printf("C11/OpenMP HYPERPLANE pipeline execution on 2D grid\n");
#else
  printf("C11/Serial HYPERPLANE pipeline execution on 2D grid\n");
#endif

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  if (argc < 3) {
    printf("Usage: <# iterations> <array dimension> <chunk size>\n");
    return 1;
  }

  // number of times to run the pipeline algorithm
  int iterations = atoi(argv[1]);
  if (iterations < 1) {
    printf("ERROR: iterations must be >= 1\n");
    return 1;
  }

  // grid dimensions
  int n = atoi(argv[2]);
  if (n < 1) {
    printf("ERROR: grid dimension must be positive: %d\n", n);
    return 1;
  }

  // grid chunk dimensions
  int nc = (argc > 3) ? atoi(argv[3]) : 1;
  nc = MAX(1,nc);
  nc = MIN(n,nc);

  // number of grid blocks
  int nb = (n-1)/nc;
  if ((n-1)%nc) nb++;

#ifdef _OPENMP
  printf("Number of threads (max)   = %d\n", omp_get_max_threads());
#endif
  printf("Number of iterations      = %d\n", iterations);
  printf("Grid sizes                = %d,%d\n", n, n);
  printf("Grid chunk sizes, blocks  = %d,%d\n", nc, nb);

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double pipeline_time = 0.0; // silence compiler warning

  size_t bytes = n*n*sizeof(double);
  double * restrict grid = prk_malloc(bytes);

  OMP_PARALLEL()
  {
    OMP_FOR()
    for (int i=0; i<n; i++) {
      OMP_SIMD
      for (int j=0; j<n; j++) {
        grid[i*n+j] = 0.0;
      }
    }

    // set boundary values (bottom and left side of grid)
    OMP_MASTER
    {
      for (int j=0; j<n; j++) {
        grid[0*n+j] = (double)j;
      }
      for (int i=0; i<n; i++) {
        grid[i*n+0] = (double)i;
      }
    }
    OMP_BARRIER

    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) {
          OMP_BARRIER
          OMP_MASTER
          pipeline_time = prk_wtime();
      }

      if (nc==1) {
        for (int i=2; i<=2*n-2; i++) {
          OMP_FOR_SIMD()
          for (int j=MAX(2,i-n+2); j<=MIN(i,n); j++) {
            const int x = i-j+1;
            const int y = j-1;
            grid[x*n+y] = grid[(x-1)*n+y] + grid[x*n+(y-1)] - grid[(x-1)*n+(y-1)];
          }
        }
      } else {
        for (int i=2; i<=2*(nb+1)-2; i++) {
          OMP_FOR()
          for (int j=MAX(2,i-(nb+1)+2); j<=MIN(i,nb+1); j++) {
            const int ib = nc*(i-j+1-1)+1;
            const int jb = nc*(j-1-1)+1;
            sweep_tile(ib, MIN(n,ib+nc), jb, MIN(n,jb+nc), n, grid);
          }
        }
      }
      OMP_MASTER
      grid[0*n+0] = -grid[(n-1)*n+(n-1)];
    }
    OMP_BARRIER
    OMP_MASTER
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
