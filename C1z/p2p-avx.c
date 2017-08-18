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

#include "immintrin.h"

#if 1
void print_m256d(const char * label, __m256d r)
{
  double d[4];
  _mm256_store_pd(d, r);
  printf("%s = {%f,%f,%f,%f}\n", label, d[0], d[1], d[2], d[3]);
}
#endif

static inline void sweep_tile(int startm, int endm,
                              int startn, int endn,
                              int n, double g[])
{
  const __m256d zero  = _mm256_setzero_pd();
  const __m256d ones  = _mm256_cmp_pd( _mm256_setzero_pd() , _mm256_setzero_pd() , _CMP_EQ_OQ);
  const __m256d el1   = _mm256_castsi256_pd( _mm256_set_epi8(0,0,0,0,0,0,0,0,
                                                             0,0,0,0,0,0,0,0,
                                                             0,0,0,0,0,0,0,0,
                                                             255,255,255,255,255,255,255,255) );
  const __m256d el12  = _mm256_castsi256_pd( _mm256_set_epi8(0,0,0,0,0,0,0,0,
                                                             0,0,0,0,0,0,0,0,
                                                             255,255,255,255,255,255,255,255,
                                                             255,255,255,255,255,255,255,255) );
  const __m256i mask  = _mm256_set_epi64x(0,0,0,-1);
  for (int i=startm; i<endm; i++) {
    int j;
    for (j=startn; j<endn-3; j+=4) {
      //g[i*n+j] = g[(i-1)*n+j] + g[i*n+(j-1)] - g[(i-1)*n+(j-1)];
#if 0
      // NO UNROLLING
      __m256d c0 = _mm256_load_pd( &( g[(i-1)*n+(j-1)] ) ); // { g[i-1][j-1] , g[i-1][ j ] , g[i-1][j+1] , g[i-1][j+2] }
      __m256d c1 = _mm256_load_pd( &( g[  i  *n+(j-1)] ) ); // { g[ i ][j-1] , g[ i ][ j ] , g[ i ][j+1] , g[ i ][j+2] }
      __m256d j1 = _mm256_and_pd( c1 , el1 );               // { g[ i ][j-1] , 0 , 0 , 0 }
      __m256d i0 = _mm256_addsub_pd( j1 , c0 );             // { g[ i ][j-1] - g[i-1][j-1] , g[i-1][ j ] , .. }
      __m256d i1 = _mm256_and_pd( i0 , el12 );              // { g[ i ][j-1] - g[i-1][j-1] , g[ i ][ j ] + g[i-1][ j ] , 0 , 0 }
      __m256d i2 = _mm256_hadd_pd( i1 , zero );             // { g[ i ][j-1] - g[i-1][j-1] + g[ i ][ j ] + g[i-1][ j ] , .. }
      _mm256_maskstore_pd( &( g[i*n+j] ) , mask, i2 );      // g[i][j] = { g[i][j-1] - g[i-1][j-1] + g[i][j] + g[i-1][j] }
#elif 0
      // NO UNROLLING
      __m256d c0 = _mm256_load_pd( &( g[(i-1)*n+(j-1)] ) ); // { g[i-1][j-1] , g[i-1][ j ] , g[i-1][j+1] , g[i-1][j+2] }
      __m256d c1 = _mm256_load_pd( &( g[  i  *n+(j-1)] ) ); // { g[ i ][j-1] , g[ i ][ j ] , g[ i ][j+1] , g[ i ][j+2] }
      __m256d j1 = _mm256_and_pd( c1 , el1 );               // { g[ i ][j-1] , 0 , 0 , 0 }
      __m256d i0 = _mm256_addsub_pd( j1 , c0 );             // { g[ i ][j-1] - g[i-1][j-1] , g[i-1][ j ] , .. }
      __m256d i1 = _mm256_and_pd( i0 , el12 );              // { g[ i ][j-1] - g[i-1][j-1] , g[ i ][ j ] + g[i-1][ j ] , 0 , 0 }
      __m256d i2 = _mm256_hadd_pd( i1 , zero );             // { g[ i ][j-1] - g[i-1][j-1] + g[ i ][ j ] + g[i-1][ j ] , .. }
      _mm256_maskstore_pd( &( g[i*n+j] ) , mask, i2 );      // g[i][j] = { g[i][j-1] - g[i-1][j-1] + g[i][j] + g[i-1][j] }
#elif 0
      // NO UNROLLING
      double c0[4] = { g[(i-1)*n+(j-1)] , g[(i-1)*n+(j+0)] , g[(i-1)*n+(j+1)] , g[(i-1)*n+(j+2)] };
      double c1[4] = { g[  i  *n+(j-1)] , g[  i  *n+(j+0)] , g[  i  *n+(j+1)] , g[  i  *n+(j+2)] };
      double j1[4] = { c1[0] , 0 , 0 , 0 };
      double i0[4] = { j1[0] - c0[0] , j1[1] + c0[1] , j1[2] - c0[2] , j1[3] + c0[3] };
      double i1[4] = { i0[0] , i0[1] , 0 , 0 };
      double i2[4] = { i1[0] + i1[1] , 0 , i1[2] + i1[3] , 0 };
      g[i*n+j] = i2[0];
#elif 1
      // WORKS
      double c0s[4] = { g[(i-1)*n+(j+0)] , g[(i-1)*n+(j+1)] , g[(i-1)*n+(j+2)] }; // shifted
      double c0r[4] = { g[(i-1)*n+(j-1)] , g[(i-1)*n+(j+0)] , g[(i-1)*n+(j+1)] }; // regular
      double i0[4]  = { c0s[0] - c0r[0] , c0s[1] - c0r[1] ,c0s[2] - c0r[2] ,c0s[3] - c0r[3] }; // subtract
      double c1[4]  = { g[  i  *n+(j-1)] , g[  i  *n+(j+0)] , g[  i  *n+(j+1)] , g[  i  *n+(j+2)] }; // regular
      double i1[4]  = { c1[0] + i0[0] , 0 , 0 };        // add first element
      double i2[4]  = { 0 , i1[0] , 0 , 0 };            // shift right
      double i3[4]  = { 0 , i2[1] + i0[1] , 0 , 0 };    // add second element
      double i4[4]  = { 0 , 0 , i3[2] , 0 };            // shift right
      double i5[4]  = { 0 , 0 , i4[2] + i0[2] , 0 };    // add third element
      double i6[4]  = { 0 , 0 , 0 , i5[2] };            // shift right
      double i7[4]  = { 0 , 0 , 0 , i6[3] + i0[3] };    // add fourth element
      g[i*n+j+0] = i1[0];
      g[i*n+j+1] = i3[1];
      g[i*n+j+2] = i5[2];
      g[i*n+j+3] = i7[3];
      //printf("g[%d][%d]=%f\n",i,j+0,g[i*n+j+0]);
      //printf("g[%d][%d]=%f\n",i,j+1,g[i*n+j+1]);
      //printf("g[%d][%d]=%f\n",i,j+2,g[i*n+j+2]);
      //printf("g[%d][%d]=%f\n",i,j+3,g[i*n+j+3]);
#else
      // WORKS
      g[i*n+j] = g[i*n+j-1] + g[(i-1)*n+j] - g[(i-1)*n+j-1];
      //printf("g[%d][%d]=%f\n",i,j,g[i*n+j]);
      g[i*n+j+1] = g[i*n+j] + g[(i-1)*n+j+1] - g[(i-1)*n+j];
      //printf("g[%d][%d]=%f\n",i,j,g[i*n+j]);
      g[i*n+j+2] = g[i*n+j+1] + g[(i-1)*n+j+2] - g[(i-1)*n+j+1];
      //printf("g[%d][%d]=%f\n",i,j,g[i*n+j]);
      g[i*n+j+3] = g[i*n+j+2] + g[(i-1)*n+j+3] - g[(i-1)*n+j+2];
      //printf("g[%d][%d]=%f\n",i,j,g[i*n+j]);
#endif
    }
    for (int jj=j; j<endn; j++) {
      g[i*n+j] = g[i*n+j-1] + g[(i-1)*n+j] - g[(i-1)*n+j-1];
      //printf("g[%d][%d]=%f\n",i,j,g[i*n+j]);
    }
  }
}

int main(int argc, char * argv[])
{
  printf("Parallel Research Kernels version %.2f\n", PRKVERSION);
  printf("C11 pipeline execution on 2D grid\n");

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

  printf("Number of iterations      = %d\n", iterations);
  printf("Grid sizes                = %d,%d\n", m, n);
  printf("Grid chunk sizes          = %d,%d\n", mc, nc);

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double pipeline_time = 0.0; // silence compiler warning

  size_t bytes = m*n*sizeof(double);
  double * restrict grid = prk_malloc(bytes);

  {
    for (int i=0; i<m; i++) {
      for (int j=0; j<n; j++) {
        grid[i*n+j] = 0.0;
      }
    }
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
          sweep_tile(i, MIN(m,i+mc), j, MIN(n,j+nc), n, grid);
        }
      }
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
