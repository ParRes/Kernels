
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
/// NAME:    Stencil
///
/// PURPOSE: This program tests the efficiency with which a space-invariant,
///          linear, symmetric filter (stencil) can be applied to a square
///          grid or image.
///
/// USAGE:   The program takes as input the linear
///          dimension of the grid, and the number of iterations on the grid
///
///                <progname> <iterations> <grid size>
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

typedef void (*stencil_t)(const int, const int, const double * restrict, double * restrict);

void nothing(const int n, const int gs, const double * restrict in, double * restrict out)
{
    printf("You are trying to use a stencil that does not exist.\n");
    printf("Please generate the new stencil using the code generator.\n");
    // n will never be zero - this is to silence compiler warnings.
    if (n==0 || gs==0) printf("%p %p\n", in, out);
    abort();
}

#ifdef _OPENMP
#include "stencil_taskloop.h"
#else
#include "stencil_seq.h"
#endif

int main(int argc, char * argv[])
{
  printf("Parallel Research Kernels version %.2f\n", PRKVERSION);
#ifdef _OPENMP
  printf("C11/OpenMP TASKLOOP Stencil execution on 2D grid\n");
#else
  printf("C11/Serial Stencil execution on 2D grid\n");
#endif

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  if (argc < 3){
    printf("Usage: <# iterations> <array dimension> [<taskloop grainsize> <star/grid> <radius>]\n");
    return 1;
  }

  // number of times to run the algorithm
  int iterations  = atoi(argv[1]);
  if (iterations < 1) {
    printf("ERROR: iterations must be >= 1\n");
    return 1;
  }

  // linear grid dimension
  int n  = atoi(argv[2]);
  if (n < 1) {
    printf("ERROR: grid dimension must be positive\n");
    return 1;
  } else if (n > floor(sqrt(INT_MAX))) {
    printf("ERROR: grid dimension too large - overflow risk\n");
    return 1;
  }

  // taskloop grainsize
  int gs = (argc > 3) ? atoi(argv[3]) : 100;

  // stencil pattern
  bool star = true;
  if (argc > 4) {
      char* pattern = argv[4];
      star = (0==strncmp(pattern,"star",4)) ? true : false;
  }

  // stencil radius
  int radius = 2;
  if (argc > 5) {
      radius = atoi(argv[5]);
  }

  if ( (radius < 1) || (2*radius+1 > n) ) {
    printf("ERROR: Stencil radius negative or too large\n");
    return 1;
  }

#ifdef _OPENMP
  printf("Number of threads (max)   = %d\n", omp_get_max_threads());
  printf("Taskloop grainsize        = %d\n", gs);
#endif
  printf("Number of iterations      = %d\n", iterations);
  printf("Grid sizes                = %d\n", n);
  printf("Type of stencil           = %s\n", (star ? "star" : "grid") );
  printf("Radius of stencil         = %d\n", radius );

  stencil_t stencil = nothing;
  if (star) {
      switch (radius) {
          case 1: stencil = star1; break;
          case 2: stencil = star2; break;
          case 3: stencil = star3; break;
          case 4: stencil = star4; break;
          case 5: stencil = star5; break;
          case 6: stencil = star6; break;
          case 7: stencil = star7; break;
          case 8: stencil = star8; break;
          case 9: stencil = star9; break;
      }
  } else {
      switch (radius) {
          case 1: stencil = grid1; break;
          case 2: stencil = grid2; break;
          case 3: stencil = grid3; break;
          case 4: stencil = grid4; break;
          case 5: stencil = grid5; break;
          case 6: stencil = grid6; break;
          case 7: stencil = grid7; break;
          case 8: stencil = grid8; break;
          case 9: stencil = grid9; break;
      }
  }

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double stencil_time = 0.0;

  // interior of grid with respect to stencil
  size_t active_points = (n-2*radius)*(n-2*radius);
  size_t bytes = n*n*sizeof(double);

  double * restrict in  = prk_malloc(bytes);
  double * restrict out = prk_malloc(bytes);

  OMP_PARALLEL()
  OMP_MASTER
  {
    OMP_TASKLOOP( firstprivate(n) shared(in,out) grainsize(gs) )
    for (int i=0; i<n; i++) {
      OMP_SIMD
      for (int j=0; j<n; j++) {
        in[i*n+j]  = (double)(i+j);
        out[i*n+j] = 0.0;
      }
    }
    OMP_TASKWAIT

    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) stencil_time = prk_wtime();

      // Apply the stencil operator
      stencil(n, gs, in, out);
      OMP_TASKWAIT

      // Add constant to solution to force refresh of neighbor data, if any
      OMP_TASKLOOP( firstprivate(n) shared(in,out) grainsize(gs) )
      for (int i=0; i<n; i++) {
        OMP_SIMD
        for (int j=0; j<n; j++) {
          in[i*n+j] += 1.0;
        }
      }
      OMP_TASKWAIT
    }
    stencil_time = prk_wtime() - stencil_time;
  }

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  // compute L1 norm in parallel
  double norm = 0.0;
  OMP_PARALLEL_FOR_REDUCE( +:norm )
  for (int i=radius; i<n-radius; i++) {
    for (int j=radius; j<n-radius; j++) {
      norm += fabs(out[i*n+j]);
    }
  }
  norm /= active_points;

  prk_free(in);
  prk_free(out);

  // verify correctness
  const double epsilon = 1.0e-8;
  double reference_norm = 2.*(iterations+1.);
  if (fabs(norm-reference_norm) > epsilon) {
    printf("ERROR: L1 norm = %lf Reference L1 norm = %lf\n", norm, reference_norm);
    return 1;
  } else {
    printf("Solution validates\n");
#ifdef VERBOSE
    printf("L1 norm = %lf Reference L1 norm = %lf\n", norm, reference_norm);
#endif
    const int stencil_size = star ? 4*radius+1 : (2*radius+1)*(2*radius+1);
    size_t flops = (2*stencil_size+1) * active_points;
    double avgtime = stencil_time/iterations;
    printf("Rate (MFlops/s): %lf Avg time (s): %lf\n", 1.0e-6 * (double)flops/avgtime, avgtime );
  }

  return 0;
}
