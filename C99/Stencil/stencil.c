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

NAME:    Stencil

PURPOSE: This program tests the efficiency with which a space-invariant,
         linear, symmetric filter (stencil) can be applied to a square
         grid or image.
  
USAGE:   The program takes as input the linear
         dimension of the grid, and the number of iterations on the grid

               <progname> <iterations> <grid size> 
  
         The output consists of diagnostics to make sure the 
         algorithm worked, and of timing statistics.

FUNCTIONS CALLED:

         Other than standard C functions, the following functions are used in 
         this program:
         wtime()

HISTORY: - Written by Rob Van der Wijngaart, February 2009.
         - RvdW: Removed unrolling pragmas for clarity;
           added constant to array "in" at end of each iteration to force 
           refreshing of neighbor data in parallel versions; August 2013
  
**********************************************************************************/

#include <prk_util.h>
#include <prk_openmp.h>

#include <tgmath.h>

#ifdef DOUBLE
typedef double prk_float_t;
const double epsilon = 1.0e-8;
#else
typedef float prk_float_t;
/* error computed in double to avoid round-off issues in reduction */
const double epsilon = 1.0e-4;
#endif

const int radius = RADIUS;

int main(int argc, char * argv[])
{
  printf("Parallel Research Kernels Version %s\n", PRKVERSION);
  printf("Serial stencil execution on 2D grid\n");

  /*******************************************************************************
  ** process and test input parameters
  ********************************************************************************/

  if (argc != 3 && argc !=4){
    printf("Usage: %s <# iterations> <array dimension> [tilesize]\n", argv[0]);
    return(EXIT_FAILURE);
  }

  int iterations  = atoi(argv[1]); /* number of times to run the algorithm */
  if (iterations < 1){
    printf("ERROR: iterations must be >= 1 : %d \n",iterations);
    exit(EXIT_FAILURE);
  }

  int n  = atoi(argv[2]); /* linear grid dimension */
  if (n < 1){
    printf("ERROR: grid dimension must be positive: %d\n", n);
    exit(EXIT_FAILURE);
  }

  int tilesize = 1; /* loop nest block factor */
  if (argc == 4) {
    tilesize = atoi(argv[3]);
    if (tilesize>n)  tilesize=n;
    if (tilesize<=0) tilesize=1;
  }

  if (radius < 1) {
    printf("ERROR: Stencil radius %d should be positive\n", radius);
    exit(EXIT_FAILURE);
  }

  if (2*radius+1 > n) {
    printf("ERROR: Stencil radius %d exceeds grid size %d\n", radius, n);
    exit(EXIT_FAILURE);
  }

  size_t bytes = (size_t)n*(size_t)n*sizeof(prk_float_t);
  prk_float_t (* const restrict in)[n]  = (prk_float_t (*)[n]) prk_malloc(bytes); /* input grid values  */
  prk_float_t (* const restrict out)[n] = (prk_float_t (*)[n]) prk_malloc(bytes); /* output grid values */
  if (in==NULL || out==NULL) {
    printf("ERROR: could not allocate space for input or output array\n");
    exit(EXIT_FAILURE);
  }

  prk_float_t weight[2*radius+1][2*radius+1]; /* weights of points in the stencil     */
  /* fill the stencil weights to reflect a discrete divergence operator         */
  for (int jj=-radius; jj<=radius; jj++) {
    for (int ii=-radius; ii<=radius; ii++) {
      weight[ii+radius][jj+radius]= (prk_float_t)0;
    }
  }

#ifdef STAR
  const int stencil_size = 4*radius+1;
  for (int ii=1; ii<=radius; ii++) {
    weight[radius][radius+ii] = weight[radius+ii][radius] = ((prk_float_t)+1)/(2*ii*radius);
    weight[radius][radius-ii] = weight[radius-ii][radius] = ((prk_float_t)-1)/(2*ii*radius);
  }
#else
  const int stencil_size = (2*radius+1)*(2*radius+1);
  for (int jj=1; jj<=radius; jj++) {
    for (int ii=-jj+1; ii<jj; ii++) {
      weight[radius+ii][radius+jj]  = ((prk_float_t)+1)/(4*jj*(2*jj-1)*radius);
      weight[radius+ii][radius-jj]  = ((prk_float_t)-1)/(4*jj*(2*jj-1)*radius);
      weight[radius+jj][radius+ii]  = ((prk_float_t)+1)/(4*jj*(2*jj-1)*radius);
      weight[radius-jj][radius+ii]  = ((prk_float_t)-1)/(4*jj*(2*jj-1)*radius);
    }
    weight[radius+jj][radius+jj]    = ((prk_float_t)+1)/(4*jj*radius);
    weight[radius-jj][radius-jj]    = ((prk_float_t)-1)/(4*jj*radius);
  }
#endif

  /* interior of grid with respect to stencil */
  size_t active_points = ((size_t)(n-2*radius))*((size_t)(n-2*radius));

#ifdef _OPENMP
  printf("Number of threads     = %d\n", omp_get_max_threads());
#endif
  printf("Grid size            = %d\n", n);
  printf("Radius of stencil    = %d\n", radius);
#ifdef STAR
  printf("Type of stencil      = star\n");
#else
  printf("Type of stencil      = compact\n");
#endif
#ifdef DOUBLE
  printf("Data type            = double precision\n");
#else
  printf("Data type            = single precision\n");
#endif
  printf("Compact representation of stencil loop body\n");
  if ((1<tilesize) && (tilesize<n)) {
      printf("Tile size            = %d\n", tilesize);
  } else {
      printf("Untiled\n");
  }
  printf("Number of iterations = %d\n", iterations);

  double stencil_time = 0.0; /* silence compiler warning */

  OMP_PARALLEL(shared(in,out) firstprivate(weight))
  {
    /* initialize the input and output arrays */
    OMP_FOR()
    for (int i=0; i<n; i++) {
      OMP_SIMD()
      for (int j=0; j<n; j++) {
        in[i][j] = (prk_float_t)i + (prk_float_t)j;
      }
    }
    {
      const prk_float_t zero = (prk_float_t)0;
      OMP_FOR()
      for (int i=radius; i<n-radius; i++) {
        OMP_SIMD()
        for (int j=radius; j<n-radius; j++) {
          out[i][j] = zero;
        }
      }
    }

    for (int iter = 0; iter<=iterations; iter++) {

      /* start timer after a warmup iteration */
      if (iter==1) {
        OMP_BARRIER
        OMP_MASTER
        { stencil_time = wtime(); }
      }

      /* Apply the stencil operator */
      if ((tilesize==1) || (tilesize==n)) {
        OMP_FOR()
        for (int i=radius; i<n-radius; i++) {
          for (int j=radius; j<n-radius; j++) {
            #ifdef STAR
                OMP_SIMD()
                for (int jj=-radius; jj<=radius; jj++) {
                  out[i][j] += weight[radius][radius+jj]*in[i][j+jj];
                }
                OMP_SIMD()
                for (int ii=-radius; ii<0; ii++) {
                  out[i][j] += weight[radius+ii][radius]*in[i+ii][j];
                }
                OMP_SIMD()
                for (int ii=1; ii<=radius; ii++) {
                  out[i][j] += weight[radius+ii][radius]*in[i+ii][j];
                }
            #else
                OMP_SIMD()
                for (int ii=-radius; ii<=radius; ii++) {
                  OMP_SIMD()
                  for (int jj=-radius; jj<=radius; jj++) {
                    out[i][j] += weight[radius+ii][radius+jj]*in[i+ii][j+jj];
                  }
                }
            #endif
          }
        }
      } else {
        OMP_FOR()
        for (int it=radius; it<n-radius; it+=tilesize) {
          for (int jt=radius; jt<n-radius; jt+=tilesize) {
            for (int i=it; i<MIN(n-radius,it+tilesize); i++) {
              for (int j=jt; j<MIN(n-radius,jt+tilesize); j++) {
                #ifdef STAR
                  OMP_SIMD()
                  for (int jj=-radius; jj<=radius; jj++) {
                    out[i][j] += weight[radius][radius+jj]*in[i][j+jj];
                  }
                  OMP_SIMD()
                  for (int ii=-radius; ii<0; ii++) {
                    out[i][j] += weight[radius+ii][radius]*in[i+ii][j];
                  }
                  OMP_SIMD()
                  for (int ii=1; ii<=radius; ii++) {
                    out[i][j] += weight[radius+ii][radius]*in[i+ii][j];
                  }
                #else
                  OMP_SIMD()
                  for (int ii=-radius; ii<=radius; ii++) {
                    OMP_SIMD()
                    for (int jj=-radius; jj<=radius; jj++) {
                      out[i][j] += weight[radius+ii][radius+jj]*in[i+ii][j+jj];
                    }
                  }
                #endif
              }
            }
          }
        }
      }

      /* add constant to solution to force refresh of neighbor data, if any       */
      {
          const prk_float_t one = (prk_float_t)1;
          OMP_FOR()
          for (int i=0; i<n; i++) {
            OMP_SIMD()
            for (int j=0; j<n; j++) {
              in[i][j] += one;
            }
          }
      }

    } /* end of iterations */

    OMP_BARRIER
    OMP_MASTER
    { stencil_time = wtime() - stencil_time; }

  } /* end OMP_PARALLEL */

  prk_free(in);

  /*******************************************************************************
  ** Analyze and output results.
  ********************************************************************************/

  /* compute L1 norm in parallel */
  double norm = 0.0; /* the reduction will not be accurate enough to validate
                        if single precision is used here */
  OMP_PARALLEL(shared(norm))
  {
    OMP_FOR(reduction(+:norm))
    for (int i=radius; i<n-radius; i++) {
      for (int j=radius; j<n-radius; j++) {
        norm += (double)fabs(out[i][j]);
      }
    }
  } /* end OMP_PARALLEL */

  norm /= active_points;

  prk_free(out);

  /* verify correctness */
  double reference_norm = 2.*(iterations+1.);
  if (fabs(norm-reference_norm) > epsilon) {
    printf("ERROR: L1 norm = %lf, Reference L1 norm = %lf\n",
           norm, reference_norm);
    exit(EXIT_FAILURE);
  }
  else {
    printf("Solution validates\n");
#ifdef VERBOSE
    printf("Reference L1 norm = %lf, L1 norm = %lf\n",
           reference_norm, norm);
#endif
  }

  size_t flops = (2L*(size_t)stencil_size+1L) * active_points;
  double avgtime = stencil_time/iterations;
  printf("Rate (MFlops/s): %lf  Avg time (s): %lf\n",
         1.0e-6 * (double)flops/avgtime, avgtime);

  exit(EXIT_SUCCESS);

  return 0;
}
