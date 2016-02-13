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

#include <par-res-kern_general.h>

#ifdef DOUBLE
  #define DTYPE   double
  #define EPSILON 1.e-8
  #define COEFX   1.0
  #define COEFY   1.0
  #define FSTR    "%lf"
#else
  #define DTYPE   float
  #define EPSILON 0.0001f
  #define COEFX   1.0f
  #define COEFY   1.0f
  #define FSTR    "%f"
#endif

/* define shorthand for indexing a multi-dimensional array                       */
#define IN(i,j)       in[i+(j)*(n)]
#define OUT(i,j)      out[i+(j)*(n)]
#define WEIGHT(ii,jj) weight[ii+RADIUS][jj+RADIUS]

int main(int argc, char ** argv)
{
  printf("Parallel Research Kernels Version %s\n", PRKVERSION);
  printf("Serial stencil execution on 2D grid\n");

  /*******************************************************************************
  ** process and test input parameters
  ********************************************************************************/

  if (argc != 3 && argc !=4){
    printf("Usage: %s <# iterations> <array dimension> [tilesize]\n",
           *argv);
    return(EXIT_FAILURE);
  }

  int iterations  = atoi(*++argv); /* number of times to run the algorithm */
  if (iterations < 1){
    printf("ERROR: iterations must be >= 1 : %d \n",iterations);
    exit(EXIT_FAILURE);
  }

  int n  = atoi(*++argv); /* linear grid dimension */
  if (n < 1){
    printf("ERROR: grid dimension must be positive: %d\n", n);
    exit(EXIT_FAILURE);
  }

  int tilesize = 1; /* loop nest block factor */
  if (argc == 4) {
    tilesize = atoi(*++argv);
    if (tilesize>n)  tilesize=n;
    if (tilesize<=0) tilesize=1;
  }

  if (RADIUS < 1) {
    printf("ERROR: Stencil radius %d should be positive\n", RADIUS);
    exit(EXIT_FAILURE);
  }

  if (2*RADIUS +1 > n) {
    printf("ERROR: Stencil radius %d exceeds grid size %d\n", RADIUS, n);
    exit(EXIT_FAILURE);
  }

  size_t bytes = (size_t)n*(size_t)n*sizeof(DTYPE);
  DTYPE * restrict in  = (DTYPE *) prk_malloc(bytes); /* input grid values  */
  DTYPE * restrict out = (DTYPE *) prk_malloc(bytes); /* output grid values */
  if (!in || !out) {
    printf("ERROR: could not allocate space for input or output array\n");
    exit(EXIT_FAILURE);
  }

  DTYPE weight[2*RADIUS+1][2*RADIUS+1]; /* weights of points in the stencil     */
  /* fill the stencil weights to reflect a discrete divergence operator         */
  for (int jj=-RADIUS; jj<=RADIUS; jj++) {
      for (int ii=-RADIUS; ii<=RADIUS; ii++) {
          WEIGHT(ii,jj) = (DTYPE) 0.0;
      }
  }

  int stencil_size; /* number of points in stencil */
#ifdef STAR
  stencil_size = 4*RADIUS+1;
  for (int ii=1; ii<=RADIUS; ii++) {
    WEIGHT(0, ii) = WEIGHT( ii,0) =  (DTYPE) (1.0/(2.0*ii*RADIUS));
    WEIGHT(0,-ii) = WEIGHT(-ii,0) = -(DTYPE) (1.0/(2.0*ii*RADIUS));
  }
#else
  stencil_size = (2*RADIUS+1)*(2*RADIUS+1);
  for (int jj=1; jj<=RADIUS; jj++) {
    for (int ii=-jj+1; ii<jj; ii++) {
      WEIGHT(ii,jj)  =  (DTYPE) (1.0/(4.0*jj*(2.0*jj-1)*RADIUS));
      WEIGHT(ii,-jj) = -(DTYPE) (1.0/(4.0*jj*(2.0*jj-1)*RADIUS));
      WEIGHT(jj,ii)  =  (DTYPE) (1.0/(4.0*jj*(2.0*jj-1)*RADIUS));
      WEIGHT(-jj,ii) = -(DTYPE) (1.0/(4.0*jj*(2.0*jj-1)*RADIUS));
    }
    WEIGHT(jj,jj)    =  (DTYPE) (1.0/(4.0*jj*RADIUS));
    WEIGHT(-jj,-jj)  = -(DTYPE) (1.0/(4.0*jj*RADIUS));
  }
#endif

  size_t active_points = ((size_t)n-2L*RADIUS)*((size_t)n-2L*RADIUS); /* interior of grid with respect to stencil */

  printf("Grid size            = %d\n", n);
  printf("Radius of stencil    = %d\n", RADIUS);
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
#if LOOPGEN
  printf("Script used to expand stencil loop body\n");
#else
  printf("Compact representation of stencil loop body\n");
#endif
  if (1<tilesize && tilesize<n) {
      printf("Tile size            = %d\n", tilesize);
  } else {
      printf("Untiled\n");
  }
  printf("Number of iterations = %d\n", iterations);

  /* intialize the input and output arrays                                     */
  for (int j=0; j<n; j++) {
    for (int i=0; i<n; i++) {
      IN(i,j) = COEFX*i+COEFY*j;
    }
  }
  for (int j=RADIUS; j<n-RADIUS; j++) {
    for (int i=RADIUS; i<n-RADIUS; i++) {
      OUT(i,j) = (DTYPE)0.0;
    }
  }

  double stencil_time = 0.0; /* silence compiler warning */

  for (int iter = 0; iter<=iterations; iter++) {

    /* start timer after a warmup iteration */
    if (iter == 1)  stencil_time = wtime();

    /* Apply the stencil operator */

    if (tilesize==1 || tilesize==n) {
      for (int j=RADIUS; j<n-RADIUS; j++) {
        for (int i=RADIUS; i<n-RADIUS; i++) {
          #ifdef STAR
            #if LOOPGEN
              #include "loop_body_star.incl"
            #else
              for (int jj=-RADIUS; jj<=RADIUS; jj++) {
                OUT(i,j) += WEIGHT(0,jj)*IN(i,j+jj);
              }
              for (int ii=-RADIUS; ii<0; ii++) {
                OUT(i,j) += WEIGHT(ii,0)*IN(i+ii,j);
              }
              for (int ii=1; ii<=RADIUS; ii++) { 
                OUT(i,j) += WEIGHT(ii,0)*IN(i+ii,j);
              }
            #endif
          #else 
            #if LOOPGEN
              #include "loop_body_compact.incl"
            #else
              /* would like to be able to unroll this loop, but compiler will ignore  */
              for (int jj=-RADIUS; jj<=RADIUS; jj++) {
                for (int ii=-RADIUS; ii<=RADIUS; ii++) {
                  OUT(i,j) += WEIGHT(ii,jj)*IN(i+ii,j+jj);
                }
              }
            #endif
          #endif
        }
      }
    } else {
      for (int jt=RADIUS; jt<n-RADIUS; jt+=tilesize) {
        for (int it=RADIUS; it<n-RADIUS; it+=tilesize) {
          for (int j=jt; j<MIN(n-RADIUS,jt+tilesize); j++) {
            for (int i=it; i<MIN(n-RADIUS,it+tilesize); i++) {
              #ifdef STAR
                #if LOOPGEN
                  #include "loop_body_star.incl"
                #else
                  for (int jj=-RADIUS; jj<=RADIUS; jj++) {
                    OUT(i,j) += WEIGHT(0,jj)*IN(i,j+jj);
                  }
                  for (int ii=-RADIUS; ii<0; ii++) {
                    OUT(i,j) += WEIGHT(ii,0)*IN(i+ii,j);
                  }
                  for (int ii=1; ii<=RADIUS; ii++) {
                    OUT(i,j) += WEIGHT(ii,0)*IN(i+ii,j);
                  }
                #endif
              #else
                #if LOOPGEN
                  #include "loop_body_compact.incl"
                #else
                  /* would like to be able to unroll this loop, but compiler will ignore  */
                  for (int jj=-RADIUS; jj<=RADIUS; jj++) {
                    for (int ii=-RADIUS; ii<=RADIUS; ii++) {
                      OUT(i,j) += WEIGHT(ii,jj)*IN(i+ii,j+jj);
                    }
                  }
                #endif
              #endif
            }
          }
        }
      }
    }

    /* add constant to solution to force refresh of neighbor data, if any       */
    for (int j=0; j<n; j++) {
      for (int i=0; i<n; i++) {
        IN(i,j)+= 1.0;
      }
    }

  } /* end of iterations                                                        */

  stencil_time = wtime() - stencil_time;

  /* compute L1 norm in parallel                                                */
  DTYPE norm = 0; /* L1 norm of solution */
  for (int j=RADIUS; j<n-RADIUS; j++) {
    for (int i=RADIUS; i<n-RADIUS; i++) {
      norm += (DTYPE)ABS(OUT(i,j));
    }
  }

  norm /= active_points;

  /*******************************************************************************
  ** Analyze and output results.
  ********************************************************************************/

  /* verify correctness */
  DTYPE reference_norm = (DTYPE) (iterations+1) * (COEFX + COEFY);
  if (ABS(norm-reference_norm) > EPSILON) {
    printf("ERROR: L1 norm = %lf, Reference L1 norm = %lf\n",
           (double)norm, (double)reference_norm);
    exit(EXIT_FAILURE);
  }
  else {
    printf("Solution validates\n");
#ifdef VERBOSE
    printf("Reference L1 norm = %lf, L1 norm = %lf\n",
           (double)reference_norm, (double)norm);
#endif
  }

  size_t flops = (2L*(size_t)stencil_size+1L) * active_points;
  double avgtime = stencil_time/iterations;
  printf("Rate (MFlops/s): %lf  Avg time (s): %lf\n",
         1.0e-6 * (double)flops/avgtime, (double)avgtime);

  exit(EXIT_SUCCESS);

  return 0;
}
