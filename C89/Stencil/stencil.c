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

#if DOUBLE
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

int main(int argc, char ** argv) {

  long   n;               /* linear grid dimension                               */
  int    tile_size;       /* loop nest block factor                              */
  int    tiling=0;        /* boolean indication loop nest blocking               */
  int    i, j, ii, jj, it, jt, iter;  /* dummies                                 */
  DTYPE  norm,            /* L1 norm of solution                                 */
         reference_norm;
  DTYPE  f_active_points; /* interior of grid with respect to stencil            */
  DTYPE  flops;           /* floating point ops per iteration                    */
  int    iterations;      /* number of times to run the algorithm                */
  double stencil_time,    /* timing parameters                                   */
         avgtime;
  int    stencil_size;    /* number of points in stencil                         */
  DTYPE  * RESTRICT in;   /* input grid values                                   */
  DTYPE  * RESTRICT out;  /* output grid values                                  */
  long   total_length;    /* total required length to store grid values          */
  DTYPE  weight[2*RADIUS+1][2*RADIUS+1]; /* weights of points in the stencil     */

  printf("Parallel Research Kernels Version %s\n", PRKVERSION);
  printf("Serial stencil execution on 2D grid\n");

  /*******************************************************************************
  ** process and test input parameters    
  ********************************************************************************/

  if (argc != 3 && argc !=4){
    printf("Usage: %s <# iterations> <array dimension> [tile_size]\n", 
           *argv);
    return(EXIT_FAILURE);
  }

  iterations  = atoi(*++argv); 
  if (iterations < 1){
    printf("ERROR: iterations must be >= 1 : %d \n",iterations);
    exit(EXIT_FAILURE);
  }

  n  = atol(*++argv);

  if (n < 1){
    printf("ERROR: grid dimension must be positive: %ld\n", n);
    exit(EXIT_FAILURE);
  }

  if (argc == 4) {
    tile_size = atoi(*++argv);
    if (tile_size<=0 || tile_size>n) tile_size=n;
    else tiling=1; 
  }

  if (RADIUS < 1) {
    printf("ERROR: Stencil radius %d should be positive\n", RADIUS);
    exit(EXIT_FAILURE);
  }

  if (2*RADIUS +1 > n) {
    printf("ERROR: Stencil radius %d exceeds grid size %ld\n", RADIUS, n);
    exit(EXIT_FAILURE);
  }

  /*  make sure the vector space can be represented                             */
  total_length = n*n*sizeof(DTYPE);
  if (total_length/n != n*(signed)sizeof(DTYPE)) {
    printf("ERROR: Space for %ld x %ld grid cannot be represented; ", n, n);
    exit(EXIT_FAILURE);
  }

  in  = (DTYPE *) prk_malloc(total_length);
  out = (DTYPE *) prk_malloc(total_length);
  if (!in || !out) {
    printf("ERROR: could not allocate space for input or output array\n");
    exit(EXIT_FAILURE);
  }

  /* fill the stencil weights to reflect a discrete divergence operator         */
  for (jj=-RADIUS; jj<=RADIUS; jj++) {
      for (ii=-RADIUS; ii<=RADIUS; ii++) {
          WEIGHT(ii,jj) = (DTYPE) 0.0;
      }
  }
#if STAR
  stencil_size = 4*RADIUS+1;
  for (ii=1; ii<=RADIUS; ii++) {
    WEIGHT(0, ii) = WEIGHT( ii,0) =  (DTYPE) (1.0/(2.0*ii*RADIUS));
    WEIGHT(0,-ii) = WEIGHT(-ii,0) = -(DTYPE) (1.0/(2.0*ii*RADIUS));
  }
#else
  stencil_size = (2*RADIUS+1)*(2*RADIUS+1);
  for (jj=1; jj<=RADIUS; jj++) {
    for (ii=-jj+1; ii<jj; ii++) {
      WEIGHT(ii,jj)  =  (DTYPE) (1.0/(4.0*jj*(2.0*jj-1)*RADIUS));
      WEIGHT(ii,-jj) = -(DTYPE) (1.0/(4.0*jj*(2.0*jj-1)*RADIUS));
      WEIGHT(jj,ii)  =  (DTYPE) (1.0/(4.0*jj*(2.0*jj-1)*RADIUS));
      WEIGHT(-jj,ii) = -(DTYPE) (1.0/(4.0*jj*(2.0*jj-1)*RADIUS));
    }
    WEIGHT(jj,jj)    =  (DTYPE) (1.0/(4.0*jj*RADIUS));
    WEIGHT(-jj,-jj)  = -(DTYPE) (1.0/(4.0*jj*RADIUS));
  }
#endif

  norm = (DTYPE) 0.0;
  f_active_points = (DTYPE) (n-2*RADIUS)*(DTYPE) (n-2*RADIUS);

  printf("Grid size            = %ld\n", n);
  printf("Radius of stencil    = %d\n", RADIUS);
#if STAR
  printf("Type of stencil      = star\n");
#else
  printf("Type of stencil      = compact\n");
#endif
#if DOUBLE
  printf("Data type            = double precision\n");
#else
  printf("Data type            = single precision\n");
#endif
#if LOOPGEN
  printf("Script used to expand stencil loop body\n");
#else
  printf("Compact representation of stencil loop body\n");
#endif
  if (tiling) printf("Tile size            = %d\n", tile_size);
  else        printf("Untiled\n");
  printf("Number of iterations = %d\n", iterations);

  /* intialize the input and output arrays                                     */
  for (j=0; j<n; j++) for (i=0; i<n; i++) 
    IN(i,j) = COEFX*i+COEFY*j;
  for (j=RADIUS; j<n-RADIUS; j++) for (i=RADIUS; i<n-RADIUS; i++) 
    OUT(i,j) = (DTYPE)0.0;

  stencil_time = 0.0; /* silence compiler warning */

  for (iter = 0; iter<=iterations; iter++){

    /* start timer after a warmup iteration */
    if (iter == 1)  stencil_time = wtime();

    /* Apply the stencil operator                                              */

    if (!tiling) {
      for (j=RADIUS; j<n-RADIUS; j++) {
        for (i=RADIUS; i<n-RADIUS; i++) {
          #if STAR
            #if LOOPGEN
              #include "loop_body_star.incl"
            #else
              for (jj=-RADIUS; jj<=RADIUS; jj++)  OUT(i,j) += WEIGHT(0,jj)*IN(i,j+jj);
              for (ii=-RADIUS; ii<0; ii++)        OUT(i,j) += WEIGHT(ii,0)*IN(i+ii,j);
              for (ii=1; ii<=RADIUS; ii++)        OUT(i,j) += WEIGHT(ii,0)*IN(i+ii,j);
            #endif
          #else 
            #if LOOPGEN
              #include "loop_body_compact.incl"
            #else
              /* would like to be able to unroll this loop, but compiler will ignore  */
              for (jj=-RADIUS; jj<=RADIUS; jj++) 
              for (ii=-RADIUS; ii<=RADIUS; ii++)  OUT(i,j) += WEIGHT(ii,jj)*IN(i+ii,j+jj);
            #endif
          #endif
        }
      }
    }
    else {
      for (jt=RADIUS; jt<n-RADIUS; jt+=tile_size) {
        for (it=RADIUS; it<n-RADIUS; it+=tile_size) {
          for (j=jt; j<MIN(n-RADIUS,jt+tile_size); j++) {
            for (i=it; i<MIN(n-RADIUS,it+tile_size); i++) {
              #if STAR
                #if LOOPGEN
                  #include "loop_body_star.incl"
                #else
                  for (jj=-RADIUS; jj<=RADIUS; jj++)  OUT(i,j) += WEIGHT(0,jj)*IN(i,j+jj);
                  for (ii=-RADIUS; ii<0; ii++)        OUT(i,j) += WEIGHT(ii,0)*IN(i+ii,j);
                  for (ii=1; ii<=RADIUS; ii++)        OUT(i,j) += WEIGHT(ii,0)*IN(i+ii,j);
                #endif
              #else 
                #if LOOPGEN
                  #include "loop_body_compact.incl"
                #else
                  /* would like to be able to unroll this loop, but compiler will ignore  */
                  for (jj=-RADIUS; jj<=RADIUS; jj++) 
                  for (ii=-RADIUS; ii<=RADIUS; ii++)  OUT(i,j) += WEIGHT(ii,jj)*IN(i+ii,j+jj);
                #endif
              #endif
            }
          }
        }
      }
    }

    /* add constant to solution to force refresh of neighbor data, if any       */
    for (j=0; j<n; j++) for (i=0; i<n; i++) IN(i,j)+= 1.0;

  } /* end of iterations                                                        */

  stencil_time = wtime() - stencil_time;

  /* compute L1 norm in parallel                                                */
  for (j=RADIUS; j<n-RADIUS; j++) for (i=RADIUS; i<n-RADIUS; i++) {
    norm += (DTYPE)ABS(OUT(i,j));
  }

  norm /= f_active_points;

  /*******************************************************************************
  ** Analyze and output results.
  ********************************************************************************/

/* verify correctness                                                            */
  reference_norm = (DTYPE) (iterations+1) * (COEFX + COEFY);
  if (ABS(norm-reference_norm) > EPSILON) {
    printf("ERROR: L1 norm = "FSTR", Reference L1 norm = "FSTR"\n",
           norm, reference_norm);
    exit(EXIT_FAILURE);
  }
  else {
    printf("Solution validates\n");
#if VERBOSE
    printf("Reference L1 norm = "FSTR", L1 norm = "FSTR"\n", 
           reference_norm, norm);
#endif
  }

  flops = (DTYPE) (2*stencil_size+1) * f_active_points;
  avgtime = stencil_time/iterations;
  printf("Rate (MFlops/s): "FSTR"  Avg time (s): %lf\n",
         1.0E-06 * flops/avgtime, avgtime);

  exit(EXIT_SUCCESS);
}
