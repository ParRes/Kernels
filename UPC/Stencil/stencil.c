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

USAGE:   The program takes as input the linear dimension of the grid,
         and the number of iterations on the grid

         <progname> <# iterations> <grid size> <x_tiles>

         x_tiles=0 does automated 2D grid decomposition based on
         number of threads used. Otherwise, user can choose a different
         partitioning using this command-line parameter.

         The output consists of diagnostics to make sure the
         algorithm worked, and of timing statistics.

HISTORY: Written by Abdullah Kayi, June 2015

*******************************************************************/
#include <par-res-kern_general.h>
#include <par-res-kern_upc.h>

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

/* define shorthand for indexing multi-dimensional arrays */
#define WEIGHT(ii,jj) weight[ii+RADIUS][jj+RADIUS]
#define IN(i,j)   in_array_private[j][i]
#define OUT(i,j)  out_array_private[j][i]

shared DTYPE times[THREADS];

void bail_out(char *fmt, ...){
  va_list argp;
  char buffer[1024];

  va_start(argp, fmt);
  vsnprintf(buffer, 1024, fmt, argp);
  va_end(argp);

  fprintf(stderr, "ERROR: %s\n", buffer);

  upc_global_exit(EXIT_FAILURE);
}

typedef shared [] DTYPE * RESTRICT local_shared_block;
typedef shared [] local_shared_block *local_shared_block_ptrs;
typedef local_shared_block *private_shared_block_ptrs;

shared [1] double norms[THREADS];
shared [1] local_shared_block_ptrs in_arrays[THREADS];
shared [1] local_shared_block_ptrs out_arrays[THREADS];

private_shared_block_ptrs *private_in_arrays;
private_shared_block_ptrs *private_out_arrays;

shared [1] int thread_sizex[THREADS];
shared [1] int thread_sizey[THREADS];
shared [1] int thread_offsetx[THREADS];
shared [1] int thread_offsety[THREADS];

local_shared_block_ptrs shared_2d_array_alloc(int sizex, int sizey, int offsetx, int offsety){
  long int alloc_size = sizex * sizey * sizeof(DTYPE);
  local_shared_block ptr;

  ptr = upc_alloc(alloc_size);
  if(ptr == NULL)
    bail_out("Failing shared allocation of %d bytes", alloc_size);

  long int line_ptrs_size = sizeof(local_shared_block) * sizey;
  local_shared_block_ptrs line_ptrs = upc_alloc(line_ptrs_size);
  if(line_ptrs == NULL)
    bail_out("Failing shared allocation of %d bytes", line_ptrs_size);

  for(int y=0; y<sizey; y++){
    line_ptrs[y] = ptr + (y * sizex) - offsetx;
  }

  line_ptrs -= offsety;

  return line_ptrs;
}

DTYPE **shared_2d_array_to_private(local_shared_block_ptrs array, int sizex, int sizey, int offsetx, int offsety){
  long int alloc_size = sizey * sizeof(DTYPE*);
  DTYPE **ptr = prk_malloc(alloc_size);
  if(ptr == NULL)
    bail_out("Unable to allocate array");

  ptr -= offsety;

  for(int y=offsety; y<offsety + sizey; y++)
    ptr[y] = (DTYPE *)(&array[y][offsetx]) - offsetx;

  return ptr;
}

private_shared_block_ptrs partially_privatize(local_shared_block_ptrs array, int thread){
  int sizey = thread_sizey[thread];
  int offsety = thread_offsety[thread];

  long int alloc_size = sizey * sizeof(local_shared_block);
  private_shared_block_ptrs ptr = prk_malloc(alloc_size);
  if(ptr == NULL)
    bail_out("Unable to allocate array2");

  ptr -= offsety;
  for(int y=offsety; y<offsety + sizey; y++)
    ptr[y] = (&array[y][0]);

  return ptr;
}

int main(int argc, char ** argv) {

  int    n;               /* linear grid dimension */
  int    i, j, ii, jj, it, jt, iter;  /* dummies */
  double norm,            /* L1 norm of solution */
         reference_norm;
  double f_active_points; /* interior of grid with respect to stencil */
  DTYPE  flops;           /* floating point ops per iteration */
  int    iterations;      /* number of times to run the algorithm */
  double stencil_time,    /* timing parameters */
         avgtime, max_time;
  int    stencil_size;    /* number of points in stencil */
  DTYPE  weight[2*RADIUS+1][2*RADIUS+1]; /* weights of points in the stencil */
  int    istart;    /* bounds of grid tile assigned to calling rank        */
  int    jstart;    /* bounds of grid tile assigned to calling rank        */
  int    Num_procsx, Num_procsy;

  /*******************************************************************************
  ** process and test input parameters
  ********************************************************************************/
  if(MYTHREAD == 0){
    printf("Parallel Research Kernels version %s\n", PRKVERSION);
    printf("UPC stencil execution on 2D grid\n");
    fflush(stdout);
  }

  if (argc != 4 && argc != 3)
    if(MYTHREAD == 0)
      bail_out("Usage: %s <# iterations> <array dimension> [x_tiles]\n", *argv);

  iterations  = atoi(*++argv);
  if (iterations < 1)
    if(MYTHREAD == 0)
      bail_out("iterations must be >= 1 : %d", iterations);

  n  = atoi(*++argv);

  if (n < 1)
    if(MYTHREAD == 0)
      bail_out("grid dimension must be positive: %d", n);

  if (argc == 4)
    Num_procsx  = atoi(*++argv);
  else
    Num_procsx = 0;

  if(Num_procsx < 0)
    if(MYTHREAD == 0)
      bail_out("Number of tiles in the x-direction should be positive (got: %d)", Num_procsx);

  if(Num_procsx > THREADS)
    if(MYTHREAD == 0)
      bail_out("Number of tiles in the x-direction should be < THREADS (got: %d)", Num_procsx);

  /* Num_procsx=0 refers to automated calculation of division on each coordinates like MPI code */
  if(Num_procsx == 0){
    factor(THREADS, &Num_procsx, &Num_procsy);
  }
  else {
    Num_procsy = THREADS / Num_procsx;
  }

  if(RADIUS < 1)
    if(MYTHREAD == 0)
      bail_out("Stencil radius %d should be positive", RADIUS);

  if(2*RADIUS +1 > n)
    if(MYTHREAD == 0)
      bail_out("Stencil radius %d exceeds grid size %d", RADIUS, n);

  if(Num_procsx * Num_procsy != THREADS){
    bail_out("Num_procsx * Num_procsy != THREADS");
  }

  /* compute amount of space required for input and solution arrays             */

  int my_IDx = MYTHREAD % Num_procsx;
  int my_IDy = MYTHREAD / Num_procsx;

  int blockx = n / Num_procsx;
  int leftover = n % Num_procsx;
  if (my_IDx < leftover) {
    istart = (blockx + 1) * my_IDx;
    blockx += 1;
  }
  else {
    istart = (blockx+1) * leftover + blockx * (my_IDx-leftover);
  }

  if (blockx == 0)
    bail_out("No work to do on x-direction!");

  int blocky = n / Num_procsy;
  leftover = n % Num_procsy;
  if (my_IDy < leftover) {
    jstart = (blocky+1) * my_IDy;
    blocky += 1;
  }
  else {
    jstart = (blocky+1) * leftover + blocky * (my_IDy-leftover);
  }

  if (blocky == 0)
    bail_out("No work to do on y-direction!");

  if(blockx < RADIUS || blocky < RADIUS) {
    bail_out("blockx < RADIUS || blocky < RADIUS");
  }

  int myoffsetx = istart - RADIUS;
  int myoffsety = jstart - RADIUS;
  thread_offsetx[MYTHREAD] = myoffsetx;
  thread_offsety[MYTHREAD] = myoffsety;

  int sizex = blockx + 2*RADIUS;
  int sizey = blocky + 2*RADIUS;
  thread_sizex[MYTHREAD] = sizex;
  thread_sizey[MYTHREAD] = sizey;

  upc_barrier;

  local_shared_block_ptrs in_array  = shared_2d_array_alloc(sizex, sizey, myoffsetx, myoffsety);
  local_shared_block_ptrs out_array = shared_2d_array_alloc(sizex, sizey, myoffsetx, myoffsety);

  in_arrays[MYTHREAD] = in_array;
  out_arrays[MYTHREAD] = out_array;

  DTYPE **in_array_private = shared_2d_array_to_private(in_array, sizex, sizey, myoffsetx, myoffsety);
  DTYPE **out_array_private = shared_2d_array_to_private(out_array, sizex, sizey, myoffsetx, myoffsety);

  upc_barrier;

  private_in_arrays = prk_malloc(sizeof(private_shared_block_ptrs) * THREADS);
  if(private_in_arrays == NULL)
    bail_out("Cannot allocate private_in_arrays");

  private_out_arrays = prk_malloc(sizeof(private_shared_block_ptrs) * THREADS);
  if(private_out_arrays == NULL)
    bail_out("Cannot allocate private_out_arrays");

  for(int thread=0; thread<THREADS; thread++){
    private_in_arrays[thread] = partially_privatize(in_arrays[thread], thread);
    private_out_arrays[thread] = partially_privatize(out_arrays[thread], thread);
  }

  /* intialize the input and output arrays */
  for(int y=myoffsety; y<myoffsety + sizey; y++){
    for(int x=myoffsetx; x<myoffsetx + sizex; x++){
      in_array_private[y][x] = COEFX*x + COEFY*y;
      out_array[y][x] = 0.;
    }
  }
  upc_barrier;

  for(int y=myoffsety; y<myoffsety + sizey; y++){
    for(int x=myoffsetx; x<myoffsetx + sizex; x++){
      if(in_array_private[y][x] != COEFX*x + COEFY*y)
        bail_out("x=%d y=%d in_array=%f != %f", x, y, in_array[y][x], COEFX*x + COEFY*y);
    }
  }

  /* fill the stencil weights to reflect a discrete divergence operator */
  for (jj=-RADIUS; jj<=RADIUS; jj++)
    for (ii=-RADIUS; ii<=RADIUS; ii++)
      WEIGHT(ii, jj) = (DTYPE)0.0;

  stencil_size = 4*RADIUS+1;
  for (ii=1; ii<=RADIUS; ii++) {
    WEIGHT(0, ii) = WEIGHT( ii,0) =  (DTYPE) (1.0/(2.0*ii*RADIUS));
    WEIGHT(0,-ii) = WEIGHT(-ii,0) = -(DTYPE) (1.0/(2.0*ii*RADIUS));
  }

  if(MYTHREAD == 0){
    printf("Number of threads      = %d\n", THREADS);
    printf("Grid size              = %d\n", n);
    printf("Radius of stencil      = %d\n", RADIUS);
    printf("Tiles in x/y-direction = %d/%d\n", Num_procsx, Num_procsy);
#if DOUBLE
    printf("Data type              = double precision\n");
#else
    printf("Data type              = single precision\n");
#endif
#if LOOPGEN
    printf("Script used to expand stencil loop body\n");
#else
    printf("Compact representation of stencil loop body\n");
#endif
    printf("Number of iterations   = %d\n", iterations);
  }

  upc_barrier;

  int startx = myoffsetx + RADIUS;
  int endx = myoffsetx + sizex - RADIUS;

  int starty = myoffsety + RADIUS;
  int endy = myoffsety + sizey - RADIUS;

  if(my_IDx == 0)
    startx += RADIUS;

  if(my_IDx == Num_procsx - 1)
    endx -= RADIUS;

  if(my_IDy == 0)
    starty += RADIUS;

  if(my_IDy == Num_procsy - 1)
    endy -= RADIUS;

  upc_barrier;

  for (iter = 0; iter<=iterations; iter++){
    /* start timer after a warmup iteration */
    if (iter == 1) {
      upc_barrier;
      stencil_time = wtime();
    }

    /* Get ghost zones */
    /* NORTH */
    if(my_IDy != 0){
      int peer = (my_IDy - 1) * Num_procsx + my_IDx;
      for (int y=starty - RADIUS; y<starty; y++) {
        int transfer_size = (endx - startx) * sizeof(DTYPE);
        upc_memget(&in_array_private[y][startx], &private_in_arrays[peer][y][startx], transfer_size);
      }
    }
    /* SOUTH */
    if(my_IDy != Num_procsy - 1){
      int peer = (my_IDy + 1) * Num_procsx + my_IDx;
      for (int y=endy; y<endy + RADIUS; y++) {
        int transfer_size = (endx - startx) * sizeof(DTYPE);
        upc_memget(&in_array_private[y][startx], &private_in_arrays[peer][y][startx], transfer_size);
      }
    }
    /* LEFT */
    if(my_IDx != 0){
      int peer = my_IDy * Num_procsx + my_IDx - 1;
      for (int y=starty; y<endy; y++) {
        for (int x=startx - RADIUS; x<startx; x++) {
          in_array_private[y][x] = private_in_arrays[peer][y][x];
        }
      }
    }
    /* RIGHT*/
    if(my_IDx != Num_procsx - 1){
      int peer = my_IDy * Num_procsx + my_IDx + 1;
      for (int y=starty; y<endy; y++) {
        for (int x=endx; x<endx + RADIUS; x++) {
          in_array_private[y][x] = private_in_arrays[peer][y][x];
        }
      }
    }

    /* Apply the stencil operator */
    for (j=starty; j<endy; j++) {
      for (i=startx; i<endx; i++) {
        #if LOOPGEN
          #include "loop_body_star.incl"
        #else
          for (jj=-RADIUS; jj<=RADIUS; jj++) OUT(i,j) += WEIGHT(0,jj)*IN(i,j+jj);
          for (ii=-RADIUS; ii<0; ii++)       OUT(i,j) += WEIGHT(ii,0)*IN(i+ii,j);
          for (ii=1; ii<=RADIUS; ii++)       OUT(i,j) += WEIGHT(ii,0)*IN(i+ii,j);
        #endif
      }
    }

    upc_barrier; /* <- Necessary barrier: some slow threads could use future data */

    /* add constant to solution to force refresh of neighbor data, if any */
    for(int y=myoffsety + RADIUS; y<myoffsety + sizey - RADIUS; y++)
      for(int x=myoffsetx + RADIUS; x<myoffsetx + sizex - RADIUS; x++)
        in_array_private[y][x] += 1.0;

    upc_barrier; /* <- Necessary barrier: some threads could start on old data */
  } /* end of iterations */

  stencil_time = wtime() - stencil_time;
  times[MYTHREAD] = stencil_time;

  upc_barrier;

  // Compute max_time
  if(MYTHREAD == 0){
    max_time = times[MYTHREAD];
    for(i=1; i<THREADS; i++){
      if(max_time < times[i])
        max_time = times[i];
    }
  }

  norm = (double) 0.0;
  f_active_points = (double)(n-2*RADIUS) * (double)(n-2*RADIUS);

  /* compute L1 norm in parallel */
  for (int y=starty; y<endy; y++) {
    for (int x=startx; x<endx; x++) {
      norm += (double)ABS(out_array[y][x]);
    }
  }

  norm /= f_active_points;
  norms[MYTHREAD] = norm;

  upc_barrier;

  if(MYTHREAD == 0){
    norm = 0.;
    for(int i=0; i<THREADS; i++) norm += norms[i];

    /*******************************************************************************
    ** Analyze and output results.
    ********************************************************************************/

    /* verify correctness */
    reference_norm = (double) (iterations+1) * (COEFX + COEFY);

    if (ABS(norm - reference_norm) > EPSILON)
      bail_out("L1 norm = "FSTR", Reference L1 norm = "FSTR"\n", norm, reference_norm);
    else {
      printf("Solution validates\n");
#if VERBOSE
      printf("Reference L1 norm = "FSTR", L1 norm = "FSTR"\n",
             reference_norm, norm);
#endif
    }

    flops = (DTYPE) (2*stencil_size+1) * f_active_points;
    avgtime = max_time/iterations;
    printf("Rate (MFlops/s): "FSTR"  Avg time (s): %lf\n",
           1.0E-06 * flops/avgtime, avgtime);

    exit(EXIT_SUCCESS);
  }
}
