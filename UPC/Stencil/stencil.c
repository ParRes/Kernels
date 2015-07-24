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

#ifndef RADIUS
  #define RADIUS 2
#endif

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

/* define shorthand for indexing a multi-dimensional array */
#define WEIGHT(ii,jj) weight[ii+RADIUS][jj+RADIUS]

shared DTYPE times[THREADS];

int is_debugging = 0;
void debug(char *fmt, ...){
  va_list argp;
  char buffer[1024];

  if(!is_debugging)
      return;

  va_start(argp, fmt);
  vsnprintf(buffer, 1024, fmt, argp);
  va_end(argp);

  fprintf(stdout, "%2d > %s \n", MYTHREAD, buffer);
  fflush(stdout);
}

void message(char *fmt, ...){
  va_list argp;
  char buffer[1024];

  va_start(argp, fmt);
  vsnprintf(buffer, 1024, fmt, argp);
  va_end(argp);

  fprintf(stdout, "%2d > %s \n", MYTHREAD, buffer);
  fflush(stdout);
}

void die(char *fmt, ...){
  va_list argp;
  char buffer[1024];

  va_start(argp, fmt);
  vsnprintf(buffer, 1024, fmt, argp);
  va_end(argp);

  fprintf(stderr, "FATAL ERROR %s\n", buffer);

  upc_global_exit(EXIT_FAILURE);
}

typedef shared [] DTYPE *local_shared_block;
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
  int alloc_size = sizex * sizey * sizeof(DTYPE);
  local_shared_block ptr;

  debug("Allocating main array size(%d, %d) offset(%d, %d) %d", sizex, sizey, offsetx, offsety, alloc_size);
  ptr = upc_alloc(alloc_size);
  if(ptr == NULL)
    die("Failing shared allocation of %d bytes", alloc_size);

  int line_ptrs_size = sizeof(local_shared_block) * sizey;
  debug("Allocating ptr array %d", line_ptrs_size);
  local_shared_block_ptrs line_ptrs = upc_alloc(line_ptrs_size);
  if(line_ptrs == NULL)
    die("Failing shared allocation of %d bytes", line_ptrs_size);

  for(int y=0; y<sizey; y++){
    line_ptrs[y] = ptr + (y * sizex) - offsetx;
  }

  line_ptrs -= offsety;

  return line_ptrs;
}

DTYPE **shared_2d_array_to_private(local_shared_block_ptrs array, int sizex, int sizey, int offsetx, int offsety){
  int alloc_size = sizey * sizeof(DTYPE*);
  DTYPE **ptr = malloc(alloc_size);
  if(ptr == NULL)
    die("Unable to allocate array");

  ptr -= offsety;

  for(int y=offsety; y<offsety + sizey; y++)
    ptr[y] = (DTYPE *)(&array[y][offsetx]) - offsetx;

  return ptr;
}

private_shared_block_ptrs partially_privatize(local_shared_block_ptrs array, int thread){
  int sizey = thread_sizey[thread];
  int offsety = thread_offsety[thread];

  int alloc_size = sizey * sizeof(local_shared_block);
  private_shared_block_ptrs ptr = malloc(alloc_size);
  if(ptr == NULL)
    die("Unable to allocate array2");

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
  int x_divs, y_divs;

  /*******************************************************************************
  ** process and test input parameters
  ********************************************************************************/
  if(MYTHREAD == 0){
    printf("UPC stencil execution on 2D grid\n");
    fflush(stdout);
  }

  if (argc != 4 && argc != 3)
    if(MYTHREAD == 0)
      die("Usage: %s <# iterations> <array dimension> [x_tiles]\n", *argv);

  iterations  = atoi(*++argv);
  if (iterations < 1)
    if(MYTHREAD == 0)
      die("iterations must be >= 1 : %d", iterations);

  n  = atoi(*++argv);

  if (n < 1)
    if(MYTHREAD == 0)
      die("grid dimension must be positive: %d", n);

  if (argc == 4) 
    x_divs  = atoi(*++argv);
  else 
    x_divs = 0;

  if(x_divs < 0)
    if(MYTHREAD == 0)
      die("Number of tiles in the x-direction should be positive (got: %d)", x_divs);

  if(x_divs > THREADS)
    if(MYTHREAD == 0)
      die("Number of tiles in the x-direction should be < THREADS (got: %d)", x_divs);

  /* x_divs=0 refers to automated calculation of division on each coordinates like MPI code */
  if(x_divs == 0){
    for (x_divs=(int) (sqrt(THREADS+1)); x_divs>0; x_divs--) {
      if (!(THREADS%x_divs)) {
        y_divs = THREADS/x_divs;
        break;
      }
    }
  }
  else {
    y_divs = THREADS / x_divs;
  }

  if(THREADS % x_divs != 0)
    if(MYTHREAD == 0)
      die("THREADS %% x_divs != 0 (%d)", x_divs);

  if(RADIUS < 1)
    if(MYTHREAD == 0)
      die("Stencil radius %d should be positive", RADIUS);

  if(2*RADIUS +1 > n)
    if(MYTHREAD == 0)
      die("Stencil radius %d exceeds grid size %d", RADIUS, n);

  if(n%THREADS != 0)
    if(MYTHREAD == 0)
      die("n%%THREADS should be zero\n");

  int blockx = n / x_divs;
  int blocky = n / y_divs;

  int sizex = (n / x_divs) + 2*RADIUS;
  int sizey = (n / y_divs) + 2*RADIUS;

  int mygridposx = MYTHREAD % x_divs;
  int mygridposy = MYTHREAD / x_divs;

  int myoffsetx = mygridposx * blockx - RADIUS;
  int myoffsety = mygridposy * blocky - RADIUS;

  thread_sizex[MYTHREAD] = sizex;
  thread_sizey[MYTHREAD] = sizey;
  thread_offsetx[MYTHREAD] = myoffsetx;
  thread_offsety[MYTHREAD] = myoffsety;

  upc_barrier;

  debug("Allocating arrays (%d, %d), offset (%d, %d)", sizex, sizey, myoffsetx, myoffsety);
  local_shared_block_ptrs in_array  = shared_2d_array_alloc(sizex, sizey, myoffsetx, myoffsety);
  local_shared_block_ptrs out_array = shared_2d_array_alloc(sizex, sizey, myoffsetx, myoffsety);

  in_arrays[MYTHREAD] = in_array;
  out_arrays[MYTHREAD] = out_array;

  DTYPE **in_array_private = shared_2d_array_to_private(in_array, sizex, sizey, myoffsetx, myoffsety);
  DTYPE **out_array_private = shared_2d_array_to_private(out_array, sizex, sizey, myoffsetx, myoffsety);

  upc_barrier;

  private_in_arrays = malloc(sizeof(private_shared_block_ptrs) * THREADS);
  if(private_in_arrays == NULL)
    die("Cannot allocate private_in_arrays");

  private_out_arrays = malloc(sizeof(private_shared_block_ptrs) * THREADS);
  if(private_out_arrays == NULL)
    die("Cannot allocate private_out_arrays");

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
        die("x=%d y=%d in_array=%f != %f", x, y, in_array[y][x], COEFX*x + COEFY*y);
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
    printf("Tiles in x/y-direction = %d/%d\n", x_divs, y_divs);
#ifdef DOUBLE
    printf("Data type              = double precision\n");
#else
    printf("Data type              = single precision\n");
#endif
    printf("Number of iterations   = %d\n", iterations);
  }

  upc_barrier;

  int startx = myoffsetx + RADIUS;
  int endx = myoffsetx + sizex - RADIUS;

  int starty = myoffsety + RADIUS;
  int endy = myoffsety + sizey - RADIUS;

  if(mygridposx == 0)
    startx += RADIUS;

  if(mygridposx == x_divs - 1)
    endx -= RADIUS;

  if(mygridposy == 0)
    starty += RADIUS;

  if(mygridposy == y_divs - 1)
    endy -= RADIUS;

  debug("divx=%d, divy= %d, endx=%d, endy=%d", x_divs, y_divs, endx, endy);
  debug("startx =%d, starty= %d, endx=%d, endy=%d", startx, starty, endx, endy);

  upc_barrier;

  for (iter = 0; iter<=iterations; iter++){
    /* start timer after a warmup iteration */
    if (iter == 1)
      stencil_time = wtime();

    /* Get ghost zones */
    /* NORTH */
    if(mygridposy != 0){
      int peer = (mygridposy - 1) * x_divs + mygridposx;
      for (int y=starty - RADIUS; y<starty; y++) {
        int transfer_size = (endx - startx) * sizeof(DTYPE);
        upc_memget(&in_array_private[y][startx], &private_in_arrays[peer][y][startx], transfer_size);
      }
    }
    /* SOUTH */
    if(mygridposy != y_divs - 1){
      int peer = (mygridposy + 1) * x_divs + mygridposx;
      for (int y=endy; y<endy + RADIUS; y++) {
        int transfer_size = (endx - startx) * sizeof(DTYPE);
        upc_memget(&in_array_private[y][startx], &private_in_arrays[peer][y][startx], transfer_size);
      }
    }
    /* LEFT */
    if(mygridposx != 0){
      int peer = mygridposy * x_divs + mygridposx - 1;
      for (int y=starty; y<endy; y++) {
        for (int x=startx - RADIUS; x<startx; x++) {
          in_array_private[y][x] = private_in_arrays[peer][y][x];
        }
      }
    }
    /* RIGHT*/
    if(mygridposx != x_divs - 1){
      int peer = mygridposy * x_divs + mygridposx + 1;
      for (int y=starty; y<endy; y++) {
        for (int x=endx; x<endx + RADIUS; x++) {
          in_array_private[y][x] = private_in_arrays[peer][y][x];
        }
      }
    }

    /* Apply the stencil operator */
    for (int y=starty; y<endy; y++) {
      for (int x=startx; x<endx; x++) {
        for (int xx=-RADIUS; xx<=RADIUS; xx++)
          out_array_private[y][x] += WEIGHT(0, xx) * in_array_private[y][x + xx];

        for (int yy=-RADIUS; yy<0; yy++)
          out_array_private[y][x] += WEIGHT(yy, 0) * in_array_private[y + yy][x];

        for (int yy=1; yy<=RADIUS; yy++)
          out_array_private[y][x] += WEIGHT(yy, 0) * in_array_private[y + yy][x];
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

  norm = (DTYPE) 0.0;
  f_active_points = (double) (n-2*RADIUS)*(double) (n-2*RADIUS);

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
    for(int i=0;i<THREADS;i++)
      norm += norms[i];

    /*******************************************************************************
    ** Analyze and output results.
    ********************************************************************************/

    /* verify correctness */
    reference_norm = (DTYPE) (iterations+1) * (COEFX + COEFY);

    if (ABS(norm-reference_norm) > EPSILON)
      die("L1 norm = "FSTR", Reference L1 norm = "FSTR"\n", norm, reference_norm);
    else {
      printf("Solution validates\n");
#ifdef VERBOSE
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
