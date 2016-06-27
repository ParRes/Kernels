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

NAME:    Pipeline

PURPOSE: This program tests the efficiency with which point-to-point
         synchronization can be carried out. It does so by executing
         a pipelined algorithm on an m*n grid. The first array dimension
         is distributed among the ranks (stripwise decomposition).

USAGE:   The program takes as input the dimensions of the grid, and the
         number of times we loop over the grid

               <progname> <# iterations> <m> <n>

         The output consists of diagnostics to make sure the
         algorithm worked, and of timing statistics.

HISTORY: Written by Abdullah Kayi, September 2015

*******************************************************************/
#include <par-res-kern_general.h>
#include <par-res-kern_upc.h>

shared double times[THREADS];

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

typedef shared [] double * RESTRICT local_shared_block;
typedef shared [] local_shared_block *local_shared_block_ptrs;

shared [1] local_shared_block_ptrs in_arrays[THREADS];

local_shared_block_ptrs shared_2d_array_alloc(int sizex, int sizey, int offsetx, int offsety){
  size_t alloc_size = (size_t)sizex * sizey * sizeof(double);
  local_shared_block ptr;

  debug("Allocating main array size(%d, %d) offset(%d, %d) %zu", sizex, sizey, offsetx, offsety, alloc_size);
  ptr = upc_alloc(alloc_size);
  if(ptr == NULL)
    die("Failing shared allocation of %d bytes", alloc_size);

  int line_ptrs_size = sizeof(local_shared_block) * sizey;
  debug("Allocating ptr array %d", line_ptrs_size);
  local_shared_block_ptrs line_ptrs = upc_alloc(line_ptrs_size);
  if(line_ptrs == NULL)
    die("Failing shared allocation of %d bytes", line_ptrs_size);

  for(long y=0; y<sizey; y++){
    line_ptrs[y] = ptr + (y * sizex) - offsetx;
  }

  line_ptrs -= offsety;

  return line_ptrs;
}

double **shared_2d_array_to_private(local_shared_block_ptrs array, int sizex, int sizey, int offsetx, int offsety){
  size_t alloc_size = (size_t)sizey * sizeof(double*);
  double **ptr = prk_malloc(alloc_size);
  if(ptr == NULL)
    die("Unable to allocate array");

  ptr -= offsety;

  for(int y=offsety; y<offsety + sizey; y++)
    ptr[y] = (double *)(&array[y][offsetx]) - offsetx;

  return ptr;
}

/* define shorthand for indexing a multi-dimensional array                       */
#define ARRAY(i,j) in_array_private[j][i]

strict shared int current_max_line[THREADS];
#if USE_BUPC_EXT
bupc_sem_t * shared allflags[THREADS];
#endif

int main(int argc, char ** argv) {

  long   m, n;            /* grid dimensions                                     */
  int    i, j, iter;      /* dummies                                             */
  int    iterations;      /* number of times to run the pipeline algorithm       */
  double pipeline_time,   /* timing parameters                                   */
         avgtime, max_time;
  double epsilon = 1.e-8; /* error tolerance                                     */
  double corner_val;      /* verification value at top right corner of grid      */
  double *vector;/* array holding grid values                           */
  long   total_length;    /* total required length to store grid values          */

  /*******************************************************************************
  ** process and test input parameters
  ********************************************************************************/

  if(MYTHREAD == THREADS-1){
    printf("Parallel Research Kernels version %s\n", PRKVERSION);
    printf("UPC pipeline execution on 2D grid\n");
  }

  if (argc != 4){
    if(MYTHREAD == THREADS-1){
      printf("Usage: %s <# iterations> <first array dimension> ", *argv);
      printf("<second array dimension>\n");
    }
    upc_global_exit(EXIT_FAILURE);
  }

  iterations  = atoi(*++argv);
  if (iterations < 1){
    if(MYTHREAD == THREADS-1)
      printf("ERROR: iterations must be >= 1 : %d \n",iterations);
    upc_global_exit(EXIT_FAILURE);
  }

  m  = atol(*++argv);
  n  = atol(*++argv);

  if (m < 1 || n < 1){
    if(MYTHREAD == THREADS-1)
      printf("ERROR: grid dimensions must be positive: %d, %d \n", m, n);
    upc_global_exit(EXIT_FAILURE);
  }

  if(MYTHREAD == THREADS-1){
    printf("Number of threads         = %d\n", THREADS);
    printf("Grid sizes                = %ld, %ld\n", m, n);
    printf("Number of iterations      = %d\n", iterations);
#if USE_BUPC_EXT
    printf("Using Berkeley UPC extensions\n");
#endif
  }

  /*********************************************************************
  ** Allocate memory for input and output matrices
  *********************************************************************/
#if USE_BUPC_EXT
  bupc_sem_t *myflag = bupc_sem_alloc(BUPC_SEM_INTEGER | BUPC_SEM_MPRODUCER);
  upc_barrier;
  allflags[MYTHREAD] = myflag;
  upc_barrier;
  bupc_sem_t *mypeer = allflags[(MYTHREAD+1) % THREADS];
#endif

  long segment_size = m / THREADS;
  int leftover = m % THREADS;
  int myoffsetx, sizex;

  if(MYTHREAD < leftover){
    myoffsetx = (segment_size + 1) * MYTHREAD;
    sizex = segment_size + 1;
  }else{
    myoffsetx = (segment_size + 1) * leftover + segment_size * (MYTHREAD - leftover);
    sizex = segment_size;
  }

#if USE_BUPC_EXT
  if(MYTHREAD != 0){
    myoffsetx -= 1;
    sizex += 1;
  }
#endif

  int sizey = n;
  int myoffsety = 0;

  upc_barrier;

  debug("Allocating arrays (%d, %d), offset (%d, %d)", sizex, sizey, myoffsetx, myoffsety);
  local_shared_block_ptrs in_array  = shared_2d_array_alloc(sizex, sizey, myoffsetx, myoffsety);

  in_arrays[MYTHREAD] = in_array;

  double **in_array_private = shared_2d_array_to_private(in_array, sizex, sizey, myoffsetx, myoffsety);

  if(MYTHREAD == 0)
    current_max_line[MYTHREAD] = sizey;
  else
    current_max_line[MYTHREAD] = 0;

  upc_barrier;

  /*********************************************************************
  ** Initialize the matrices
  *********************************************************************/

  /* clear the array                                                             */
  for (j=0; j<n; j++)
    for (i=myoffsetx; i<myoffsetx + sizex; i++)
      ARRAY(i, j) = 0.0;

  /* set boundary values (bottom and left side of grid                           */
  if(MYTHREAD == 0)
    for (j=0; j<n; j++)
      ARRAY(0, j) = (double) j;

  for (i=myoffsetx; i<myoffsetx + sizex; i++)
    ARRAY(i, 0) = (double) i;

  upc_barrier;

  for (iter = 0; iter<=iterations; iter++){
    /* start timer after a warmup iteration */
    if (iter == 1)
      pipeline_time = wtime();
    if(MYTHREAD == 0)
      debug("start it %d, %f", iter, ARRAY(0, 0));

    if(MYTHREAD != THREADS - 1)  // Send the element in line 0
      in_arrays[MYTHREAD + 1][0][myoffsetx + sizex -1] = ARRAY(myoffsetx + sizex - 1, 0);

    for (j=1; j<n; j++) {
#if USE_BUPC_EXT
      if(MYTHREAD > 0){
        bupc_sem_wait(myflag);
      }

      for (i=myoffsetx+1; i<myoffsetx + sizex; i++)
        ARRAY(i, j) = ARRAY(i-1, j) + ARRAY(i, j-1) - ARRAY(i-1, j-1);

      if(MYTHREAD != THREADS - 1){
        in_arrays[MYTHREAD + 1][j][myoffsetx + sizex -1] = ARRAY(myoffsetx + sizex - 1, j);

        bupc_sem_post(mypeer);
      }
#else
      while(j > current_max_line[MYTHREAD]) // Normally not necessary: bupc_poll();
        ;

      if(MYTHREAD > 0)
        ARRAY(myoffsetx, j) = in_arrays[MYTHREAD - 1][j][myoffsetx-1] + ARRAY(myoffsetx, j-1) - in_arrays[MYTHREAD-1][j-1][myoffsetx-1];

      for (i=myoffsetx+1; i<myoffsetx + sizex; i++)
        ARRAY(i, j) = ARRAY(i-1, j) + ARRAY(i, j-1) - ARRAY(i-1, j-1);

      if(MYTHREAD < THREADS - 1)
        current_max_line[MYTHREAD+1] = j;

#endif
    }

    /* copy top right corner value to bottom left corner to create dependency; we
       need a barrier to make sure the latest value is used. This also guarantees
     that the flags for the next iteration (if any) are not getting clobbered  */
    if(MYTHREAD == 0)
      current_max_line[MYTHREAD] = sizey;
    else
      current_max_line[MYTHREAD] = 0;

    if(MYTHREAD == THREADS - 1){
      in_arrays[0][0][0] = -ARRAY(m-1, n-1);
    }
    upc_barrier;
  }

  pipeline_time = wtime() - pipeline_time;
  times[MYTHREAD] = pipeline_time;

  upc_barrier;

  // Compute max_time
  if(MYTHREAD == THREADS - 1){
    max_time = times[MYTHREAD];
    for(i=1; i<THREADS; i++){
      if(max_time < times[i])
        max_time = times[i];
    }
  }

  /*******************************************************************************
  ** Analyze and output results.
  ********************************************************************************/

  /* verify correctness, using top right value;                                  */
  if( MYTHREAD == THREADS - 1){
    corner_val = (double)((iterations+1)*(n+m-2));
    if (fabs(ARRAY(m-1,n-1)-corner_val)/corner_val > epsilon) {
      printf("ERROR: checksum %lf does not match verification value %lf\n",
          ARRAY(m-1, n-1), corner_val);
      exit(EXIT_FAILURE);
    }
#if VERBOSE
    printf("checksum %lf verification value %lf\n",
        ARRAY(m-1, n-1), corner_val);
    printf("Solution validates; verification value = %lf\n", corner_val);
#else
    printf("Solution validates\n");
#endif
    avgtime = max_time/iterations;
  printf("Rate (MFlops/s): %lf Avg time (s): %lf\n",
         1.0E-06 * 2 * ((double)(m-1)*(double)(n-1))/avgtime, avgtime);
  exit(EXIT_SUCCESS);
  }
}
