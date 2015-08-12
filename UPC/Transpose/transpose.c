/*
Copyright (c) 2015, Intel Corporation

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

NAME:    transpose

PURPOSE: This program tests the efficiency with which a square matrix
         can be transposed and stored in another matrix. The matrices
         are distributed identically.

USAGE:   Program inputs are the matrix order, the number of times to
         repeat the operation, and the communication mode

         transpose <# iterations> <matrix order> [tile size]

         An optional parameter specifies the tile size used to divide the
         individual matrix blocks for improved cache and TLB performance.

         The output consists of diagnostics to make sure the
         transpose worked and timing statistics.

HISTORY: Written by Abdullah Kayi, June 2015

*******************************************************************/

#include <par-res-kern_general.h>
#include <par-res-kern_upc.h>

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

typedef shared [] double *local_shared_block;
typedef shared [] local_shared_block *local_shared_block_ptrs;

shared [1] local_shared_block_ptrs in_arrays[THREADS];
shared [1] local_shared_block_ptrs out_arrays[THREADS];
shared [1] local_shared_block_ptrs buf_arrays[THREADS];

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
  double **ptr = malloc(alloc_size);
  if(ptr == NULL)
    die("Unable to allocate array");

  ptr -= offsety;

  for(int y=offsety; y<offsety + sizey; y++)
    ptr[y] = (double *)(&array[y][offsetx]) - offsetx;

  return ptr;
}

int main(int argc, char ** argv) {
  int    N;
  int    tile_size=32;  /* default tile size for tiling of local transpose */
  int    num_iterations;/* number of times to do the transpose             */
  int    tiling;        /* boolean: true if tiling is used                 */
  double total_bytes;   /* combined size of matrices                       */
  double start_time,    /* timing parameters                               */
         end_time, avgtime;

  /*********************************************************************
  ** read and test input parameters
  *********************************************************************/

  if(argc != 3 && argc != 4){
    if(MYTHREAD == 0)
      printf("Usage: %s <# iterations> <matrix order> [tile size]\n", *argv);
    upc_global_exit(EXIT_FAILURE);
  }

  num_iterations = atoi(*++argv);
  if(num_iterations < 1){
    if(MYTHREAD == 0)
      printf("ERROR: iterations must be >= 1 : %d \n", num_iterations);
    upc_global_exit(EXIT_FAILURE);
  }

  N = atoi(*++argv);
  if(N < 0){
    if(MYTHREAD == 0)
      printf("ERROR: Matrix Order must be greater than 0 : %d \n", N);
    upc_global_exit(EXIT_FAILURE);
  }

  if (argc == 4)
    tile_size = atoi(*++argv);

  /*a non-positive tile size means no tiling of the local transpose */
  tiling = (tile_size > 0) && (tile_size < N);
  if(!tiling)
    tile_size = N;

  if(MYTHREAD == 0)
    printf("UPC Transpose: THREADS=%d, num_iterations=%d, order=%d, tile_size=%d\n",
      THREADS, num_iterations, N, tile_size);

  /*********************************************************************
  ** Allocate memory for input and output matrices
  *********************************************************************/

  total_bytes = 2.0 * sizeof(double) * N * N;

  int sizex = N / THREADS;
  if(N % THREADS != 0)
      die("N %% THREADS != 0");
  int sizey = N;

  int myoffsetx = MYTHREAD * sizex;
  int myoffsety = 0;

  upc_barrier;

  debug("Allocating arrays (%d, %d), offset (%d, %d)", sizex, sizey, myoffsetx, myoffsety);
  local_shared_block_ptrs in_array  = shared_2d_array_alloc(sizex, sizey, myoffsetx, myoffsety);
  local_shared_block_ptrs out_array = shared_2d_array_alloc(sizex, sizey, myoffsetx, myoffsety);
  local_shared_block_ptrs buf_array = shared_2d_array_alloc(sizex, sizey, myoffsetx, myoffsety);

  in_arrays[MYTHREAD] = in_array;
  out_arrays[MYTHREAD] = out_array;
  buf_arrays[MYTHREAD] = buf_array;

  double **in_array_private = shared_2d_array_to_private(in_array, sizex, sizey, myoffsetx, myoffsety);
  double **out_array_private = shared_2d_array_to_private(out_array, sizex, sizey, myoffsetx, myoffsety);
  double **buf_array_private = shared_2d_array_to_private(buf_array, sizex, sizey, myoffsetx, myoffsety);

  upc_barrier;

  /*********************************************************************
  ** Initialize the matrices
  *********************************************************************/
  for(int y=myoffsety; y<myoffsety + sizey; y++){
    for(int x=myoffsetx; x<myoffsetx + sizex; x++){
      in_array_private[y][x] = (double) (x+N*y);
      out_array[y][x] = -1.0;
    }
  }
  upc_barrier;

  for(int y=myoffsety; y<myoffsety + sizey; y++){
    for(int x=myoffsetx; x<myoffsetx + sizex; x++){
      if(in_array_private[y][x] !=(double) (x+N*y))
        die("x=%d y=%d in_array=%f != %f", x, y, in_array[y][x], (x+N*y));
      if(out_array_private[y][x] != -1.0)
        die("out_array_private error");
    }
  }

  /*********************************************************************
  ** Transpose
  *********************************************************************/
  int transfer_size = sizex * sizex * sizeof(double);
  if(MYTHREAD == 0)
    debug("transfer size = %d", transfer_size);

  for(int iter=0; iter<=num_iterations; iter++){
    /* start timer after a warmup iteration */
    if(iter == 1){
      upc_barrier;
      start_time = wtime();
    }

    for(int i=0; i<THREADS; i++){
      int local_blk_id = (MYTHREAD + i) % THREADS;
      int remote_blk_id = MYTHREAD;
      int remote_thread = local_blk_id;

      upc_memget(&buf_array_private[local_blk_id * sizex][myoffsetx],
                  &in_arrays[remote_thread][remote_blk_id * sizex][remote_thread * sizex], transfer_size);

#define OUT_ARRAY(x,y) out_array_private[local_blk_id * sizex + x][myoffsetx + y]
#define BUF_ARRAY(x,y) buf_array_private[local_blk_id * sizex + x][myoffsetx + y]

      if(!tiling){
        for(int x=0; x<sizex; x++){
          for(int y=0; y<sizex; y++){
            OUT_ARRAY(x,y) = BUF_ARRAY(y,x);
          }
        }
      }
      else{
        for(int x=0; x<sizex; x+=tile_size){
          for(int y=0; y<sizex; y+=tile_size){
            for(int bx=x; bx<MIN(sizex, x+tile_size); bx++){
              for(int by=y; by<MIN(sizex, y+tile_size); by++){
                OUT_ARRAY(bx,by) = BUF_ARRAY(by,bx);
              }
            }
          }
        }
      }
    }
    upc_barrier;
  }

  upc_barrier;
  end_time = wtime();

  /*********************************************************************
  ** Analyze and output results.
  *********************************************************************/
  for(int y=myoffsety; y<myoffsety + sizey; y++){
    for(int x=myoffsetx; x<myoffsetx + sizex; x++){
      if(in_array_private[y][x] != (double)(x+ N*y))
        die("Error in input: x=%d y=%d", x, y);
      if(out_array_private[y][x] != (double)(y + N*x))
        die("x=%d y=%d in_array=%f != %f   %d %d", x, y, out_array[y][x], (double)(y + N*x), (int)(out_array[y][x]) % N, (int)(out_array[y][x]) / N);
    }
  }

  if(MYTHREAD == 0){
    printf("Solution validates\n");
    double transfer_size = 2 * N * N * sizeof(double);
    avgtime = (end_time - start_time) / num_iterations;
    double rate = transfer_size / avgtime / 1024 / 1024;
    printf("Rate (MB/s): %lf Avg time (s): %lf\n",rate, avgtime);
  }
}
