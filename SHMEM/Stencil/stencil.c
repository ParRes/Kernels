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

               <progname> <# iterations> <grid size>

         The output consists of diagnostics to make sure the
         algorithm worked, and of timing statistics.

FUNCTIONS CALLED:

         Other than SHMEM or standard C functions, the following
         functions are used in this program:

         wtime()
         bail_out()

HISTORY: - Written by Tom St. John, July 2015.
         - Adapted by Rob Van der Wijngaart to introduce double buffering, December 2015

*********************************************************************************/

#include <par-res-kern_general.h>
#include <par-res-kern_shmem.h>

#if DOUBLE
  #define DTYPE     double
  #define EPSILON   1.e-8
  #define COEFX     1.0
  #define COEFY     1.0
  #define FSTR      "%lf"
#else
  #define DTYPE     float
  #define EPSILON   0.0001f
  #define COEFX     1.0f
  #define COEFY     1.0f
  #define FSTR      "%f"
#endif

/* define shorthand for indexing multi-dimensional arrays with offsets           */
#define INDEXIN(i,j)  (i+RADIUS+(long)(j+RADIUS)*(long)(width[0]+2*RADIUS))
/* need to add offset of RADIUS to j to account for ghost points                 */
#define IN(i,j)       in[INDEXIN(i-istart,j-jstart)]
#define INDEXOUT(i,j) (i+(j)*(width[0]))
#define OUT(i,j)      out[INDEXOUT(i-istart,j-jstart)]
#define WEIGHT(ii,jj) weight[ii+RADIUS][jj+RADIUS]

int main(int argc, char ** argv) {

  int    Num_procs;       /* number of ranks                                     */
  int    Num_procsx, Num_procsy; /* number of ranks in each coord direction      */
  int    my_ID;           /* SHMEM rank                                          */
  int    my_IDx, my_IDy;  /* coordinates of rank in rank grid                    */
  int    right_nbr;       /* global rank of right neighboring tile               */
  int    left_nbr;        /* global rank of left neighboring tile                */
  int    top_nbr;         /* global rank of top neighboring tile                 */
  int    bottom_nbr;      /* global rank of bottom neighboring tile              */
  DTYPE *top_buf_out;     /* communication buffer                                */
  DTYPE *top_buf_in[2];   /*       "         "                                   */
  DTYPE *bottom_buf_out;  /*       "         "                                   */
  DTYPE *bottom_buf_in[2];/*       "         "                                   */
  DTYPE *right_buf_out;   /*       "         "                                   */
  DTYPE *right_buf_in[2]; /*       "         "                                   */
  DTYPE *left_buf_out;    /*       "         "                                   */
  DTYPE *left_buf_in[2];  /*       "         "                                   */
  int    root = 0;
  int    *width, *height, /* linear global and local grid dimension              */
         *maxwidth, *maxheight;
  int    n;
  int    i, j, ii, jj, kk, it, jt, iter, leftover;  /* dummies                   */
  int    istart, iend;    /* bounds of grid tile assigned to calling rank        */
  int    jstart, jend;    /* bounds of grid tile assigned to calling rank        */
  DTYPE  reference_norm;
  DTYPE  f_active_points; /* interior of grid with respect to stencil            */
  int    stencil_size;    /* number of points in the stencil                     */
  DTYPE  flops;           /* floating point ops per iteration                    */
  int    iterations;      /* number of times to run the algorithm                */
  double avgtime,         /* timing parameters                                   */
         *local_stencil_time, *stencil_time;
  DTYPE  * RESTRICT in;   /* input grid values                                   */
  DTYPE  * RESTRICT out;  /* output grid values                                  */
  long   total_length_in; /* total required length to store input array          */
  long   total_length_out;/* total required length to store output array         */
  int    error=0;         /* error flag                                          */
  DTYPE  weight[2*RADIUS+1][2*RADIUS+1]; /* weights of points in the stencil     */
  int    *arguments;      /* command line parameters                             */
  int    count_case=4;    /* number of neighbors of a rank                       */
  long   *pSync_bcast;    /* work space for collectives                          */
  long   *pSync_reduce;   /* work space for collectives                          */
  double *pWrk_time;      /* work space for collectives                          */
  DTYPE  *pWrk_norm;      /* work space for collectives                          */
  int    *pWrk_dim;       /* work space for collectives                          */
  int    *iterflag;       /* synchronization flags                               */
  int    sw;              /* double buffering switch                             */
  DTYPE  *local_norm, *norm; /* local and global error norms                     */

  /*******************************************************************************
  ** Initialize the SHMEM environment
  ********************************************************************************/
  prk_shmem_init();

  my_ID=prk_shmem_my_pe();
  Num_procs=prk_shmem_n_pes();

  pSync_bcast        = (long *)   prk_shmem_align(prk_get_alignment(),PRK_SHMEM_BCAST_SYNC_SIZE*sizeof(long));
  pSync_reduce       = (long *)   prk_shmem_align(prk_get_alignment(),PRK_SHMEM_REDUCE_SYNC_SIZE*sizeof(long));
  pWrk_time          = (double *) prk_shmem_align(prk_get_alignment(),PRK_SHMEM_REDUCE_MIN_WRKDATA_SIZE*sizeof(double));
  pWrk_norm          = (DTYPE *)  prk_shmem_align(prk_get_alignment(),PRK_SHMEM_REDUCE_MIN_WRKDATA_SIZE*sizeof(DTYPE));
  pWrk_dim           = (int *)    prk_shmem_align(prk_get_alignment(),PRK_SHMEM_REDUCE_MIN_WRKDATA_SIZE*sizeof(int));
  local_stencil_time = (double *) prk_shmem_align(prk_get_alignment(),sizeof(double));
  stencil_time       = (double *) prk_shmem_align(prk_get_alignment(),sizeof(double));
  local_norm         = (DTYPE *)  prk_shmem_align(prk_get_alignment(),sizeof(DTYPE));
  norm               = (DTYPE *)  prk_shmem_align(prk_get_alignment(),sizeof(DTYPE));
  iterflag           = (int *)    prk_shmem_align(prk_get_alignment(),2*sizeof(int));
  width              = (int *)    prk_shmem_align(prk_get_alignment(),sizeof(int));
  maxwidth           = (int *)    prk_shmem_align(prk_get_alignment(),sizeof(int));
  height             = (int *)    prk_shmem_align(prk_get_alignment(),sizeof(int));
  maxheight          = (int *)    prk_shmem_align(prk_get_alignment(),sizeof(int));

  if (!(pSync_bcast && pSync_reduce && pWrk_time && pWrk_norm && iterflag &&
	local_stencil_time && stencil_time && local_norm && norm))
  {
    printf("Could not allocate scalar variables on rank %d\n", my_ID);
    error = 1;
  }
  bail_out(error);

  for(i=0;i<PRK_SHMEM_BCAST_SYNC_SIZE;i++)
    pSync_bcast[i]=PRK_SHMEM_SYNC_VALUE;

  for(i=0;i<PRK_SHMEM_REDUCE_SYNC_SIZE;i++)
    pSync_reduce[i]=PRK_SHMEM_SYNC_VALUE;

  arguments=(int*)prk_shmem_align(prk_get_alignment(),2*sizeof(int));

  /*******************************************************************************
  ** process, test, and broadcast input parameters
  ********************************************************************************/

  if (my_ID == root) {
#if !STAR
    printf("ERROR: Compact stencil not supported\n");
    error = 1;
    goto ENDOFTESTS;
#endif

    if (argc != 3){
      printf("Usage: %s <# iterations> <array dimension> \n",
             *argv);
      error = 1;
      goto ENDOFTESTS;
    }

    iterations  = atoi(*++argv);
    arguments[0]=iterations;

    if (iterations < 1){
      printf("ERROR: iterations must be >= 1 : %d \n",iterations);
      error = 1;
      goto ENDOFTESTS;
    }

    n  = atoi(*++argv);
    arguments[1]=n;
    long nsquare = (long)n * (long)n;

    if (nsquare < Num_procs){
      printf("ERROR: grid size must be at least # ranks: %ld\n", nsquare);
      error = 1;
      goto ENDOFTESTS;
    }

    if (RADIUS < 0) {
      printf("ERROR: Stencil radius %d should be non-negative\n", RADIUS);
      error = 1;
      goto ENDOFTESTS;
    }

    if (2*RADIUS +1 > n) {
      printf("ERROR: Stencil radius %d exceeds grid size %d\n", RADIUS, n);
      error = 1;
      goto ENDOFTESTS;
    }

    ENDOFTESTS:;
  }
  bail_out(error);

  /* determine best way to create a 2D grid of ranks (closest to square)     */
  factor(Num_procs, &Num_procsx, &Num_procsy);

  my_IDx = my_ID%Num_procsx;
  my_IDy = my_ID/Num_procsx;
  /* compute neighbors; don't worry about dropping off the edges of the grid */
  right_nbr  = my_ID+1;
  left_nbr   = my_ID-1;
  top_nbr    = my_ID+Num_procsx;
  bottom_nbr = my_ID-Num_procsx;

  iterflag[0] = iterflag[1] = 0;

  if(my_IDx==0)            count_case--;
  if(my_IDx==Num_procsx-1) count_case--;
  if(my_IDy==0)            count_case--;
  if(my_IDy==Num_procsy-1) count_case--;

  if (my_ID == root) {
    printf("Parallel Research Kernels version %s\n", PRKVERSION);
    printf("SHMEM stencil execution on 2D grid\n");
    printf("Number of ranks        = %d\n", Num_procs);
    printf("Grid size              = %d\n", n);
    printf("Radius of stencil      = %d\n", RADIUS);
    printf("Tiles in x/y-direction = %d/%d\n", Num_procsx, Num_procsy);
    printf("Type of stencil        = star\n");
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
#if SPLITFENCE
    printf("Split fence            = ON\n");
#else
    printf("Split fence            = OFF\n");
#endif
    printf("Number of iterations   = %d\n", iterations);
  }

  shmem_broadcast32(&arguments[0], &arguments[0], 2, root, 0, 0, Num_procs, pSync_bcast);
  shmem_barrier_all();

  iterations=arguments[0];
  n=arguments[1];

  shmem_barrier_all();
  prk_shmem_free(arguments);

  /* compute amount of space required for input and solution arrays             */

  width[0] = n/Num_procsx;
  leftover = n%Num_procsx;
  if (my_IDx<leftover) {
    istart = (width[0]+1) * my_IDx;
    iend = istart + width[0] + 1;
  }
  else {
    istart = (width[0]+1) * leftover + width[0] * (my_IDx-leftover);
    iend = istart + width[0];
  }

  width[0] = iend - istart + 1;
  if (width[0] == 0) {
    printf("ERROR: rank %d has no work to do\n", my_ID);
    error = 1;
  }
  bail_out(error);

  height[0] = n/Num_procsy;
  leftover = n%Num_procsy;
  if (my_IDy<leftover) {
    jstart = (height[0]+1) * my_IDy;
    jend = jstart + height[0] + 1;
  }
  else {
    jstart = (height[0]+1) * leftover + height[0] * (my_IDy-leftover);
    jend = jstart + height[0];
  }

  height[0] = jend - jstart + 1;
  if (height == 0) {
    printf("ERROR: rank %d has no work to do\n", my_ID);
    error = 1;
  }
  bail_out(error);

  if (width[0] < RADIUS || height[0] < RADIUS) {
    printf("ERROR: rank %d has work tile smaller then stencil radius\n",
           my_ID);
    error = 1;
  }
  bail_out(error);

  total_length_in = (width[0]+2*RADIUS);
  total_length_in *= (height[0]+2*RADIUS);
  total_length_in *= sizeof(DTYPE);

  total_length_out = width[0];
  total_length_out *= height[0];
  total_length_out *= sizeof(DTYPE);

  in  = (DTYPE *) prk_malloc(total_length_in);
  out = (DTYPE *) prk_malloc(total_length_out);
  if (!in || !out) {
    printf("ERROR: rank %d could not allocate space for input/output array\n",
            my_ID);
    error = 1;
  }

  bail_out(error);

  shmem_barrier_all();

  shmem_int_max_to_all(&maxwidth[0], &width[0], 1, 0, 0, Num_procs, pWrk_dim, pSync_reduce);

  shmem_barrier_all();

  shmem_int_max_to_all(&maxheight[0], &height[0], 1, 0, 0, Num_procs, pWrk_dim, pSync_reduce);

  /* fill the stencil weights to reflect a discrete divergence operator         */
  for (jj=-RADIUS; jj<=RADIUS; jj++) for (ii=-RADIUS; ii<=RADIUS; ii++)
    WEIGHT(ii,jj) = (DTYPE) 0.0;
  stencil_size = 4*RADIUS+1;

  for (ii=1; ii<=RADIUS; ii++) {
    WEIGHT(0, ii) = WEIGHT( ii,0) =  (DTYPE) (1.0/(2.0*ii*RADIUS));
    WEIGHT(0,-ii) = WEIGHT(-ii,0) = -(DTYPE) (1.0/(2.0*ii*RADIUS));
  }

  norm[0] = (DTYPE) 0.0;
  f_active_points = (DTYPE) (n-2*RADIUS)*(DTYPE) (n-2*RADIUS);

  /* intialize the input and output arrays                                     */
  for (j=jstart; j<jend; j++) for (i=istart; i<iend; i++) {
    IN(i,j)  = COEFX*i+COEFY*j;
    OUT(i,j) = (DTYPE)0.0;
  }

  /* allocate communication buffers for halo values                            */
  top_buf_out=(DTYPE*)prk_shmem_malloc(2*sizeof(DTYPE)*RADIUS*maxwidth[0]);
  if (!top_buf_out) {
    printf("ERROR: Rank %d could not allocate output comm buffers for y-direction\n", my_ID);
    error = 1;
  }
  bail_out(error);
  bottom_buf_out = top_buf_out+RADIUS*maxwidth[0];

  top_buf_in[0]=(DTYPE*)prk_shmem_align(prk_get_alignment(),4*sizeof(DTYPE)*RADIUS*maxwidth[0]);
  if(!top_buf_in)
  {
    printf("ERROR: Rank %d could not allocate input comm buffers for y-direction\n", my_ID);
    error=1;
  }
  bail_out(error);

  top_buf_in[1]    = top_buf_in[0]    + RADIUS*maxwidth[0];
  bottom_buf_in[0] = top_buf_in[1]    + RADIUS*maxwidth[0];
  bottom_buf_in[1] = bottom_buf_in[0] + RADIUS*maxwidth[0];

  right_buf_out=(DTYPE*)prk_shmem_malloc(2*sizeof(DTYPE)*RADIUS*maxheight[0]);
  if (!right_buf_out) {
    printf("ERROR: Rank %d could not allocate output comm buffers for x-direction\n", my_ID);
    error = 1;
  }
  bail_out(error);
  left_buf_out=right_buf_out+RADIUS*maxheight[0];

  right_buf_in[0]=(DTYPE*)prk_shmem_align(prk_get_alignment(),4*sizeof(DTYPE)*RADIUS*maxheight[0]);
  if(!right_buf_in)
  {
    printf("ERROR: Rank %d could not allocate input comm buffers for x-dimension\n", my_ID);
    error=1;
  }
  bail_out(error);
  right_buf_in[1] = right_buf_in[0] + RADIUS*maxheight[0];
  left_buf_in[0]  = right_buf_in[1] + RADIUS*maxheight[0];
  left_buf_in[1]  = left_buf_in[0]  + RADIUS*maxheight[0];

  /* make sure all symmetric heaps are allocated before being used  */
  shmem_barrier_all();

  for (iter = 0; iter<=iterations; iter++){

    /* start timer after a warmup iteration */
    if (iter == 1) {
      shmem_barrier_all();
      local_stencil_time[0] = wtime();
    }
    /* sw determines which incoming buffer to select */
    sw = iter%2;

    /* need to fetch ghost point data from neighbors */

    if (my_IDy < Num_procsy-1) {
      for (kk=0,j=jend-RADIUS; j<=jend-1; j++) for (i=istart; i<=iend; i++) {
          top_buf_out[kk++]= IN(i,j);
      }
      shmem_putmem(bottom_buf_in[sw], top_buf_out, RADIUS*width[0]*sizeof(DTYPE), top_nbr);
#if SPLITFENCE
      shmem_fence();
      shmem_int_inc(&iterflag[sw], top_nbr);
#endif
    }
    if (my_IDy > 0) {
      for (kk=0,j=jstart; j<=jstart+RADIUS-1; j++) for (i=istart; i<=iend; i++) {
          bottom_buf_out[kk++]= IN(i,j);
      }
      shmem_putmem(top_buf_in[sw], bottom_buf_out, RADIUS*width[0]*sizeof(DTYPE), bottom_nbr);
#if SPLITFENCE
      shmem_fence();
      shmem_int_inc(&iterflag[sw], bottom_nbr);
#endif
    }

    if(my_IDx < Num_procsx-1) {
      for(kk=0,j=jstart;j<=jend;j++) for(i=iend-RADIUS;i<=iend-1;i++) {
	right_buf_out[kk++]=IN(i,j);
      }
      shmem_putmem(left_buf_in[sw], right_buf_out, RADIUS*height[0]*sizeof(DTYPE), right_nbr);
#if SPLITFENCE
      shmem_fence();
      shmem_int_inc(&iterflag[sw], right_nbr);
#endif
    }

    if(my_IDx>0) {
      for(kk=0,j=jstart;j<=jend;j++) for(i=istart;i<=istart+RADIUS-1;i++) {
	left_buf_out[kk++]=IN(i,j);
      }
      shmem_putmem(right_buf_in[sw], left_buf_out, RADIUS*height[0]*sizeof(DTYPE), left_nbr);
#if SPLITFENCE
      shmem_fence();
      shmem_int_inc(&iterflag[sw], left_nbr);
#endif
    }

#if SPLITFENCE == 0
    shmem_fence();
    if(my_IDy<Num_procsy-1) shmem_int_inc(&iterflag[sw], top_nbr);
    if(my_IDy>0)            shmem_int_inc(&iterflag[sw], bottom_nbr);
    if(my_IDx<Num_procsx-1) shmem_int_inc(&iterflag[sw], right_nbr);
    if(my_IDx>0)            shmem_int_inc(&iterflag[sw], left_nbr);
#endif

    shmem_int_wait_until(&iterflag[sw], SHMEM_CMP_EQ, count_case*(iter/2+1));

    if (my_IDy < Num_procsy-1) {
      for (kk=0,j=jend; j<=jend+RADIUS-1; j++) for (i=istart; i<=iend; i++) {
          IN(i,j) = top_buf_in[sw][kk++];
      }
    }
    if (my_IDy > 0) {
      for (kk=0,j=jstart-RADIUS; j<=jstart-1; j++) for (i=istart; i<=iend; i++) {
          IN(i,j) = bottom_buf_in[sw][kk++];
      }
    }

    if (my_IDx < Num_procsx-1) {
      for (kk=0,j=jstart; j<=jend; j++) for (i=iend; i<=iend+RADIUS-1; i++) {
          IN(i,j) = right_buf_in[sw][kk++];
      }
    }
    if (my_IDx > 0) {
      for (kk=0,j=jstart; j<=jend; j++) for (i=istart-RADIUS; i<=istart-1; i++) {
          IN(i,j) = left_buf_in[sw][kk++];
      }
    }

    /* Apply the stencil operator */
    for (j=MAX(jstart,RADIUS); j<=MIN(n-RADIUS-1,jend); j++) {
      for (i=MAX(istart,RADIUS); i<=MIN(n-RADIUS-1,iend); i++) {
        #if LOOPGEN
          #include "loop_body_star.incl"
        #else
          for (jj=-RADIUS; jj<=RADIUS; jj++) OUT(i,j) += WEIGHT(0,jj)*IN(i,j+jj);
          for (ii=-RADIUS; ii<0; ii++)       OUT(i,j) += WEIGHT(ii,0)*IN(i+ii,j);
          for (ii=1; ii<=RADIUS; ii++)       OUT(i,j) += WEIGHT(ii,0)*IN(i+ii,j);
        #endif
      }
    }

    /* add constant to solution to force refresh of neighbor data, if any */
    for (j=jstart; j<jend; j++) for (i=istart; i<iend; i++) IN(i,j)+= 1.0;

  }

  local_stencil_time[0] = wtime() - local_stencil_time[0];

  shmem_barrier_all();

  shmem_double_max_to_all(&stencil_time[0], &local_stencil_time[0], 1, 0, 0,
                          Num_procs, pWrk_time, pSync_reduce);

  /* compute L1 norm in parallel                                                */
  local_norm[0] = (DTYPE) 0.0;
  for (j=MAX(jstart,RADIUS); j<MIN(n-RADIUS,jend); j++) {
    for (i=MAX(istart,RADIUS); i<MIN(n-RADIUS,iend); i++) {
      local_norm[0] += (DTYPE)ABS(OUT(i,j));
    }
  }

  shmem_barrier_all();

#if DOUBLE
  shmem_double_sum_to_all(&norm[0], &local_norm[0], 1, 0, 0, Num_procs, pWrk_norm, pSync_reduce);
#else
  shmem_float_sum_to_all(&norm[0], &local_norm[0], 1, 0, 0, Num_procs, pWrk_norm, pSync_reduce);
#endif

  /*******************************************************************************
  ** Analyze and output results.
  ********************************************************************************/

/* verify correctness                                                            */
  if (my_ID == root) {
    norm[0] /= f_active_points;
    if (RADIUS > 0) {
      reference_norm = (DTYPE) (iterations+1) * (COEFX + COEFY);
    }
    else {
      reference_norm = (DTYPE) 0.0;
    }
    if (ABS(norm[0]-reference_norm) > EPSILON) {
      printf("ERROR: L1 norm = "FSTR", Reference L1 norm = "FSTR"\n",
             norm[0], reference_norm);
      error = 1;
    }
    else {
      printf("Solution validates\n");
#if VERBOSE
      printf("Reference L1 norm = "FSTR", L1 norm = "FSTR"\n",
             reference_norm, norm[0]);
#endif
    }
  }
  bail_out(error);

  if (my_ID == root) {
    /* flops/stencil: 2 flops (fma) for each point in the stencil,
       plus one flop for the update of the input of the array        */
    flops = (DTYPE) (2*stencil_size+1) * f_active_points;
    avgtime = stencil_time[0]/iterations;
    printf("Rate (MFlops/s): "FSTR"  Avg time (s): %lf\n",
           1.0E-06 * flops/avgtime, avgtime);
  }

  prk_shmem_free(top_buf_in[0]);
  prk_shmem_free(right_buf_in[0]);
  prk_shmem_free(top_buf_out);
  prk_shmem_free(right_buf_out);

  prk_shmem_free(pSync_bcast);
  prk_shmem_free(pSync_reduce);
  prk_shmem_free(pWrk_time);
  prk_shmem_free(pWrk_norm);
  prk_shmem_free(pWrk_dim);
  prk_shmem_free(width);
  prk_shmem_free(height);
  prk_shmem_free(maxwidth);
  prk_shmem_free(maxheight);

  prk_shmem_finalize();

  exit(EXIT_SUCCESS);
}
