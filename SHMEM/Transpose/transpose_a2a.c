/*
Copyright (c) 2013, Intel Corporation
Copyright (c) 2023

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

FUNCTIONS CALLED:

         Other than SHMEM or standard C functions, the following
         functions are used in this program:

          wtime()           Portable wall-timer interface.
          bail_out()        Determine global error and exit if nonzero.

HISTORY: Written by Tom St. John, July 2015.
         Rob vdW: Fixed race condition on synchronization flags, August 2015
         Marcin Rogowski: shmem_alltoall, July 2023

*******************************************************************/

/******************************************************************
                     Layout nomenclature
                     -------------------

o Each rank owns one block of columns (Colblock) of the overall
  matrix to be transposed, as well as of the transposed matrix.
o Colblock is stored contiguously in the memory of the rank.
  The stored format is column major, which means that matrix
  elements (i,j) and (i+1,j) are adjacent, and (i,j) and (i,j+1)
  are "order" words apart
o Colblock is logically composed of #ranks Blocks, but a Block is
  not stored contiguously in memory. Conceptually, the Block is
  the unit of data that gets communicated between ranks. Block i of
  rank j is locally transposed and gathered into a buffer called Work,
  which is sent to rank i, where it is scattered into Block j of the
  transposed matrix.
o The original and transposed matrices are called A and B

 -----------------------------------------------------------------
|           |           |           |                             |
| Colblock  |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|        -------------------------------                          |
|           |           |           |                             |
|           |  Block    |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|        -------------------------------                          |
|           |           |           |                             |
|           |           |           |   Overall Matrix            |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|        -------------------------------                          |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
 -----------------------------------------------------------------*/

#include <par-res-kern_general.h>
#include <par-res-kern_shmem.h>

#include <math.h>

int main(int argc, char ** argv)
{
  long Block_order;        /* number of columns owned by rank       */
  int Block_size;          /* size of a single block                */
  int Colblock_size;       /* size of column block                  */
  int Num_procs;           /* number of ranks                       */
  int order;               /* order of overall matrix               */
  int send_to, recv_from;  /* ranks with which to communicate       */
  long bytes;              /* combined size of matrices             */
  int my_ID;               /* rank                                  */
  int root=0;              /* rank of root                          */
  int iterations;          /* number of times to do the transpose   */
  long i, j, it, jt, istart;/* dummies                              */
  int iter;                /* index of iteration                    */
  int phase;               /* phase inside staged communication     */
  int colstart;            /* starting column for owning rank       */
  int error;               /* error flag                            */
  double * RESTRICT A_p;   /* original matrix column block          */
  double * RESTRICT B_p;   /* transposed matrix column block        */
  double * RESTRICT T_p;
  double epsilon = 1.e-8;  /* error tolerance                       */
  double avgtime;          /* timing parameters                     */
  long   *pSync_bcast;     /* work space for collectives            */
  long   *pSync_reduce;    /* work space for collectives            */
  double *pWrk;            /* work space for SHMEM collectives      */
  double *local_trans_time,
         *trans_time;      /* timing parameters                     */
  double *abserr,
         *abserr_tot;      /* local and aggregate error             */
#if !BARRIER_SYNCH
  int    *recv_flag;       /* synchronization flags: data received  */
  int    *send_flag;       /* synchronization flags: receiver ready */
#endif
  int    *arguments;       /* command line arguments                */

/*********************************************************************
** Initialize the SHMEM environment
*********************************************************************/

  prk_shmem_init();
  my_ID=prk_shmem_my_pe();
  Num_procs=prk_shmem_n_pes();

  if (my_ID == root) {
    printf("Parallel Research Kernels version %s\n", PRKVERSION);
    printf("SHMEM matrix transpose: B = A^T\n");
  }

// initialize sync variables for error checks
  pSync_bcast      = (long *)   prk_shmem_align(prk_get_alignment(),PRK_SHMEM_BCAST_SYNC_SIZE*sizeof(long));
  pSync_reduce     = (long *)   prk_shmem_align(prk_get_alignment(),PRK_SHMEM_REDUCE_SYNC_SIZE*sizeof(long));
  pWrk             = (double *) prk_shmem_align(prk_get_alignment(),sizeof(double) * PRK_SHMEM_REDUCE_MIN_WRKDATA_SIZE);
  local_trans_time = (double *) prk_shmem_align(prk_get_alignment(),sizeof(double));
  trans_time       = (double *) prk_shmem_align(prk_get_alignment(),sizeof(double));
  arguments        = (int *)    prk_shmem_align(prk_get_alignment(),3*sizeof(int));
  abserr           = (double *) prk_shmem_align(prk_get_alignment(),2*sizeof(double));
  if (!pSync_bcast || !pSync_reduce || !pWrk || !local_trans_time ||
      !trans_time || !arguments || !abserr) {
    printf("Rank %d could not allocate scalar work space on symm heap\n", my_ID);
    error = 1;
    goto ENDOFTESTS;
  }

  for(i=0;i<PRK_SHMEM_BCAST_SYNC_SIZE;i++)
    pSync_bcast[i]=PRK_SHMEM_SYNC_VALUE;

  for(i=0;i<PRK_SHMEM_REDUCE_SYNC_SIZE;i++)
    pSync_reduce[i]=PRK_SHMEM_SYNC_VALUE;

/*********************************************************************
** process, test and broadcast input parameters
*********************************************************************/
  error = 0;
  if (my_ID == root) {
    if (argc != 3){
      printf("Usage: %s <# iterations> <matrix order>\n",
                                                               *argv);
      error = 1; goto ENDOFTESTS;
    }

    iterations  = atoi(*++argv);
    arguments[0]=iterations;
    if(iterations < 1){
      printf("ERROR: iterations must be >= 1 : %d \n",iterations);
      error = 1; goto ENDOFTESTS;
    }

    order = atoi(*++argv);
    arguments[1]=order;
    if (order < Num_procs) {
      printf("ERROR: matrix order %d should at least # procs %d\n",
             order, Num_procs);
      error = 1; goto ENDOFTESTS;
    }
    if (order%Num_procs) {
      printf("ERROR: matrix order %d should be divisible by # procs %d\n",
             order, Num_procs);
      error = 1; goto ENDOFTESTS;
    }

    ENDOFTESTS:;
  }
  bail_out(error);

  if (my_ID == root) {
    printf("Number of ranks      = %d\n", Num_procs);
    printf("Matrix order         = %d\n", order);
    printf("Number of iterations = %d\n", iterations);
  }

  /*  Broadcast input data to all ranks */
  shmem_broadcast32(&arguments[0], &arguments[0], 2, root, 0, 0, Num_procs, pSync_bcast);
  shmem_barrier_all();

  iterations=arguments[0];
  order=arguments[1];

  shmem_barrier_all();
  prk_shmem_free(arguments);

  bytes = 2 * sizeof(double) * order * order;

/*********************************************************************
** The matrix is broken up into column blocks that are mapped one to a
** rank.  Each column block is made up of Num_procs smaller square
** blocks of order block_order.
*********************************************************************/

  Block_order    = order/Num_procs;
  colstart       = Block_order * my_ID;
  Colblock_size  = order * Block_order;
  Block_size     = Block_order * Block_order;

/*********************************************************************
** Create the column block of the test matrix, the row block of the
** transposed matrix, and workspace (workspace only if #procs>1)
*********************************************************************/
  A_p = (double *)prk_malloc(Colblock_size*sizeof(double));
  T_p = (double *)prk_malloc(Colblock_size*sizeof(double));
  if (A_p == NULL){
    printf(" Error allocating space for original matrix on node %d\n",my_ID);
    error = 1;
  }
  bail_out(error);

  B_p = (double *)prk_malloc(Colblock_size*sizeof(double));
  if (B_p == NULL){
    printf(" Error allocating space for transpose matrix on node %d\n",my_ID);
    error = 1;
  }
  bail_out(error);

  /* Fill the original column matrices                                              */
  for (i=0;i<order; i++)
    for (j=0;j<Block_order;j++)
    {
      A_p[i*Block_order+j] = (double) (my_ID*Block_order + i*order +j);
      B_p[i*Block_order+j] = 0.0;
      T_p[i*Block_order+j] = 0.0;
    }

  shmem_barrier_all();

  for (iter = 0; iter<=iterations; iter++){

    /* start timer after a warmup iteration                                        */
    if (iter == 1) {
      shmem_barrier_all();
      local_trans_time[0] = wtime();
    }

    shmem_barrier_all();
    shmem_double_alltoall(SHMEM_TEAM_WORLD, T_p, A_p, Block_order*Block_order);

    for (phase=0; phase<Num_procs; phase++){
      int lo = Block_order*Block_order*phase;

      for (i = 0; i < Block_order; i++)
          for (j = 0; j < Block_order; j++)
              B_p[lo + i + Block_order * j] += T_p[lo + j + Block_order * i];
    }  /* end of phase loop  */

    for (j=0; j<Colblock_size; j++)
    {
      A_p[j] += 1.0;
    }

  } /* end of iterations */

  shmem_barrier_all();
  local_trans_time[0] = wtime() - local_trans_time[0];

  shmem_barrier_all();
  shmem_double_max_to_all(trans_time, local_trans_time, 1, 0, 0, Num_procs, pWrk, pSync_reduce);

  abserr[0] = 0.0;
  double addit = 0.5 * ( (iterations+1.) * (double)iterations );
  for (i=0;i<order; i++)
  {
    for (j=0;j<Block_order;j++)
    {
      const double temp = (order*(my_ID*Block_order+j)+i) * (iterations+1) + addit;
      abserr[0] += ABS(B_p[i*Block_order+j] - temp);
    }
  }

  shmem_barrier_all();
  shmem_double_sum_to_all(&(abserr[1]), &(abserr[0]), 1, 0, 0, Num_procs, pWrk, pSync_reduce);

  if (abserr[1] <= epsilon) {
    avgtime = trans_time[0]/(double)iterations;
    if (my_ID == root) {
      printf("Solution validates\n");
      printf("Rate (MB/s): %lf Avg time (s): %lf\n",1.0E-06*bytes/avgtime, avgtime);
#ifdef VERBOSE
      printf("Summed errors: %30.15lf \n", abserr[1]);
#endif
    }
  } else {
    error = 1;
    if (my_ID == root) {
      printf("ERROR: Aggregate squared error %30.15lf exceeds threshold %30.15lf\n", abserr[1], epsilon);
    }
    fflush(stdout);
    printf("ERROR: PE=%d, error = %30.15lf\n", my_ID, abserr[0]);
  }

  bail_out(error);

  prk_shmem_free(pSync_bcast);
  prk_shmem_free(pSync_reduce);
  prk_shmem_free(pWrk);

  prk_shmem_finalize();
  exit(EXIT_SUCCESS);

}  /* end of main */

