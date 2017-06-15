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

         Other than MPI or standard C functions, the following
         functions are used in this program:

          wtime()           Portable wall-timer interface.
          bail_out()        Determine global error and exit if nonzero.

HISTORY: Written by Tim Mattson, April 1999.
         Updated by Rob Van der Wijngaart, December 2005.
         Updated by Rob Van der Wijngaart, October 2006.
         Updated by Rob Van der Wijngaart, November 2014::
         - made variable names more consistent
         - put timing around entire iterative loop of transposes
         - fixed incorrect matrix block access; no separate function
           for local transpose of matrix block
         - reordered initialization and verification loops to
           produce unit stride
         - changed initialization values, such that the input matrix
           elements are: A(i,j) = i+order*j


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
o When tiling is applied to reduce TLB misses, each block gets
  accessed by tiles.
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
|           |Tile|      |           |                             |
|           |    |      |           |   Overall Matrix            |
|           |----       |           |                             |
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
#include <par-res-kern_mpi.h>

#define A(i,j)          A_p[(i+istart)+order*(j)]
#define B(i,j)          B_p[(i+istart)+order*(j)]
#define Work_in(p,i,j)  Work_in_p[(p)*Block_size+i+Block_order*(j)]
#define Work_out(p,i,j) Work_out_p[(p)*Block_size+i+Block_order*(j)]

int main(int argc, char ** argv)
{
  long Block_order;        /* number of columns owned by rank       */
  long Block_size;         /* size of a single block                */
  long Colblock_size;      /* size of column block                  */
  int Tile_order=32;       /* default Tile order                    */
  int tiling;              /* boolean: true if tiling is used       */
  int Num_procs;           /* number of ranks                       */
  long order;              /* order of overall matrix               */
  int send_to, recv_from;  /* ranks with which to communicate       */
  long bytes;              /* combined size of matrices             */
  int my_ID;               /* rank                                  */
  int root=0;              /* rank of root                          */
  int iterations;          /* number of times to do the transpose   */
  int i, j, it, jt, istart;/* dummies                               */
  int iter;                /* index of iteration                    */
  int phase;               /* phase inside staged communication     */
  int colstart;            /* starting column for owning rank       */
  int error;               /* error flag                            */
  double * RESTRICT A_p;   /* original matrix column block          */
  double * RESTRICT B_p;   /* transposed matrix column block        */
  double * RESTRICT Work_in_p;/* workspace for transpose function   */
  double * RESTRICT Work_out_p;/* workspace for transpose function  */
  double abserr,           /* absolute error                        */
         abserr_tot;       /* aggregate absolute error              */
  double epsilon = 1.e-8;  /* error tolerance                       */
  double local_trans_time, /* timing parameters                     */
         trans_time,
         avgtime;
  MPI_Win  rma_win = MPI_WIN_NULL;
  MPI_Info rma_winfo = MPI_INFO_NULL;
  int passive_target = 0;  /* use passive target RMA sync           */
#if MPI_VERSION >= 3
  int  flush_local  = 1;   /* flush local (or remote) after put     */
  int  flush_bundle = 1;   /* flush every <bundle> put calls        */
#endif

/*********************************************************************
** Initialize the MPI environment
*********************************************************************/
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_ID);
  MPI_Comm_size(MPI_COMM_WORLD, &Num_procs);

/*********************************************************************
** process, test and broadcast input parameters
*********************************************************************/
  error = 0;
  if (my_ID == root) {
    printf("Parallel Research Kernels version %s\n", PRKVERSION);
    printf("MPIRMA matrix transpose: B = A^T\n");

    if (argc <= 3){
      printf("Usage: %s <# iterations> <matrix order> [Tile size]"
             "[sync (0=fence, 1=flush)] [flush local?] [flush bundle]\n",
             *argv);
      error = 1; goto ENDOFTESTS;
    }

    iterations  = atoi(*++argv);
    if(iterations < 1){
      printf("ERROR: iterations must be >= 1 : %d \n",iterations);
      error = 1; goto ENDOFTESTS;
    }

    order = atol(*++argv);
    if (order < Num_procs) {
      printf("ERROR: matrix order %ld should at least # procs %d\n",
             order, Num_procs);
      error = 1; goto ENDOFTESTS;
    }
    if (order%Num_procs) {
      printf("ERROR: matrix order %ld should be divisible by # procs %d\n",
             order, Num_procs);
      error = 1; goto ENDOFTESTS;
    }

    if (argc >= 4) Tile_order     = atoi(*++argv);
    if (argc >= 5) passive_target = atoi(*++argv);
#if MPI_VERSION >= 3
    if (argc >= 6) flush_local    = atoi(*++argv);
    if (argc >= 7) flush_bundle   = atoi(*++argv);
#endif

    ENDOFTESTS:;
  }
  bail_out(error);

  if (my_ID == root) {
    printf("Number of ranks      = %d\n", Num_procs);
    printf("Matrix order         = %ld\n", order);
    printf("Number of iterations = %d\n", iterations);
    if ((Tile_order > 0) && (Tile_order < order))
          printf("Tile size            = %d\n", Tile_order);
    else  printf("Untiled\n");
    if (passive_target) {
#if MPI_VERSION < 3
        printf("Synchronization      = MPI_Win_(un)lock\n");
#else
        printf("Synchronization      = MPI_Win_flush%s (bundle=%d)\n", flush_local ? "_local" : "", flush_bundle);
#endif
    } else {
        printf("Synchronization      = MPI_Win_fence\n");
    }
  }

  /*  Broadcast input data to all ranks */
  MPI_Bcast (&order,          1, MPI_LONG, root, MPI_COMM_WORLD);
  MPI_Bcast (&iterations,     1, MPI_INT,  root, MPI_COMM_WORLD);
  MPI_Bcast (&Tile_order,     1, MPI_INT,  root, MPI_COMM_WORLD);
  MPI_Bcast (&passive_target, 1, MPI_INT,  root, MPI_COMM_WORLD);
#if MPI_VERSION >= 3
  MPI_Bcast (&flush_local,    1, MPI_INT,  root, MPI_COMM_WORLD);
  MPI_Bcast (&flush_bundle,   1, MPI_INT,  root, MPI_COMM_WORLD);
#endif

  /* a non-positive tile size means no tiling of the local transpose */
  tiling = (Tile_order > 0) && (Tile_order < order);
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

  /* debug message size effects */
  if (my_ID == root) {
    printf("Block_size           = %ld\n", Block_size);
  }

/*********************************************************************
** Create the column block of the test matrix, the row block of the
** transposed matrix, and workspace (workspace only if #procs>1)
*********************************************************************/
  A_p = (double *)prk_malloc(Colblock_size*sizeof(double));
  if (A_p == NULL){
    printf(" Error allocating space for original matrix on node %d\n",my_ID);
    error = 1;
  }
  bail_out(error);

  MPI_Info_create (&rma_winfo);
  MPI_Info_set (rma_winfo, "no locks", "true");
  B_p = (double *)prk_malloc(Colblock_size*sizeof(double));
  if (B_p == NULL){
    printf(" Error allocating space for transpose matrix on node %d\n",my_ID);
    error = 1;
  }
  bail_out(error);

  if (Num_procs>1) {
    Work_out_p = (double *) prk_malloc(Block_size*(Num_procs-1)*sizeof(double));
    if (Work_out_p == NULL){
      printf(" Error allocating space for work_out on node %d\n",my_ID);
      error = 1;
    }
    bail_out(error);

    PRK_Win_allocate(Block_size*(Num_procs-1)*sizeof(double), sizeof(double),
                     rma_winfo, MPI_COMM_WORLD, &Work_in_p, &rma_win);
    if (Work_in_p == NULL){
      printf(" Error allocating space for work on node %d\n",my_ID);
      error = 1;
    }
    bail_out(error);
  }

#if MPI_VERSION >= 3
  if (passive_target && Num_procs>1) {
    MPI_Win_lock_all(MPI_MODE_NOCHECK,rma_win);
  }
#endif

  /* Fill the original column matrix                                                */
  istart = 0;
  for (j=0;j<Block_order;j++) {
    for (i=0;i<order; i++) {
      A(i,j) = (double) (order*(j+colstart) + i);
      B(i,j) = 0.0;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  for (iter = 0; iter<=iterations; iter++) {

    /* start timer after a warmup iteration                                        */
    if (iter == 1) {
      MPI_Barrier(MPI_COMM_WORLD);
      local_trans_time = wtime();
    }

    /* do the local transpose                                                     */
    istart = colstart;
    if (!tiling) {
      for (i=0; i<Block_order; i++) {
        for (j=0; j<Block_order; j++) {
          B(j,i) += A(i,j);
          A(i,j) += 1.0;
        }
      }
    } else {
      for (i=0; i<Block_order; i+=Tile_order) {
        for (j=0; j<Block_order; j+=Tile_order) {
          for (it=i; it<MIN(Block_order,i+Tile_order); it++) {
            for (jt=j; jt<MIN(Block_order,j+Tile_order);jt++) {
              B(jt,it) += A(it,jt);
              A(it,jt) += 1.0;
            }
          }
        }
      }
    }

    if (!passive_target && Num_procs>1) {
      MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPRECEDE, rma_win);
    }

    for (phase=1; phase<Num_procs; phase++){
      send_to = (my_ID - phase + Num_procs)%Num_procs;

      istart = send_to*Block_order;
      if (!tiling) {
        for (i=0; i<Block_order; i++) {
          for (j=0; j<Block_order; j++) {
            Work_out(phase-1,j,i) = A(i,j);
            A(i,j) += 1.0;
          }
        }
      } else {
        for (i=0; i<Block_order; i+=Tile_order) {
          for (j=0; j<Block_order; j+=Tile_order) {
            for (it=i; it<MIN(Block_order,i+Tile_order); it++) {
              for (jt=j; jt<MIN(Block_order,j+Tile_order);jt++) {
                Work_out(phase-1,jt,it) = A(it,jt);
                A(it,jt) += 1.0;
              }
            }
          }
        }
      }

#if MPI_VERSION < 3
      if (passive_target) {
          MPI_Win_lock(MPI_LOCK_SHARED, send_to, MPI_MODE_NOCHECK, rma_win);
      }
#endif
      MPI_Put(Work_out_p+Block_size*(phase-1), Block_size, MPI_DOUBLE, send_to,
              Block_size*(phase-1), Block_size, MPI_DOUBLE, rma_win);

      if (passive_target) {
#if MPI_VERSION < 3
        MPI_Win_unlock(send_to, rma_win);
#else
        if (flush_bundle==1) {
          if (flush_local==1) {
              MPI_Win_flush_local(send_to, rma_win);
          } else {
              MPI_Win_flush(send_to, rma_win);
          }
        } else if ( (phase%flush_bundle) == 0) {
          /* Too lazy to record all targets, so let MPI do it internally (hopefully) */
          if (flush_local==1) {
              MPI_Win_flush_local_all(rma_win);
          } else {
              MPI_Win_flush_all(rma_win);
          }
        }
#endif
      }
    }  /* end of phase loop for puts  */
    if (Num_procs>1) {
      if (passive_target) {
#if MPI_VERSION >= 3
          MPI_Win_flush_all(rma_win);
#endif
          MPI_Barrier(MPI_COMM_WORLD);
      } else {
          MPI_Win_fence(MPI_MODE_NOSTORE, rma_win);
      }
    }

    for (phase=1; phase<Num_procs; phase++) {
      recv_from = (my_ID + phase)%Num_procs;
      istart = recv_from*Block_order;
      /* scatter received block to transposed matrix; no need to tile */
      for (j=0; j<Block_order; j++) {
        for (i=0; i<Block_order; i++) {
          B(i,j) += Work_in(phase-1,i,j);
        }
      }
    } /* end of phase loop for scatters */

    /* for the flush case we need to make sure we have consumed Work_in
       before overwriting it in the next iteration                    */
    if (Num_procs>1 && passive_target) {
      MPI_Barrier(MPI_COMM_WORLD);
    }

  } /* end of iterations */

  local_trans_time = wtime() - local_trans_time;
  MPI_Reduce(&local_trans_time, &trans_time, 1, MPI_DOUBLE, MPI_MAX, root,
             MPI_COMM_WORLD);

  abserr = 0.0;
  istart = 0;
  double addit = ((double)(iterations+1) * (double) (iterations))/2.0;
  for (j=0;j<Block_order;j++) {
    for (i=0;i<order; i++) {
      abserr += ABS(B(i,j) - ((double)(order*i + j+colstart)*(iterations+1)+addit));
    }
  }

  MPI_Reduce(&abserr, &abserr_tot, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

  if (my_ID == root) {
    if (abserr_tot < epsilon) {
      printf("Solution validates\n");
      avgtime = trans_time/(double)iterations;
      printf("Rate (MB/s): %lf Avg time (s): %lf\n",1.0E-06*bytes/avgtime, avgtime);
    }
    else {
      printf("ERROR: Aggregate absolute error %lf exceeds threshold %e\n", abserr_tot, epsilon);
      error = 1;
    }
  }

  bail_out(error);

  if (rma_win!=MPI_WIN_NULL) {
#if MPI_VERSION >=3
    if (passive_target) {
      MPI_Win_unlock_all(rma_win);
    }
#endif
    PRK_Win_free(&rma_win);
  }

  MPI_Finalize();
  exit(EXIT_SUCCESS);

}  /* end of main */
