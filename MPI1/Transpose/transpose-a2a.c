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
#include <par-res-kern_mpi.h>

int main(int argc, char ** argv)
{
  long Block_order;        /* number of columns owned by rank       */
  long Colblock_size;      /* size of column block                  */
  int Num_procs;           /* number of ranks                       */
  long order;              /* order of overall matrix               */
  long bytes;              /* combined size of matrices             */
  int my_ID;               /* rank                                  */
  int root=0;              /* rank of root                          */
  int iterations;          /* number of times to do the transpose   */
  int i, j;                /* dummies                               */
  int iter;                /* index of iteration                    */
  int phase;               /* phase inside staged communication     */
  int error;               /* error flag                            */
  double * RESTRICT A_p;   /* original matrix column block          */
  double * RESTRICT B_p;   /* transposed matrix column block        */
  double * RESTRICT T_p;
  double abserr,           /* absolute error                        */
         abserr_tot;       /* aggregate absolute error              */
  double epsilon = 1.e-8;  /* error tolerance                       */
  double local_trans_time, /* timing parameters                     */
         trans_time,
         avgtime;

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
  if (my_ID == root)
  {
    printf("Parallel Research Kernels version %s\n", PRKVERSION);
    printf("MPI matrix transpose: B = A^T\n");

    if (argc != 3)
    {
      printf("Usage: %s <# iterations> <matrix order>\n", *argv);
      error = 1; goto ENDOFTESTS;
    }

    iterations  = atoi(*++argv);
    if(iterations < 1)
    {
      printf("ERROR: iterations must be >= 1 : %d \n",iterations);
      error = 1; goto ENDOFTESTS;
    }

    order = atol(*++argv);
    if (order < Num_procs)
    {
      printf("ERROR: matrix order %ld should at least # procs %d\n",
             order, Num_procs);
      error = 1; goto ENDOFTESTS;
    }

    if (order%Num_procs)
    {
      printf("ERROR: matrix order %ld should be divisible by # procs %d\n",
             order, Num_procs);
      error = 1; goto ENDOFTESTS;
    }

    ENDOFTESTS:;
  }
  bail_out(error);

  if (my_ID == root)
  {
    printf("Number of ranks      = %d\n", Num_procs);
    printf("Matrix order         = %ld\n", order);
    printf("Number of iterations = %d\n", iterations);
    printf("Alltoall\n");
  }

  /*  Broadcast input data to all ranks */
  MPI_Bcast(&order,      1, MPI_LONG, root, MPI_COMM_WORLD);
  MPI_Bcast(&iterations, 1, MPI_INT,  root, MPI_COMM_WORLD);

  /* a non-positive tile size means no tiling of the local transpose */
  bytes = 2 * sizeof(double) * order * order;

/*********************************************************************
** The matrix is broken up into column blocks that are mapped one to a
** rank.  Each column block is made up of Num_procs smaller square
** blocks of order block_order.
*********************************************************************/

  Block_order    = order/Num_procs;
  Colblock_size  = order * Block_order;

/*********************************************************************
** Create the column block of the test matrix, the row block of the
** transposed matrix, and workspace (workspace only if #procs>1)
*********************************************************************/
  A_p = (double *)prk_malloc(Colblock_size*sizeof(double));
  T_p = (double *)prk_malloc(Colblock_size*sizeof(double));
  if (A_p == NULL)
  {
    printf(" Error allocating space for original matrix on node %d\n",my_ID);
    error = 1;
  }
  bail_out(error);

  B_p = (double *)prk_malloc(Colblock_size*sizeof(double));
  if (B_p == NULL)
  {
    printf(" Error allocating space for transpose matrix on node %d\n",my_ID);
    error = 1;
  }
  bail_out(error);

  /* Fill the original column matrix                                                */
  for (i=0;i<order; i++)
    for (j=0;j<Block_order;j++)
    {
      A_p[i*Block_order+j] = (double) (my_ID*Block_order + i*order +j);
      B_p[i*Block_order+j] = 0.0;
      T_p[i*Block_order+j] = 0.0;
    }

  for (iter = 0; iter<=iterations; iter++)
  {
    /* start timer after a warmup iteration                                        */
    if (iter == 1)
    {
      MPI_Barrier(MPI_COMM_WORLD);
      local_trans_time = wtime();
    }

    MPI_Alltoall(A_p, Block_order*Block_order, MPI_DOUBLE,
                 T_p, Block_order*Block_order, MPI_DOUBLE, MPI_COMM_WORLD);

    for (phase=0; phase<Num_procs; phase++)
    {
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

  local_trans_time = wtime() - local_trans_time;
  MPI_Reduce(&local_trans_time, &trans_time, 1, MPI_DOUBLE, MPI_MAX, root,
             MPI_COMM_WORLD);

  abserr = 0.0;
  double addit = ((double)(iterations+1) * (double) (iterations))/2.0;
  for (i=0;i<order; i++)
  {
    for (j=0;j<Block_order;j++)
    {
      const double temp = (order*(my_ID*Block_order+j)+i) * (iterations+1) + addit;
      abserr += ABS(B_p[i*Block_order+j] - temp);
    }
  }

  MPI_Reduce(&abserr, &abserr_tot, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

  if (my_ID == root)
  {
    if (abserr_tot < epsilon)
    {
      printf("Solution validates\n");
      avgtime = trans_time/(double)iterations;
      printf("Rate (MB/s): %lf Avg time (s): %lf\n",1.0E-06*bytes/avgtime, avgtime);
#if VERBOSE
      printf("Summed errors: %f \n", abserr);
#endif
    }
    else
    {
      printf("ERROR: Aggregate squared error %lf exceeds threshold %e\n", abserr, epsilon);
      error = 1;
    }
  }

  bail_out(error);

  MPI_Finalize();
  exit(EXIT_SUCCESS);

}  /* end of main */

