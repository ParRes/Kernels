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

         transpose <matrix size> <# iterations> <comm. mode> [tile size]

         An optional parameter specifies the tile size used to divide the 
         individual matrix blocks for improved cache and TLB performance. 
  
         The output consists of diagnostics to make sure the 
         transpose worked and timing statistics.

FUNCTIONS CALLED:

         Other than MPI or standard C functions, the following 
         functions are used in this program:

          wtime()           Portable wall-timer interface.
          transpose()       Transpose a local matrix Block
          bail_out()        Determine global error and exit if nonzero.

HISTORY: Written by Tim Mattson, April 1999.  
         Updated by Rob Van der Wijngaart, December 2005.
         Updated by Rob Van der Wijngaart, October 2006.
         Updated by Rob Van der Wijngaart, November 2014::
         - made variable names more consistent 
         - introduced pointers to individual blocks inside matrix
           column blocks
         - put timing around entire iterative loop of transposes
  
*******************************************************************/

/******************************************************************
                     Layout nomenclature                         
                     -------------------

o Each rank owns one block of columns (Colblock) of the overall
  matrix to be transposed, as well as of the transposed matrix.
o Colblock is stored contiguously in the memory of the rank
o Colblock is composed of #ranks Blocks. The Block is the unit
  of data that gets communicated between ranks. Block i of rank
  j is locally tranposed and then sent to rank i, where it is
  stored in Block j of the transposed matrix.
o When tiling is applied to reduce TLB misses, each block gets 
  accessed by tiles. 
o When a Block is prepared for communication by carrying out the
  local transpose, the result of the local transpose is stored
  in a buffer called Work on the sending side. At the receiving
  side it is stored in the right location without going through
  an intermediate buffer
o The original and transposed matrix data structures are 
  distinguishde by the prefixes Orig_ and Trans_

 -----------------------------------------------------------------
|Name:      |           |           |                             |
| Colblock  |           |           |                             |
|Start addr:|           |           |                             |
| Colblock_p|           |           |                             |
|           |           |           |                             |
|        -------------------------------                          |
|           |Name:      |           |                             |
|           |  Block    |           |                             |
|           |Start addr:|           |                             |
|           | Block_p   |           |                             |
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

/* Constant to shift column index */
#define  COL_SHIFT  1000.00
/* Constant to shift row index */
#define  ROW_SHIFT  0.001  
#define A(i,j) A[(i)+Block_cols*(j)]
#define B(i,j) B[(i)+Block_cols*(j)]
#define Orig_Colblock(i,j)  Orig_Colblock_p[(i)*Block_order+j]
#define Trans_Colblock(i,j) Trans_Colblock_p[(i)*Block_order+j]

void transpose(double *A, double *B, int tile_size, int sub_rows, int sub_cols);

int main(int argc, char ** argv)
{
  int Block_order;         /* order of Colblock and Block           */
  int Block_size;          /* size of a single block                */
  int Colblock_size;       /* size of local column block            */
  int Tile_order=32;       /* default Tile order                    */
  int Tile_size;           /* Tile size                             */
  int Num_procs;           /* number of ranks                       */
  int order;               /* order of overall matrix               */
  int send_to, recv_from;  /* ranks with which to communicate       */
  MPI_Status status;       
#ifndef SYNCHRONOUS
  MPI_Request send_req;
  MPI_Request recv_req;
#endif
  long bytes;              /* combined size of matrices             */
  int my_ID;               /* Process ID (i.e. MPI rank)            */
  int root=0;              /* rank of root process                  */
  int iterations;          /* number of times to do the transpose   */
  int i, j, iter, phase;   /* dummies                               */
  int error;               /* error flag                            */
  double *Orig_Colblock_p; /* original local matrix column block    */
  double *Trans_Colblock_p;/* local transposed matrix column block  */
  double **Orig_Block_p;   /* buffer holding local matrix block     */
  double **Trans_Block_p;  /* buffer to hold transposed data        */
  double *Work_p;          /* workspace for the transpose function  */
  double errsq,            /* squared error                         */
         errsq_tot,        /* aggregate squared error               */
         diff;             /* pointwise error                       */
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
  if (my_ID == root) {
    if (argc != 3 && argc != 4){
      printf("Usage: %s <# iterations> <matrix order> [Tile size]\n",
                                                               *argv);
      error = 1; goto ENDOFTESTS;
    }

    iterations  = atoi(*++argv);
    if(iterations < 1){
      printf("ERROR: iterations must be >= 1 : %d \n",iterations);
      error = 1; goto ENDOFTESTS;
    }

    order = atoi(*++argv);
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

    if (argc == 4) Tile_order = atoi(*++argv);
    /* a non-positive tile size means no tiling of the local transpose */
    if (Tile_order <=0 || Tile_order >= order/Num_procs) {
        Tile_order = order/Num_procs;
     }
    ENDOFTESTS:;
  }
  bail_out(error);

  if (my_ID == root) {
    printf("MPI Matrix transpose: B = A^T\n");
    printf("Number of processes  = %d\n", Num_procs);
    printf("Matrix order         = %d\n", order);
    if (Tile_order < order/Num_procs) 
          printf("Tile size            = %d\n", Tile_order);
    else  printf("Untiled\n");
#ifndef SYNCHRONOUS
    printf("Non-");
#endif
    printf("Blocking messages\n");
    printf("Number of iterations = %d\n", iterations);
  }

  /*  Broadcast input data to all processes */
  MPI_Bcast (&order,      1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast (&iterations, 1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast (&Tile_order, 1, MPI_INT, root, MPI_COMM_WORLD);

  bytes = 2 * sizeof(double) * order * order;

/*********************************************************************
** The matrix is broken up into column blocks that are mapped one to a 
** rank.  Each column block is made up of Num_procs smaller square 
** blocks of order block_order.
*********************************************************************/

  Block_order    = order/Num_procs;
  int jstart     = Block_order * my_ID;
  int jend       = jstart + Block_order -1;
  Colblock_size  = order * Block_order;
  Block_size     = Block_order * Block_order;

/*********************************************************************
** Create the column block of the test matrix, the row block of the 
** transposed matrix, and workspace (workspace only if #procs>1)
*********************************************************************/
  Orig_Colblock_p = (double *)malloc(Colblock_size*sizeof(double));
  if (Orig_Colblock_p == NULL){
    printf(" Error allocating space for original matrix on node %d\n",my_ID);
    error = 1;
  }
  bail_out(error);

  Trans_Colblock_p = (double *)malloc(Colblock_size*sizeof(double));
  if (Trans_Colblock_p == NULL){
    printf(" Error allocating space for transpose matrix on node %d\n",my_ID);
    error = 1;
  }
  bail_out(error);

  if (Num_procs>1) {
    Work_p   = (double *)malloc(Block_size*sizeof(double));
    if (Work_p == NULL){
      printf(" Error allocating space for work on node %d\n",my_ID);
      error = 1;
    }
    bail_out(error);
  }
  
  /* Fill the original column matrix in Orig_Colblock.                    */
  for (i=0;i<order; i++) for (j=0;j<Block_order;j++) {
    Orig_Colblock(i,j) = COL_SHIFT*(my_ID*Block_order+j) + ROW_SHIFT*i;
  }

  /*  Set the transpose matrix to a known garbage value.            */
  for (i=0;i<Colblock_size; i++) Trans_Colblock_p[i] = -1.0;

/*********************************************************************
 ** create entry points to the Blocks within Colblocks
 ********************************************************************/

  Orig_Block_p  = (double **) malloc(2*sizeof(double *)*Num_procs);
  if (Orig_Block_p == NULL) {
    printf(" Error allocating space for block pointers on node %d\n", my_ID);
    error = 1;
  }
  bail_out(error);
  Trans_Block_p = Orig_Block_p + Num_procs;
  for (i=0; i<Num_procs; i++) {
    Orig_Block_p[i]  = Orig_Colblock_p  + i*Block_size;
    Trans_Block_p[i] = Trans_Colblock_p + i*Block_size;
  }

  errsq = 0.0;
  for (iter = 0; iter<=iterations; iter++){

    /* start timer after a warmup iterations                                        */
    if (iter == 1) { 
      MPI_Barrier(MPI_COMM_WORLD);
      local_trans_time = wtime();
    }

    /* do the local transpose                                                       */
    transpose(Orig_Block_p[my_ID], Trans_Block_p[my_ID], Tile_order, 
              Block_order, Block_order);

    for (phase=1; phase<Num_procs; phase++){
      recv_from = (my_ID + phase            )%Num_procs;
      send_to   = (my_ID - phase + Num_procs)%Num_procs;

#ifndef SYNCHRONOUS
      MPI_Irecv(Trans_Block_p[recv_from], Block_size, MPI_DOUBLE, 
                recv_from, phase, MPI_COMM_WORLD, &recv_req);  
#endif

      transpose(Orig_Block_p[send_to], Work_p, Tile_order, 
                Block_order, Block_order);
	 
#ifndef SYNCHRONOUS  
      MPI_Isend(Work_p, Block_size, MPI_DOUBLE, send_to,
                phase, MPI_COMM_WORLD, &send_req);
      MPI_Wait(&recv_req, &status);
      MPI_Wait(&send_req, &status);
#else
      MPI_Sendrecv(Work_p, Block_size, MPI_DOUBLE, send_to, phase,
                   Trans_Block_p[recv_from], Block_size, MPI_DOUBLE, 
	           recv_from, phase, MPI_COMM_WORLD, &status);
#endif

    }  /* end of phase loop  */
  } /* end of iterations */

  local_trans_time = wtime() - local_trans_time;
  MPI_Reduce(&local_trans_time, &trans_time, 1, MPI_DOUBLE, MPI_MAX, root,
             MPI_COMM_WORLD);

  for (i=0;i<order; i++) {
    for (j=jstart;j<=jend;j++) {
      diff = Trans_Colblock(i,j-jstart) - (COL_SHIFT*i + ROW_SHIFT*j);
      errsq += diff*diff;
    }
  }

  MPI_Reduce(&errsq, &errsq_tot, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

  if (my_ID == root) {
    if (errsq_tot < epsilon) {
      printf("Solution validates\n");
      avgtime = trans_time/(double)iterations;
      printf("Rate (MB/s): %lf Avg time (s): %lf\n",1.0E-06*bytes/avgtime, avgtime);
#ifdef VERBOSE
      printf("Squared errors: %f \n", errsq);
#endif
    }
    else {
      printf("ERROR: Aggregate squared error %lf exceeds threshold %e\n", errsq, epsilon);
      error = 1;
    }
  }

  bail_out(error);

  MPI_Finalize();
  exit(EXIT_SUCCESS);

}  /* end of main */

/*******************************************************************
** Transpose a local matrix Block and store the result in another
** Block. Apply tiling as needed
**
**  Parameters:
**
**    A, B                   - base address of the Blocks
**    Block_rows, Block_cols - Number of rows/cols in the Block
**    Tile_order             - Number of rows/cols in matrix Tile
** 
*******************************************************************/

void transpose(
  double *A, double *B,           /* input and output Blocks      */
  int Tile_order,                 /* tile order                   */
  int Block_rows, int Block_cols) /* size of tile to transpose    */
{
  int    i, j, it, jt;

  /* tile only if the tile size is smaller than the matrix block  */
  if (Tile_order < Block_cols) {
    for (i=0; i<Block_cols; i+=Tile_order) 
      for (j=0; j<Block_rows; j+=Tile_order) 
        for (it=i; it<MIN(Block_cols,i+Tile_order); it++)
          for (jt=j; jt<MIN(Block_rows,j+Tile_order);jt++)
            B(it,jt) = A(jt,it); 
  }
  else {
    for (i=0;i<Block_cols; i++) 
      for (j=0;j<Block_rows;j++)
        B(i,j) = A(j,i);
  }
}
