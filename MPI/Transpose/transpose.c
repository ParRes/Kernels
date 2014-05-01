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

          wtime()           portable wall-timer interface.
          trans_comm()   Do the transpose with MPI_Sendrecv()
          bail_out()        Determines global error and exits if nonzero.

HISTORY: Written by Tim Mattson, April 1999.  
         Updated by Rob Van der Wijngaart, December 2005.
         Updated by Rob Van der Wijngaart, October 2006.
  
*******************************************************************/

#include <par-res-kern_general.h>
#include <par-res-kern_mpi.h>

/* Constant to shift column index */
#define  COL_SHIFT  1000.00
/* Constant to shift row index */
#define  ROW_SHIFT  0.001  

void trans_comm(double *buff,  double *trans, int Block_order,
                int tile_size, double *work,  int my_ID, int Num_procs);
void transpose(double *A, double *B, int tile_size, int sub_rows, int sub_cols);


int main(int argc, char ** argv)
{
  int Block_order;      /* order of a local matrix block */
  int Num_procs;        /* Number of processors */
  int order;            /* Order of overall matrix */
  int Col_block_size;   /* size of local col blck */
  double bytes;         /* combined size of matrices                       */
  int block_size;       /* size of a single block */
  int tile_size=32;     /* default tile size for tiling of local transpose */
  int my_ID;            /* Process ID (i.e. MPI rank) */
  int root=0;           /* rank of root process */
  int iterations;       /* number of times to do the transpose */
  int i, j, iter;
  int error;            /* error flag */

  double *buff;           /* buffer to hold local matrix block */
  double *trans;          /* buffer to hold transposed data */
  double *work;           /* workspace for the tranpose funtion */

  double errsq,         /* squared error */
         errsq_tot,     /* aggregate squared error */
         diff;          /* pointwise error */
  double epsilon = 1.e-8; /* error tolerance */
  double col_val;
  double trans_time,    /* timing parameters                               */
         avgtime = 0.0, 
         maxtime = 0.0, 
         mintime = 366.0*24.0*3600.0; /* set the minimum time to a large 
                             value; one leap year should be enough         */
   
  double test_results (int , int , int , double*);
  void   trans_isnd_ircv (double*, double*, int, int, double*, int, int);
  void   trans_sendrcv   (double*, double*, int, int, double*, int, int);
  void   fill_mat(int , int , int , double*);
  void   fill_garbage(int , int , int , double*);
  void   output_timings (char *, int , int , double , double, double, 
                         int, int , int);
  void   analyze_results (int, double, int, double *, double*, double*, 
                         int*, int*, int*);

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
      error = 1;
      goto ENDOFTESTS;
    }

    iterations  = atoi(*++argv);
    if(iterations < 1){
      printf("ERROR: iterations must be >= 1 : %d \n",iterations);
      error = 1;
      goto ENDOFTESTS;
    }

    order = atoi(*++argv);
    if (order < Num_procs) {
      printf("ERROR: matrix order %d should at least # procs %d\n", 
             order, Num_procs);
      error = 1;
      goto ENDOFTESTS;
    }
    if (order%Num_procs) {
      printf("ERROR: matrix order %d should be divisible by # procs %d\n",
             order, Num_procs);
      error = 1;
      goto ENDOFTESTS;
    }

    if (argc == 4) tile_size = atoi(*++argv);
    /* a non-positive tile size means no tiling of the local transpose */
    if (tile_size <=0 || tile_size >= order/Num_procs) {
        tile_size = order/Num_procs;
     }
    
    ENDOFTESTS:;
  }
  bail_out(error);

  if (my_ID == root) {
    printf("MPI Matrix transpose: B = A^T\n");
    printf("Number of processes  = %d\n", Num_procs);
    printf("Matrix order         = %d\n", order);
    if (tile_size < order/Num_procs) 
          printf("Tile size            = %d\n", tile_size);
    else  printf("Untiled\n");
#ifndef SYNCHRONOUS
    printf("Non-");
#endif
    printf("Blocking messages\n");
    printf("Number of iterations = %d\n", iterations);
  }

  /*  Broadcast benchmark data to all processes */
  MPI_Bcast (&order,      1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast (&iterations, 1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast (&tile_size,  1, MPI_INT, root, MPI_COMM_WORLD);

  bytes = 2.0 * sizeof(double) * order * order;

/*********************************************************************
** Key Matrix parameters: The matrix is broken up into column blocks
** that are mapped one to a node.  Each column block is made of 
** Num_procs smaller blocks of order block_order.
**
**   order   - order of the global matrix
**   Col_block_size - numb of elements in a local column block.
**   block_size     - numb of elements in a working block.
*********************************************************************/
  Block_order    = order/Num_procs;
  Col_block_size = order * Block_order;
  block_size     = Block_order  * Block_order;

/*********************************************************************
** Create the column block of the test matrix, the row block of the 
** transposed matrix, and workspace (workspace only if #procs>1)
*********************************************************************/
  buff   = (double *)malloc(2*Col_block_size*sizeof(double));
  if (buff == NULL){
    printf(" Error allocating space for buff on node %d\n",my_ID);
    error = 1;
  }
  bail_out(error);

  trans  = buff + Col_block_size;

  if (Num_procs>1) {
    work   = (double *)malloc(block_size*sizeof(double));
    if (work == NULL){
      printf(" Error allocating space for work on node %d\n",my_ID);
      error = 1;
    }
    bail_out(error);
  }
  
  /* Fill the original column matrix in buff.                       */
  for (i=0;i<order; i++) for (j=0;j<Block_order;j++) {
    col_val = COL_SHIFT * (my_ID*Block_order + j);
    buff[i*Block_order+j] = col_val + ROW_SHIFT * i;
  }

  /*  Set the transpose matrix to a known garbage value.            */
  for (i=0;i<Col_block_size; i++) trans[i] = -1.0;

  errsq = 0.0;
  for (iter = 0; iter<iterations; iter++){

    MPI_Barrier(MPI_COMM_WORLD);
    trans_time = wtime();

    trans_comm(buff, trans, Block_order, tile_size,
               work, my_ID, Num_procs);

    trans_time = wtime() - trans_time;

#ifdef VERBOSE
    printf("\n Node %d has finished with transpose \n",my_ID);
#endif
    if (iter>0 || iterations==1) { /* skip the first iteration */
      avgtime = avgtime + trans_time;
      mintime = MIN(mintime, trans_time);
      maxtime = MAX(maxtime, trans_time);
    }
    
    for (i=0;i<order; i++) {
      col_val = COL_SHIFT * i; 	
      for (j=0;j<Block_order;j++) {
        diff = trans[i*Block_order+j] -
               (col_val  + ROW_SHIFT * (my_ID*Block_order + j));
        errsq += diff*diff;
      }
    }

  }  /* end of iter loop  */

  MPI_Reduce(&errsq, &errsq_tot, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

  if (my_ID == root) {
    if (errsq_tot < epsilon) {
      printf("Solution validates\n");
      avgtime = avgtime/(double)(MAX(iterations-1,1));
      printf("Rate (MB/s): %lf, Avg time (s): %lf, Min time (s): %lf",
             1.0E-06 * bytes/mintime, avgtime, mintime);
      printf(", Max time (s): %lf\n", maxtime);
#ifdef VERBOSE
      printf("Squared errors: %f \n", errsq);
#endif
    }
    else {
      printf("ERROR: Aggregate squared error %lf exceeds threshold %e\n",
             errsq, epsilon);
      error = 1;
    }
  }

  bail_out(error);

  MPI_Finalize();
  exit(EXIT_SUCCESS);

}  /* end of main */


/*******************************************************************

NAME:    trans_comm

PURPOSE: This function uses MPI SND's and RCV's to transpose 
         of a column-bock distributed matrix in either synchronous
         or asynchronous mode
  
/*******************************************************************
** Define macros to compute processor source and destinations
*******************************************************************/
#define TO(ID,   PHASE, NPROC)  ((ID + PHASE        ) % NPROC)
#define FROM(ID, PHASE, NPROC)  ((ID + NPROC - PHASE) % NPROC)

void trans_comm(double *buff,  double *trans, int Block_order,
                int tile_size, double *work,  int my_ID, int Num_procs)
{
  int iphase;
  int block_size;
  int send_to, recv_from;
  double *bblock;    /* pointer to current location in buff */
  double *tblock;    /* pointer to current location in trans */
  MPI_Status status;
#ifndef SYNCHRONOUS
  MPI_Request send_req, recv_req;
#endif
 
  block_size = Block_order * Block_order;

/*******************************************************************
**  Do the tranpose in Num_procs phases.  
**
**  In the first phase, do the diagonal block.  Then move out 
**  from the diagonal copying the local matrix into a communication 
**  buffer (while doing the local transpose) and send to processor 
**  (diag+phase)%Num_procs.
*******************************************************************/
  bblock = buff  + my_ID*block_size;
  tblock = trans + my_ID*block_size;

  transpose(bblock, tblock, tile_size, Block_order, Block_order);

  for (iphase=1; iphase<Num_procs; iphase++){

    recv_from = FROM(my_ID, iphase, Num_procs);
    tblock  = trans + recv_from * block_size;
    send_to = TO(my_ID, iphase, Num_procs);
    bblock  = buff + send_to * block_size;

#ifndef SYNCHRONOUS
    MPI_Irecv (tblock, block_size, MPI_DOUBLE, recv_from,
              iphase, MPI_COMM_WORLD, &recv_req);  
#endif

    transpose(bblock, work, tile_size, Block_order, Block_order);
	 
#ifdef VERBOSE
    printf("\n transposed block for phase %d on node %d is \n",
               iphase,my_ID);
#endif

#ifndef SYNCHRONOUS  
    MPI_Isend(work, block_size, MPI_DOUBLE, send_to,
              iphase, MPI_COMM_WORLD, &send_req);
    MPI_Wait(&recv_req,  &status);
    MPI_Wait(&send_req, &status);
#else
    MPI_Sendrecv(work,   block_size, MPI_DOUBLE, send_to,   iphase,
		 tblock, block_size, MPI_DOUBLE, recv_from, iphase, 
		 MPI_COMM_WORLD, &status);
#endif

#ifdef VERBOSE
   printf("\n phase %d on node %d, recv_from = %d and send_to = %d\n",
                             iphase, my_ID, recv_from, send_to);
#endif


   }
}

/*******************************************************************
** Copy the transpose of an array slice inside matrix A into 
** the an array slice inside matrix B.
**
**  Parameters:
**
**       A, B                base address of the slices
**       sub_rows, sub_cols  Numb of rows/cols in the slice.
** 
*******************************************************************/

void transpose(
  double *A, double *B,         /* input and output matrix        */
  int tile_size,                /* local tile size                */
  int sub_rows, int sub_cols)   /* size of slice to  transpose    */
{
  int    i, j, it, jt;

  /*  Transpose the  matrix.  */

  /* tile only if the tile size is smaller than the matrix block  */
  if (tile_size < sub_cols) {
    for (i=0; i<sub_cols; i+=tile_size) { 
      for (j=0; j<sub_rows; j+=tile_size) { 
        for (it=i; it<MIN(sub_cols,i+tile_size); it++){ 
          for (jt=j; jt<MIN(sub_rows,j+tile_size);jt++){ 
            B[it+sub_cols*jt] = A[jt+sub_rows*it]; 
          } 
        } 
      } 
    } 
  }
  else {
    for (i=0;i<sub_cols; i++) {
      for (j=0;j<sub_rows;j++) {
        B[i+sub_cols*j] = A[j+i*sub_rows];
      }
    }	
  }
}
