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

         transpose <#ranks per coherence domain><# iterations><matrix size> [tile size]

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
         
**********************************************************************************/

#include <par-res-kern_general.h>
#include <par-res-kern_mpi.h>

#define ARRAY(i,j) vector[i+1+(j)*(segment_size+1)]
#define NBR_INDEX(i,j) (i+(j)*(nbr_segment_size+1))

int main(int argc, char ** argv)
{
  int    my_ID;         /* Process ID (i.e. MPI rank)                            */
  int    root;
  int    m, n;          /* grid dimensions                                       */
  double local_pipeline_time, /* timing parameters                               */
         pipeline_time,
         avgtime;
  double epsilon = 1.e-8; /* error tolerance                                     */
  double corner_val;    /* verification value at top right corner of grid        */
  int    i, j, iter, ID;/* dummies                                               */
  int    iterations;    /* number of times to run the pipeline algorithm         */
  int    *start, *end;  /* starts and ends of grid slices                        */
  int    segment_size;
  int    error=0;       /* error flag                                            */
  int    Num_procs;     /* Number of processors                                  */
  double *vector;       /* array holding grid values                             */
  int    total_length;  /* total required length to store grid values            */
  MPI_Status status;    /* completion status of message                          */
  MPI_Win rmawin;       /* RMA window object */
  MPI_Group shm_group, origin_group, target_group;
  int origin_ranks[1], target_ranks[1];
  MPI_Aint nbr_segment_size;
  MPI_Win shm_win;       /* Shared Memory window object                          */
  MPI_Comm shm_comm_prep;/* Shared Memory prep Communicator                      */
  MPI_Comm shm_comm;     /* Shared Memory Communicator                           */
  int shm_procs;        /* # of processes in shared domain                       */
  int shm_ID;           /* MPI rank within coherence domain                      */
  int target_disp;
  double *target_ptr;

/*********************************************************************************
** Initialize the MPI environment
**********************************************************************************/
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_ID);
  MPI_Comm_size(MPI_COMM_WORLD, &Num_procs);

  root = 0;

/*********************************************************************
** process, test and broadcast input parameter
*********************************************************************/

  if (my_ID == root){
    if (argc != 4 && argc !=5){
      printf("Usage: %s  <#ranks per coherence domain> <# iterations> <matrix order> [tile size]\n", 
             *argv);
      error = 1;
      goto ENDOFTESTS;
    }

    group_size = atoi(*argv);
    if (group_size < 1) {
      printf("ERROR: # ranks per coherence domain must be >= 1 : %d \n",group_size);
      error = 1;
      goto ENDOFTESTS;
    } 
    if (Num_procs%group_size) {
      printf("ERROR: toal # ranks not divisible by ranks per coherence domain%d \n");
      error = 1;
      goto ENDOFTESTS;
    } 

    iterations = atoi(*++argv);
    if (iterations < 1){
      printf("ERROR: iterations must be >= 1 : %d \n",iterations);
      error = 1;
      goto ENDOFTESTS;
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

    if (argc == 5) Tile_order = atoi(*++argv);

    ENDOFTESTS:;
  }
  bail_out(error); 

  /*  Broadcast input data to all processes */
  MPI_Bcast(&order,      1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&iterations, 1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&Tile_order, 1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&group_size, 1, MPI_INT, root, MPI_COMM_WORLD);

  if (my_ID == root) {
    printf("MPI+SHM Matrix transpose: B = A^T\n");
    printf("Number of processes  = %d\n", Num_procs);
    printf("Rank group size      = %d\n", group_size);
    printf("Matrix order         = %d\n", order);
    if ((Tile_order > 0) && (Tile_order < order))
          printf("Tile size            = %d\n", Tile_order);
    else  printf("Untiled\n");
    //#ifndef SYNCHRONOUS
    printf("Non-");
    //#endif
    printf("Blocking messages\n");
    printf("Number of iterations = %d\n", iterations);
  }

  /* Setup for Shared memory regions */

  /* first divide WORLD in groups of size group_size */
  MPI_Comm_split(MPI_COMM_WORLD, my_ID/group_size, my_ID%group_size, &shm_comm_prep);
  /* derive from that a SHM communicator */
  MPI_Comm_split_type(shm_comm_prep, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);
  MPI_Comm_rank(shm_comm, &shm_ID);
  MPI_Comm_size(shm_comm, &shm_procs);
  /* do sanity check, making sure groups did not shrink in second comm split */
  if (shm_procs != group_size) MPI_Abort(MPI_COMM_WORLD, 666);

  /* a non-positive tile size means no tiling of the local transpose */
  tiling = (Tile_order > 0) && (Tile_order < order);
  bytes = 2 * sizeof(double) * order * order;

/*********************************************************************
** The matrix is broken up into column blocks that are mapped one to a 
** rank.  Each column block is made up of Num_procs smaller square 
** blocks of order block_order.
*********************************************************************/

  Num_groups     = Num_procs/group_size;
  Block_order    = order/Num_groups;

  group_ID       = ID/group_size;
  colstart       = Block_order * group_ID;
  Colblock_size  = order * Block_order;
  Block_size     = Block_order * Block_order;

/*********************************************************************
** Create the column block of the test matrix, the column block of the 
** transposed matrix, and workspace (workspace only if #procs>1)
*********************************************************************/

  /* only the root of each SHM domain specifies window of nonzero size */
  size_mul = (shm_ID==0);  
  MPI_Win_allocate_shared(Colblock_size*sizeof(double)*size_mul, sizeof(double)
                          MPI_INFO_NULL, shm_comm, (void *) &A_p, &shm_win);
  if (A_p == NULL){
    printf(" Error allocating space for original matrix on node %d\n",my_ID);
    error = 1;
  }
  bail_out(error);

  MPI_Win_allocate_shared(Colblock_size*sizeof(double)*size_mul, sizeof(double)
                          MPI_INFO_NULL, shm_comm, (void *) &B_p, &shm_win);
  if (B_p == NULL){
    printf(" Error allocating space for transposed matrix by group %d\n",group_ID);
    error = 1;
  }
  bail_out(error);

  if (Num_procs>1) {

    MPI_Win_allocate_shared(Block_size*sizeof(double)*size_mul, sizeof(double)
                            MPI_INFO_NULL, shm_comm, (void *) &Work_in_p, &shm_win);
    if (Work_in_p == NULL){
      printf(" Error allocating space for in block by group %d\n",group_ID);
      error = 1;
    }
    bail_out(error);

    MPI_Win_allocate_shared(Block_size*sizeof(double)*size_mul, sizeof(double)
                            MPI_INFO_NULL, shm_comm, (void *) &Work_out_p, &shm_win);
    if (Work_out_p == NULL){
      printf(" Error allocating space for out block by group %d\n",group_ID);
      error = 1;
    }
    bail_out(error);
  }
  

  /* clear the array                                                             */
  for (j=0; j<n; j++) for (i=start[my_ID]-1; i<=end[my_ID]; i++) {
    ARRAY(i-start[my_ID],j) = 0.0;
  }
  /* set boundary values (bottom and left side of grid                           */
  if (my_ID==0) for (j=0; j<n; j++) ARRAY(0,j) = (double) j;
  for (i=start[my_ID]-1; i<=end[my_ID]; i++) ARRAY(i-start[my_ID],0) = (double) i;

  /* redefine start and end for calling process to reflect local indices          */
  if (my_ID==0) start[my_ID] = 1; 
  else          start[my_ID] = 0;
  end[my_ID] = segment_size-1;

  /* Set up origin and target process groups for PSCW */
  MPI_Comm_group(shm_comm, &shm_group);
  /* Target group consists of rank my_ID+1, right neighbor */
  if (shm_ID < shm_procs-1) {
    target_ranks[0] = shm_ID+1;
    MPI_Group_incl(shm_group, 1, target_ranks, &target_group);
    MPI_Win_shared_query(shm_win, shm_ID+1, &nbr_segment_size, &target_disp, &target_ptr);
    nbr_segment_size = end[my_ID+1] - start[my_ID+1] + 1;
  } else {
    target_ranks[0] = MPI_PROC_NULL;
  }

  /* Origin group consists of rank my_ID-1, left neighbor */
  if (shm_ID > 0) {
    origin_ranks[0] = shm_ID-1;
    MPI_Group_incl(shm_group, 1, origin_ranks, &origin_group);
  } else {
    origin_ranks[0] = MPI_PROC_NULL;
  }

  for (iter=0; iter<=iterations; iter++) {

    /* start timer after a warmup iteration */
    if (iter == 1) { 
      MPI_Barrier(MPI_COMM_WORLD);
      local_pipeline_time = wtime();
    }

    /* execute pipeline algorithm for grid lines 1 through n-1 (skip bottom line) */
    for (j=1; j<n; j++) {

      /* if I am not at the left boundary, I need to wait for my left neighbor to
         send data                                                                */
      if (my_ID > 0) {
	if (shm_ID > 0) {
	  /*  Exposure epoch at target*/
	  MPI_Win_post(origin_group, 0, shm_win);
	  MPI_Win_wait(shm_win);
	} else {
	  MPI_Recv(&(ARRAY(start[my_ID]-1,j)), 1, MPI_DOUBLE, 
		   my_ID-1, j, MPI_COMM_WORLD, &status);
	}
      }

      for (i=start[my_ID]; i<= end[my_ID]; i++) {
        ARRAY(i,j) = ARRAY(i-1,j) + ARRAY(i,j-1) - ARRAY(i-1,j-1);
      }

      /* if I am not on the right boundary, send data to my right neighbor        */  
      if (my_ID != Num_procs-1) {
	if (shm_ID != shm_procs-1) {
	  /* Access epoch at origin */
	  MPI_Win_start(target_group, 0, shm_win);
	  target_ptr[NBR_INDEX(0,j)] = ARRAY(end[my_ID],j);
	  MPI_Win_complete(shm_win);	
	} else {
	  MPI_Send(&(ARRAY(end[my_ID],j)), 1, MPI_DOUBLE,
		   my_ID+1, j, MPI_COMM_WORLD);
	}
      }
    }

    /* copy top right corner value to bottom left corner to create dependency      */
    if (Num_procs >1) {
      if (my_ID==root) {
        corner_val = -ARRAY(end[my_ID],n-1);
        MPI_Send(&corner_val,1,MPI_DOUBLE,0,888,MPI_COMM_WORLD);
      }
      if (my_ID==0) {
        MPI_Recv(&(ARRAY(0,0)),1,MPI_DOUBLE,root,888,MPI_COMM_WORLD,&status);
      }
    }
    else ARRAY(0,0)= -ARRAY(end[my_ID],n-1);

  }

  local_pipeline_time = wtime() - local_pipeline_time;
  MPI_Reduce(&local_pipeline_time, &pipeline_time, 1, MPI_DOUBLE, MPI_MAX, root,
             MPI_COMM_WORLD);

  /*******************************************************************************
  ** Analyze and output results.
  ********************************************************************************/
 
  /* verify correctness, using top right value                                     */
  corner_val = (double) ((iterations+1)*(m+n-2));
  if (my_ID == root) {
    if (abs(ARRAY(end[my_ID],n-1)-corner_val)/corner_val >= epsilon) {
      printf("ERROR: checksum %lf does not match verification value %lf\n",
             ARRAY(end[my_ID],n-1), corner_val);
      error = 1;
    }
  }
  bail_out(error);

  if (my_ID == root) {
    avgtime = pipeline_time/iterations;
#ifdef VERBOSE   
    printf("Solution validates; verification value = %lf\n", corner_val);
    printf("Point-to-point synchronizations/s: %lf\n",
           ((float)((n-1)*(Num_procs-1)))/(avgtime));
#else
    printf("Solution validates\n");
#endif
    printf("Rate (MFlops/s): %lf Avg time (s): %lf\n",
           1.0E-06 * 2 * ((double)((m-1)*(n-1)))/avgtime, avgtime);
  }
 
  MPI_Win_free(&shm_win);

  MPI_Finalize();
  exit(EXIT_SUCCESS);

}  /* end of main */

