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

FUNCTIONS CALLED:

         Other than MPI or standard C functions, the following
         functions are used in this program:

         wtime()
         bail_out()

HISTORY: - Written by Rob Van der Wijngaart, March 2006.
         - modified by Rob Van der Wijngaart, August 2006:
            * changed boundary conditions and stencil computation to avoid
              overflow
            * introduced multiple iterations over grid and dependency between
              iterations

**********************************************************************************/

#include <par-res-kern_general.h>
#include <par-res-kern_mpi.h>

#define ARRAY(i,j,start,offset,width)     vector[i-start+offset+(j)*(width)]
#define NBR_ARRAY(i,j,start,offset,width) source_ptr[i-start+offset+(j)*(width)]

int main(int argc, char ** argv)
{
  int    my_ID;         /* rank                                                  */
  int    root=0, final; /* IDs of root rank and rank that verifies result        */
  long   m, n;          /* grid dimensions                                       */
  double local_pipeline_time, /* timing parameters                               */
         pipeline_time,
         avgtime;
  double epsilon =1.e-8;/* error tolerance                                       */
  double corner_val;    /* verification value at top right corner of grid        */
  int    i, j, iter, ID;/* dummies                                               */
  int    iterations;    /* number of times to run the pipeline algorithm         */
  int    *start, *end;  /* starts and ends of grid slices                        */
  MPI_Aint segment_size,/* net size of first dimension of 2D array of grid values
                           of target and neighbor ranks, exclusing ghost point   */
         nbr_segment_size;
  int    error=0;       /* error flag                                            */
  int    Num_procs;     /* Number of ranks                                       */
  double * RESTRICT vector;/* array holding grid values                          */
  long   total_length;  /* total required length to store grid values            */
  MPI_Win shm_win;      /* Shared Memory window object                           */
  MPI_Info rma_winfo;   /* info for window                                       */
  MPI_Comm shm_comm;    /* Shared Memory Communicator                            */
  int shm_procs;        /* # of ranks in shared domain                           */
  int shm_ID;           /* MPI rank                                              */
  int source_disp;      /* ignored                                               */
  double *source_ptr;   /* pointer to left neighbor's shared memory window       */
  int p2pbuf;           /* dummy buffer used for empty synchronization message   */
  long width, nbr_width;/* size of first dimension of 2D array of grid values of
                           target and neighbor ranks, including ghost point      */
  int  offset, nbr_offset;/* space reserved for ghost point, if present          */
  int  skip;            /* grid point to be skipped (only leftmost global rank)  */

/*********************************************************************************
** Initialize the MPI environment
**********************************************************************************/
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_ID);
  MPI_Comm_size(MPI_COMM_WORLD, &Num_procs);

  /* set final equal to highest rank, because it computes verification value       */
  final = Num_procs-1;

  /* Setup for Shared memory regions */
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);
  MPI_Comm_rank(shm_comm, &shm_ID);
  MPI_Comm_size(shm_comm, &shm_procs);

/*********************************************************************
** process, test and broadcast input parameter
*********************************************************************/

  if (my_ID == root){
    printf("Parallel Research Kernels version %s\n", PRKVERSION);
    printf("MPI+SHM pipeline execution on 2D grid\n");

    if (argc != 4){
      printf("Usage: %s  <#iterations> <1st array dimension> <2nd array dimension>\n",
             *argv);
      error = 1;
      goto ENDOFTESTS;
    }

    iterations = atoi(*++argv);
    if (iterations < 1){
      printf("ERROR: iterations must be >= 1 : %d \n",iterations);
      error = 1;
      goto ENDOFTESTS;
    }

    m = atol(*++argv);
    n = atol(*++argv);
    if (m < 1 || n < 1){
      printf("ERROR: grid dimensions must be positive: %ld, %ld \n", m, n);
      error = 1;
      goto ENDOFTESTS;
    }

    if (m<Num_procs) {
      printf("ERROR: First grid dimension %ld smaller than number of ranks %d\n",
             m, Num_procs);
      error = 1;
      goto ENDOFTESTS;
    }

    ENDOFTESTS:;
  }
  bail_out(error);

  if (my_ID == root) {
    printf("Number of ranks                = %d\n",Num_procs);
    printf("Grid sizes                     = %ld, %ld\n", m, n);
    printf("Number of iterations           = %d\n", iterations);
  }

  /* Broadcast benchmark data to all ranks */
  MPI_Bcast(&m, 1, MPI_LONG, root, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_LONG, root, MPI_COMM_WORLD);
  MPI_Bcast(&iterations, 1, MPI_INT, root, MPI_COMM_WORLD);

  start = (int *) prk_malloc(2*Num_procs*sizeof(int));
  if (!start) {
    printf("ERROR: Could not allocate space for array of slice boundaries\n");
    exit(EXIT_FAILURE);
  }
  end = start + Num_procs;
  start[0] = 0;
  for (ID=0; ID<Num_procs; ID++) {
    segment_size = m/Num_procs;
    if (ID < (m%Num_procs)) segment_size++;
    if (ID>0) start[ID] = end[ID-1]+1;
    end[ID] = start[ID]+segment_size-1;
  }
  /* now set segment_size to the value needed by the calling rank               */
  segment_size = end[my_ID] - start[my_ID] + 1;

  /* RMA win info */
  MPI_Info_create(&rma_winfo);
  /* This key indicates that passive target RMA will not be used.
   * It is the one info key that MPICH actually uses for optimization. */
  MPI_Info_set(rma_winfo, "no_locks", "true");

  /* leftmost rank in shared memory region leaves memory for a ghost point      */
  offset = (shm_ID == 0);

  /* leftmost global rank skips updating left boundary                          */
  skip = (my_ID==0);

  /* width takes into account an offset for a ghost point for the leftmost
     rank in a shared memory region                                             */
  width = segment_size+offset;
  /* storage takes the ghost points into account                                */
  total_length = width*n;

  MPI_Win_allocate_shared(total_length*sizeof(double), sizeof(double), 
                          rma_winfo, shm_comm, (void *) &vector, &shm_win);
  if (vector == NULL) {
    printf("Could not allocate space for grid slice of %ld by %ld points",
           segment_size, n);
    printf(" on rank %d\n", my_ID);
    error = 1;
  }
  bail_out(error);

  /* Get left neighbor base address */
  if (shm_ID > 0) {
    MPI_Win_shared_query(shm_win, shm_ID-1, &nbr_segment_size, &source_disp, &source_ptr);
    nbr_segment_size = end[my_ID-1] - start[my_ID-1] + 1;
    nbr_offset = (shm_ID==1);
    nbr_width = nbr_segment_size+nbr_offset;
  }

  /* clear the array                                                             */
  for (j=0; j<n; j++) for (i=start[my_ID]; i<=end[my_ID]; i++) {
      ARRAY(i,j,start[my_ID],offset,width) = 0.0;
  }
  /* set boundary values: left side of grid                                      */
  if (my_ID==0) for (j=0; j<n; j++) ARRAY(0,j,start[my_ID],offset,width) = (double) j;
  /* set boundary values: bottom side of grid, including ghost point values for
     "leftmost ranks inside coherence domains                                    */
  for (i=start[my_ID]-offset; i<=end[my_ID]; i++) 
    ARRAY(i,0,start[my_ID],offset,width) = (double) i;

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
	  MPI_Recv(&p2pbuf, 0, MPI_INT, shm_ID-1, 1, shm_comm, MPI_STATUS_IGNORE);
	} else {
	  MPI_Recv(&(ARRAY(start[my_ID]-1,j,start[my_ID],offset,width)), 1, MPI_DOUBLE,
		   my_ID-1, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
      }

      i = start[my_ID]+skip;
 
      if (shm_ID != 0) {
	ARRAY(i,j,start[my_ID],offset,width) = 
	  NBR_ARRAY(end[my_ID-1],j,start[my_ID-1],nbr_offset,nbr_width) + 
          ARRAY(i,j-1,start[my_ID],offset,width) - 
          NBR_ARRAY(end[my_ID-1],j-1,start[my_ID-1],nbr_offset,nbr_width);
	i++;
      }

      for (; i<= end[my_ID]; i++) {
	ARRAY(i,j,start[my_ID],offset,width) = ARRAY(i-1,j,  start[my_ID],offset,width) + 
                                               ARRAY(i,  j-1,start[my_ID],offset,width) - 
                                               ARRAY(i-1,j-1,start[my_ID],offset,width);
      }

      /* if I am not on the right boundary, send data to my right neighbor        */
      if (my_ID != Num_procs-1) {
	if (shm_ID != shm_procs-1) {
	  MPI_Send(&p2pbuf, 0, MPI_INT, shm_ID+1, 1, shm_comm);
	} else {
	  MPI_Send(&(ARRAY(end[my_ID],j,start[my_ID],offset,width)), 1, MPI_DOUBLE,
		   my_ID+1, j, MPI_COMM_WORLD);
	}
      }
    }

    /* copy top right corner value to bottom left corner to create dependency      */
    if (Num_procs >1) {
      if (my_ID==final) {
        corner_val = -ARRAY(end[my_ID],n-1,start[my_ID],offset,width);
        MPI_Send(&corner_val,1,MPI_DOUBLE,root,888,MPI_COMM_WORLD);
      }
      if (my_ID==root) {
        MPI_Recv(&(ARRAY(0,0,start[my_ID],offset,width)),1,MPI_DOUBLE,final,888,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      }
    }
    else ARRAY(0,0,start[my_ID],offset,width)= -ARRAY(end[my_ID],n-1,start[my_ID],offset,width);

  }

  local_pipeline_time = wtime() - local_pipeline_time;
  MPI_Reduce(&local_pipeline_time, &pipeline_time, 1, MPI_DOUBLE, MPI_MAX, final,
             MPI_COMM_WORLD);

  /*******************************************************************************
  ** Analyze and output results.
  ********************************************************************************/

  /* verify correctness, using top right value                                     */
  corner_val = (double) ((iterations+1)*(m+n-2));
  if (my_ID == final) {
    if (fabs(ARRAY(end[my_ID],n-1,start[my_ID],offset,width)-corner_val)/corner_val >= epsilon) {
      printf("ERROR: checksum %lf does not match verification value %lf\n",
             ARRAY(end[my_ID],n-1,start[my_ID],offset,width), corner_val);
      error = 1;
    }
  }
  bail_out(error);

  if (my_ID == final) {
    avgtime = pipeline_time/iterations;
#if VERBOSE
    printf("Solution validates; corner value = %lf, verification value = %lf\n", 
	   ARRAY(end[my_ID],n-1,start[my_ID],offset,width), corner_val);
    printf("Point-to-point synchronizations/s: %lf\n",
           ((float)((n-1)*(Num_procs-1)))/(avgtime));
#else
    printf("Solution validates\n");
#endif
    printf("Rate (MFlops/s): %lf Avg time (s): %lf\n",
           1.0E-06 * 2 * (double)((m-1)*(double)(n-1))/avgtime, avgtime);
  }

  MPI_Win_free(&shm_win);
  MPI_Finalize();
  exit(EXIT_SUCCESS);

}  /* end of main */
