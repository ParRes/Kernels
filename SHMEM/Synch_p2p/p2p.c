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

         Other than SHMEM or standard C functions, the following 
         functions are used in this program:

         wtime()
         bail_out()

HISTORY: - Written by Rob Van der Wijngaart, March 2006.
         - modified by Rob Van der Wijngaart, August 2006:
            * changed boundary conditions and stencil computation to avoid 
              overflow
            * introduced multiple iterations over grid and dependency between
              iterations
         - modified by Gabriele Jost, March 2015:
            * adapted for SHMEM
  
**********************************************************************************/

#include <par-res-kern_general.h>
#include <par-res-kern_shmem.h>

#define ARRAY(i,j) vector[i+1+(j)*(segment_size+1)]

int main(int argc, char ** argv)
{
  int    my_ID;           /* SHMEM thread ID                                     */
  int    root;            /* ID of master rank                                   */
  long   m, n;            /* grid dimensions                                     */
  double *pipeline_time,   /* timing parameters                                  */
         *local_pipeline_time, avgtime;
  double epsilon = 1.e-8; /* error tolerance                                     */
  double corner_val;      /* verification value at top right corner of grid      */
  int    i, j, iter, ID;  /* dummies                                             */
  int    iterations;      /* number of times to run the pipeline algorithm       */
  int    *start, *end;    /* starts and ends of grid slices                      */
  long   segment_size;    /* x-dimension of grid slice owned by calling rank     */
  int    error=0;         /* error flag                                          */
  int    Num_procs;       /* Number of ranks                                     */
  double * RESTRICT vector;/* array holding grid values                          */
  long   total_length;    /* total required length to store grid values          */
  int    *flag_left;      /* synchronization flags                               */
#if SYNCHRONOUS
  int    *flag_right;     /* synchronization flags                               */
#endif
  double *dst;            /* target address of communication                     */
  double *src;            /* source address of communication                     */
  long   *pSync;          /* work space for SHMEM collectives                    */
  double *pWrk;           /* work space for SHMEM collectives                    */
  
/*********************************************************************************
** Initialize the SHMEM environment
**********************************************************************************/
  prk_shmem_init();
  my_ID =  prk_shmem_my_pe();
  Num_procs =  prk_shmem_n_pes();
/* we set root equal to the highest rank, because this is also the rank that 
   reports on the verification value                                            */
  root = Num_procs-1;

/*********************************************************************
** process, test and broadcast input parameter
*********************************************************************/

  if (my_ID == root) {
    printf("Parallel Research Kernels version %s\n", PRKVERSION);
    printf("SHMEM pipeline execution on 2D grid\n");
  }

  if (argc != 4){
    if (my_ID == root)
      printf("Usage: %s  <#iterations> <1st array dimension> <2nd array dimension>\n", 
           *argv);
    error = 1;
    goto ENDOFTESTS;
  }

  iterations = atoi(*++argv);
  if (iterations < 1){
    if (my_ID==root)
      printf("ERROR: iterations must be >= 1 : %d \n",iterations);
    error = 1;
    goto ENDOFTESTS;
  } 

  m = atol(*++argv);
  n = atol(*++argv);
  if (m < 1 || n < 1){
    if (my_ID == root)
      printf("ERROR: grid dimensions must be positive: %ld, %ld \n", m, n);
    error = 1;
    goto ENDOFTESTS;
  }

// initialize sync variables for error checks
  pSync = (long *)   prk_shmem_align(prk_get_alignment(), sizeof(long) * PRK_SHMEM_REDUCE_SYNC_SIZE );
  pWrk  = (double *) prk_shmem_align(prk_get_alignment(), sizeof(double) * PRK_SHMEM_REDUCE_MIN_WRKDATA_SIZE );
  if (!pSync || !pWrk) {
    printf("Rank %d could not allocate work space for collectives\n", my_ID);
    error = 1;
    goto ENDOFTESTS;
  }
  for (i = 0; i < PRK_SHMEM_BCAST_SYNC_SIZE; i += 1) {
    pSync[i] = PRK_SHMEM_SYNC_VALUE;
  }

  if (m<=Num_procs) {
    if (my_ID == root)
      printf("ERROR: First grid dimension %ld must be > #ranks %d\n", m, Num_procs);
    error = 1;
  }
  ENDOFTESTS:;
  bail_out (error);
  shmem_barrier_all ();

  if (my_ID == root) {
    printf("Number of ranks            = %d\n",Num_procs);
    printf("Grid sizes                 = %ld, %ld\n", m, n);
    printf("Number of iterations       = %d\n", iterations);
#if SYNCHRONOUS
    printf("Handshake between neighbor threads\n");
#else
    printf("No handshake between neighbor threads\n");
#endif
  }

  flag_left = (int *) prk_shmem_align(prk_get_alignment(),sizeof(int) * n);
#if SYNCHRONOUS
  flag_right = (int *) prk_shmem_align(prk_get_alignment(),sizeof(int) * n);
  int recv_val [1];
  recv_val [0] = -1;
#endif
  dst = (double *) prk_shmem_align(prk_get_alignment(),sizeof(double) * (n));
  src = (double *) prk_malloc (sizeof(double) * (n));
  local_pipeline_time = (double *) prk_shmem_align(prk_get_alignment(),sizeof(double));
  pipeline_time = (double *) prk_shmem_align(prk_get_alignment(),sizeof(double));
  if (!flag_left || !dst || !src) {
    printf("ERROR: could not allocate flags or communication buffers on rank %d\n", 
           my_ID);
    error = 1;
  }
  bail_out(error); 

  start = (int *) prk_shmem_align(prk_get_alignment(),2*Num_procs*sizeof(int));
  if (!start) {
    printf("ERROR: Could not allocate space for array of slice boundaries on rank %d\n",
           my_ID);
    error = 1;
  }
  bail_out(error);
  end = start + Num_procs;
  start[0] = 0;
  for (ID=0; ID<Num_procs; ID++) {
    segment_size = m/Num_procs;
    if (ID < (m%Num_procs)) segment_size++;
    if (ID>0) start[ID] = end[ID-1]+1;
    end[ID] = start[ID]+segment_size-1;
  }

  /* now set segment_size to the value needed by the calling rank                */
  segment_size = end[my_ID] - start[my_ID] + 1;

  /* total_length takes into account one ghost cell on left side of segment      */
  total_length = ((end[my_ID]-start[my_ID]+1)+1)*n;
  vector = (double *) prk_malloc(total_length*sizeof(double));
  if (vector == NULL) {
    printf("Could not allocate space for grid slice of %ld by %ld points", 
           segment_size, n);
    printf(" on rank %d\n", my_ID);
    error = 1;
  }
  bail_out(error);

  /* clear the array                                                             */
  for (j=0; j<n; j++) for (i=start[my_ID]-1; i<=end[my_ID]; i++) {
    ARRAY(i-start[my_ID],j) = 0.0;
  }
  /* set boundary values (bottom and left side of grid                           */
  if (my_ID==0) for (j=0; j<n; j++) ARRAY(0,j) = (double) j;
  for (i=start[my_ID]-1; i<=end[my_ID]; i++) ARRAY(i-start[my_ID],0) = (double) i;

  /* redefine start and end for calling rank to reflect local indices            */
  if (my_ID==0) start[my_ID] = 1; 
  else          start[my_ID] = 0;
  end[my_ID] = segment_size-1;

  /* initialize synchronization flags                                            */
  /* set flags to zero to indicate no data is available yet                      */
  int true = 1; int false = !true;
  for (j=0; j<n; j++) {
    flag_left[j] = false;
#if SYNCHRONOUS
    flag_right[j] = false;
#endif
  }  

  shmem_barrier_all ();

  for (iter=0; iter<=iterations; iter++) {

#if !SYNCHRONOUS
    /* true and false toggle each iteration                                      */
    true = (iter+1)%2; false = !true;
#endif

    if (iter == 1) {
      shmem_barrier_all ();
      local_pipeline_time[0] = wtime();
    }

    if (my_ID==0 && Num_procs>1) { 
      /* first thread waits for corner value to be copied                        */
      shmem_int_wait_until(&flag_left[0], SHMEM_CMP_EQ, false);
      if (iter>0) {
        ARRAY(start[my_ID]-1,0) = dst[0];
      }
#if SYNCHRONOUS
      flag_left[0]= true;
      shmem_int_p(&flag_right[0], true, root);
      shmem_fence();
#endif      
    }

    for (j=1; j<n; j++) {

      /* if I am not at the left boundary, wait for left neighbor to send data   */
      if (my_ID > 0) {
        shmem_int_wait_until(&flag_left[j], SHMEM_CMP_EQ, true);
#if SYNCHRONOUS
        flag_left[j]= false;
        // tell the left neighbor I got the data
        shmem_int_p(&flag_right[j], false, my_ID-1);
#endif      
        ARRAY(start[my_ID]-1,j) = dst[j];
      }

      for (i=start[my_ID]; i<= end[my_ID]; i++) {
        ARRAY(i,j) = ARRAY(i-1,j) + ARRAY(i,j-1) - ARRAY(i-1,j-1);
      }

      if (my_ID != Num_procs-1) {
#if SYNCHRONOUS 
        shmem_int_wait_until(&flag_right[j], SHMEM_CMP_EQ, false);
        flag_right[j] = true;
#endif 
        src[j] = ARRAY (end[my_ID],j);
        shmem_double_p(&dst[j], src[j], my_ID+1);
        shmem_fence();
     /* indicate to right neighbor that data is available  */
        shmem_int_p(&flag_left[j], true, my_ID+1);
      }  
    }

    /* copy top right corner value to bottom left corner to create dependency      */
    if (Num_procs >1) {
      if (my_ID==root) {
        corner_val = -ARRAY(end[my_ID],n-1);
        src [0] = corner_val;
        shmem_double_p(&dst[0], src[0], 0);
        shmem_fence();
        /* indicate to PE 0 that data is available  */
#if SYNCHRONOUS
        shmem_int_wait_until(&flag_right[0], SHMEM_CMP_EQ, true);
        flag_right[j] = false;
        shmem_int_p(&flag_left[0], false, 0);
#else
        shmem_int_p(&flag_left[0], true, 0);
#endif
      }
    }
    else ARRAY(0,0)= -ARRAY(end[my_ID],n-1);
  }

  local_pipeline_time [0] = wtime() - local_pipeline_time [0];
  shmem_double_max_to_all(pipeline_time, local_pipeline_time, 1, 0, 0, Num_procs, 
                          pWrk, pSync);

  /* verify correctness, using top right value                                     */
  corner_val = (double) ((iterations+1)*(m+n-2));
  if (my_ID == root) {
    if (fabs(ARRAY(end[my_ID],n-1)-corner_val)/corner_val >= epsilon) {
      printf("ERROR: checksum %lf does not match verification value %lf\n",
             ARRAY(end[my_ID],n-1), corner_val);
      error = 1;
    }
  }
  bail_out(error);

  if (my_ID == root) {
    avgtime = pipeline_time [0]/iterations;
#if VERBOSE   
    printf("Solution validates; verification value = %lf\n", corner_val);
    printf("Point-to-point synchronizations/s: %lf\n",
           ((float)((n-1)*(Num_procs-1)))/(avgtime));
#else
    printf("Solution validates\n");
#endif
    printf("Rate (MFlops/s): %lf Avg time (s): %lf\n",
           1.0E-06 * 2 * ((double)((m-1)*(n-1)))/avgtime, avgtime);
  }

  prk_shmem_finalize();

  exit(EXIT_SUCCESS);

}  /* end of main */

