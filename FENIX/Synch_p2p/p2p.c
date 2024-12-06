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

NAME:    Pipeline_FT

PURPOSE: This program tests the efficiency with which point-to-point
         synchronization can be carried out. It does so by executing 
         a pipelined algorithm on an m*n grid. The first array dimension
         is distributed among the ranks (stripwise decomposition).
  
USAGE:   The program takes as input the dimensions of the grid, and the
         number of times we loop over the grid

               <progname> <# iterations> <m> <n> <spare ranks> \
                          <kill set size> <kill period> <checkpointing>
  
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

#include <signal.h>
#include <sys/types.h>
#include <par-res-kern_general.h>
#include <par-res-kern_fenix.h>
#include <random_draw.h>
#include <unistd.h>

#define ARRAY(i,j) vector[i+1+(j)*(segment_size+1)]

void time_step(int    my_ID,
               int    root, int final,
               long   m, long n,
               long   start, long end,
               long   segment_size,
               int    Num_procs,
               int    grp,
               double * RESTRICT vector,
               double *inbuf, double *outbuf);

int main(int argc, char ** argv)
{
  int    my_ID;           /* MPI rank                                            */
  int    root=0, final;   /* IDs of root rank and rank that verifies result      */
  long   m, n;            /* grid dimensions                                     */
  double pipeline_time, /* timing parameters                                     */
         avgtime;
  double epsilon = 1.e-8; /* error tolerance                                     */
  double corner_val;      /* verification value at top right corner of grid      */
  int    i, j, jj, iter, iter_init;  /* dummies                                  */
  int    iterations;      /* number of times to run the pipeline algorithm       */
  long   start, end;      /* start and end of grid slice owned by calling rank   */
  long   segment_size;    /* size of x-dimension of grid owned by calling rank   */
  int    error=0;         /* error flag                                          */
  int    Num_procs;       /* Number of ranks                                     */
  int    grp;             /* grid line aggregation factor                        */
  int    jjsize;          /* actual line group size                              */
  double * RESTRICT vector;/* array holding grid values                          */
  double *inbuf, *outbuf; /* communication buffers used when aggregating         */
  long   total_length;    /* total required length to store grid values          */
  int    spare_ranks;     /* number of ranks to keep in reserve                  */
  int    kill_ranks;      /* number of ranks that die with each failure          */
  int    *kill_set;       /* instance of set of ranks to be killed               */
  int    kill_period;     /* average number of iterations between failures       */
  int    *fail_iter;      /* list of iterations when a failure will be triggered */
  int    fail_iter_s=0;   /* latest  */
  int    checkpointing;   /* indicates if data is restored using Fenix or
                             analytically                                        */
  int    num_fenix_init=1;/* number of times Fenix_Init is called                */
  int    num_fenix_init_loc;/* number of times Fenix_Init was called             */
  int    fenix_status;
  random_draw_t dice;

/*********************************************************************************
** Initialize the MPI environment
**********************************************************************************/
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_ID);
  MPI_Comm_size(MPI_COMM_WORLD, &Num_procs);

/*********************************************************************
** process, test and broadcast input parameter
*********************************************************************/

  if (my_ID == root){
    printf("Parallel Research Kernels version %s\n", PRKVERSION);
    printf("MPI pipeline execution on 2D grid with Fenix fault tolerance\n");

    if (argc != 8 && argc != 9){
      printf("Usage: %s  <#iterations> <1st array dimension> <2nd array dimension> ",
             *argv);
      printf("<spare ranks><kill set size> <kill period> <checkpointing> [group factor]\n");
      error = 1;
      goto ENDOFTESTS;
    }

    iterations = atoi(argv[1]);
    if (iterations < 1){
      printf("ERROR: iterations must be >= 1 : %d \n",iterations);
      error = 2;
      goto ENDOFTESTS;
    } 

    m = atol(argv[2]);
    n = atol(argv[3]);
    if (m < 1 || n < 1){
      printf("ERROR: grid dimensions must be positive: %ld, %ld \n", m, n);
      error = 3;
      goto ENDOFTESTS;
    }

    spare_ranks  = atoi(argv[4]);
    if (spare_ranks < 0 || spare_ranks >= Num_procs){
      printf("ERROR: Illegal number of spare ranks : %d \n", spare_ranks);
      error = 4;
      goto ENDOFTESTS;     
    }

    if (m<=Num_procs-spare_ranks) {
      printf("ERROR: First grid dimension %ld must be >= number of ranks %d\n", 
             m, Num_procs);
      error = 5;
      goto ENDOFTESTS;
    }

    kill_ranks = atoi(argv[5]);
    if (kill_ranks < 0 || kill_ranks > spare_ranks) {
      printf("ERROR: Number of ranks in kill set invalid: %d\n", kill_ranks);
      error = 6;
      goto ENDOFTESTS;     
    }

    kill_period = atoi(argv[6]);
    if (kill_period < 1) {
      printf("ERROR: rank kill period must be positive: %d\n", kill_period);
      error = 7;
      goto ENDOFTESTS;     
    }

    checkpointing = atoi(argv[7]);
    if (checkpointing) {
      printf("ERROR: Fenix checkpointing not yet implemented\n");
      error = 8;
      goto ENDOFTESTS;     
    }

    if (argc==9) {
      grp = atoi(argv[8]);
      if (grp < 1) grp = 1;
      else if (grp >= n) grp = n-1;
    }
    else grp = 1;

    ENDOFTESTS:;
  }
  bail_out(error); 

  /* Broadcast benchmark data to all rankes */
  MPI_Bcast(&m,             1, MPI_LONG, root, MPI_COMM_WORLD);
  MPI_Bcast(&n,             1, MPI_LONG, root, MPI_COMM_WORLD);
  MPI_Bcast(&grp,           1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&iterations,    1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&spare_ranks,   1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&kill_ranks,    1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&kill_period,   1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&checkpointing, 1, MPI_INT, root, MPI_COMM_WORLD);

  if (my_ID == root) {
    printf("Number of ranks          = %d\n",Num_procs);
    printf("Grid sizes               = %ld, %ld\n", m, n);
    printf("Number of iterations     = %d\n", iterations);
    if (grp > 1)
    printf("Group factor             = %d (cheating!)\n", grp);
    printf("Number of spare ranks    = %d\n", spare_ranks);
    printf("Kill set size            = %d\n", kill_ranks);
    printf("Fault period             = %d\n", kill_period);
    if (checkpointing)
      printf("Data recovery            = Fenix checkpointing\n");
    else
      printf("Data recovery            = analytical\n");
  }

  /* initialize the random number generator for each rank; we do that before
     starting Fenix, so that all ranks, including spares, are initialized      */
  LCG_init(&dice);
  /* compute the iterations during which errors will be incurred               */
  for (iter=0; iter<=iterations; iter++) {
    fail_iter_s += random_draw(kill_period, &dice);
    if (fail_iter_s > iterations) break;
    num_fenix_init++;
  }
  if ((num_fenix_init-1)*kill_ranks>spare_ranks) {
    if (my_ID==0) printf("ERROR: number of injected errors %d exceeds spare ranks %d\n",
                         (num_fenix_init-1)*kill_ranks, spare_ranks);
    error = 9;
  }
  else if(my_ID==root) printf("Total injected failures  = %d times %d errors\n", 
                           num_fenix_init-1, kill_ranks);
  bail_out(error);
  if ((num_fenix_init-1)*kill_ranks>=Num_procs-spare_ranks) if (my_ID==root)
  printf("WARNING: All active ranks will be replaced by recovered ranks; timings not valid\n");

  fail_iter = (int *) prk_malloc(sizeof(int)*num_fenix_init);
  if (!fail_iter) {
    printf("ERROR: Rank %d could not allocate space for array fail_iter\n", my_ID);
    error = 10;
  }
  bail_out(error);

  /* reinitialize random number generator to obtain identical error series     */
  LCG_init(&dice);
  /* now record the actual failure iterations                                  */
  for (fail_iter_s=iter=0; iter<num_fenix_init; iter++) {
    fail_iter_s += random_draw(kill_period, &dice);
    fail_iter[iter] = fail_iter_s;
  }

  /* Here is where we initialize Fenix and mark the return point after failure */
  Fenix_Init(&fenix_status, MPI_COMM_WORLD, NULL, &argc, &argv, spare_ranks, 
             0, MPI_INFO_NULL, &error);

  if (error==FENIX_WARNING_SPARE_RANKS_DEPLETED) 
    printf("ERROR: Rank %d: Cannot reconstitute original communicator\n", my_ID);
  bail_out(error);

  MPI_Comm_rank(MPI_COMM_WORLD, &my_ID);
  MPI_Comm_size(MPI_COMM_WORLD, &Num_procs);

  /* if rank is recovered, set iter to a large number, to be reduced
     to the actual value corresponding to the current iter value among
     survivor ranks; handle number of Fenix_Init calls similarly               */
  switch (fenix_status){
    case FENIX_ROLE_INITIAL_RANK:   iter_init = num_fenix_init_loc = 0;    break;
    case FENIX_ROLE_RECOVERED_RANK: iter_init = num_fenix_init_loc = iterations+1;   break;
    case FENIX_ROLE_SURVIVOR_RANK:  iter_init = iter;  num_fenix_init_loc++;
  }

  MPI_Allreduce(&iter_init, &iter, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&num_fenix_init_loc, &num_fenix_init, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  
  int leftover;
  segment_size = m/(Num_procs);
  leftover     = m%(Num_procs);
  if (my_ID < leftover) {
    start = (segment_size+1)* my_ID;
    end   = start + segment_size;
  }
  else {
    start = (segment_size+1) * leftover + segment_size * (my_ID-leftover);
    end   = start + segment_size -1;
  }

  /* now set segment_size to the value needed by the calling rank               */
  segment_size = end - start + 1;

  if (fenix_status != FENIX_ROLE_SURVIVOR_RANK) {
    /* total_length takes into account one ghost cell on left side of segment   */
    total_length = ((end-start+1)+1)*n;
    vector = (double *) prk_malloc(total_length*sizeof(double));
    if (vector == NULL) {
      printf("Could not allocate space for grid slice of %ld by %ld points",
             segment_size, n);
      printf(" on rank %d\n", my_ID);
      error = 11;
    }
  }
  bail_out(error);

  /* reserve space for in and out buffers                                        */
  if (fenix_status != FENIX_ROLE_SURVIVOR_RANK) {
    inbuf = (double *) prk_malloc(2*sizeof(double)*(grp));
    if (inbuf == NULL) {
      printf("Could not allocate space for %d words of communication buffers", 
              2*grp);
      printf(" on rank %d\n", my_ID);
      error = 12;
    }
    bail_out(error);
    outbuf = inbuf + grp;
  }

  if (iter==0) {
    /* clear the array                                                           */
    for (j=0; j<n; j++) for (i=start-1; i<=end; i++) {
      ARRAY(i-start,j) = 0.0;
    }
  }
  else {
    for (j=0; j<n; j++) for (i=start-1; i<=end; i++) {
      ARRAY(i-start,j) = (double)(iter*(m+n-2) + i + j);
    }
  }

  /* set boundary values (bottom and left side of grid)                           */
  if (my_ID==0) for (j=1; j<n; j++) ARRAY(0,j) = (double) j;
  for (i=start-1; i<=end; i++)      ARRAY(i-start,0) = (double) i;
  
  if (my_ID==0 && iter!=0) {
    ARRAY(0,0) = (double)(-(m+n-2)*iter);
  }

  /* redefine start and end for calling rank to reflect local indices             */
  if (my_ID==0) start = 1; 
  else          start = 0;
  end = segment_size-1;

  /* set final equal to highest rank, because it computes verification value      */
  final = Num_procs-1;


  for (; iter<=iterations; iter++) {

    /* start timer after a warmup iteration */
    if (iter == 1) { 
      MPI_Barrier(MPI_COMM_WORLD);
      pipeline_time = wtime();
    }

    /* inject failure if appropriate                                                */
    if (iter == fail_iter[num_fenix_init]) {
      pid_t pid = getpid();
      if (my_ID < kill_ranks) {
#if VERBOSE
        printf("Rank %d, pid %d commits suicide in iter %d\n", my_ID, pid, iter);
#endif
        kill(pid, SIGKILL);
      }
#if VERBOSE
      else printf("Rank %d, pid %d is survivor rank in iter %d\n", my_ID, pid, iter);
#endif
    }

    time_step(my_ID, root, final, m, n, start, end, segment_size,
              Num_procs, grp, vector, inbuf, outbuf);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  pipeline_time = wtime() - pipeline_time;

  /*******************************************************************************
  ** Analyze and output results.
  ********************************************************************************/
 
  /* verify correctness, using top right value                                     */
  corner_val = (double) ((iterations+1)*(m+n-2));
  if (my_ID == final) {
    if (fabs(ARRAY(end,n-1)-corner_val)/corner_val >= epsilon) {
      printf("ERROR: checksum %lf does not match verification value %lf\n",
             ARRAY(end,n-1), corner_val);
      error = 13;
    }
  }
  MPI_Bcast(&error, 1, MPI_INT, final, MPI_COMM_WORLD);
  bail_out(error);

  if (my_ID == final) {
    avgtime = pipeline_time/iterations;
    /* flip the sign of the execution time to indicate cheating                    */
    if (grp>1) avgtime *= -1.0;
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
 
  Fenix_Finalize();
  MPI_Finalize();
  exit(EXIT_SUCCESS);

}  /* end of main */

