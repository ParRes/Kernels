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

#include <signal.h>
#include <sys/types.h>
#include <par-res-kern_general.h>
#include <par-res-kern_fenix.h>
#include <random_draw.h>
#include <unistd.h>

#define A(i,j)        A_p[(i+istart)+order*(j)]
#define B(i,j)        B_p[(i+istart)+order*(j)]
#define Work_in(i,j)  Work_in_p[i+Block_order*(j)]
#define Work_out(i,j) Work_out_p[i+Block_order*(j)]

void time_step(long Block_order,
	       long Block_size,
	       long Colblock_size,
	       int Tile_order,
	       int tiling,
	       int Num_procs,
	       long order,
	       int my_ID,
	       int colstart,
	       double * RESTRICT A_p,
	       double * RESTRICT B_p,
	       double * RESTRICT Work_in_p,
	       double * RESTRICT Work_out_p);

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
#if !SYNCHRONOUS
  MPI_Request send_req;
  MPI_Request recv_req;
#endif
  long bytes;              /* combined size of matrices             */
  int my_ID;               /* rank                                  */
  int root=0;              /* rank of root                          */
  int iterations;          /* number of times to do the transpose   */
  int i, j, it, jt, istart;/* dummies                               */
  int iter, iter_init;     /* index of iteration                    */
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
  double transpose_time,   /* timing parameters                     */
         avgtime;
  int    spare_ranks;      /* number of ranks to keep in reserve                  */
  int    kill_ranks;       /* number of ranks that die with each failure          */
  int    *kill_set;        /* instance of set of ranks to be killed        */
  int    kill_period;      /* average number of iterations between failures       */
  int    *fail_iter;       /* list of iterations when a failure will be triggered */
  int    fail_iter_s=0;    /* latest  */
  double init_add, addit;  /* used to offset initial solutions       */
  int    checkpointing;    /* indicates if data is restored using Fenix or
                             analytically                            */
  int    num_fenix_init=1; /* number of times Fenix_Init is called   */
  int    num_fenix_init_loc;/* number of times Fenix_Init was called */
  int    fenix_status;
  random_draw_t dice;

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
    printf("MPI matrix transpose with Fenix fault tolerance: B = A^T\n");

    if (argc != 7 && argc != 8){
      printf("Usage: %s <# iterations> <matrix order> <spare ranks> ",
                                                               *argv);
      printf("<kill set size> <kill period> <checkpointing> [Tile size]\n",
                                                               *argv);
      error = 1; goto ENDOFTESTS;
    }

    iterations  = atoi(argv[1]);
    if(iterations < 1){
      printf("ERROR: iterations must be >= 1 : %d \n",iterations);
      error = 1; goto ENDOFTESTS;
    }

    order = atol(argv[2]);
    spare_ranks  = atoi(argv[3]);
    if (order < Num_procs-spare_ranks) {
      printf("ERROR: matrix order %ld should at least # procs %d\n",
             order, Num_procs-spare_ranks);
      error = 1; goto ENDOFTESTS;
    }
    if (order%(Num_procs-spare_ranks)) {
      printf("ERROR: matrix order %ld should be divisible by # procs %d\n",
             order, Num_procs-spare_ranks);
      error = 1; goto ENDOFTESTS;
    }

    if (spare_ranks < 0 || spare_ranks >= Num_procs){
      printf("ERROR: Illegal number of spare ranks : %d \n", spare_ranks);
      error = 1;
      goto ENDOFTESTS;     
    }

    kill_ranks = atoi(argv[4]);
    if (kill_ranks < 0 || kill_ranks > spare_ranks) {
      printf("ERROR: Number of ranks in kill set invalid: %d\n", kill_ranks);
      error = 1;
      goto ENDOFTESTS;     
    }

    kill_period = atoi(argv[5]);
    if (kill_period < 1) {
      printf("ERROR: rank kill period must be positive: %d\n", kill_period);
      error = 1;
      goto ENDOFTESTS;     
    }

    checkpointing = atoi(argv[6]);
    if (checkpointing) {
      printf("ERROR: Fenix checkpointing not yet implemented\n");
      error = 1;
      goto ENDOFTESTS;     
    }

    if (argc == 8) Tile_order = atoi(argv[7]);

    ENDOFTESTS:;
  }
  bail_out(error);

  /*  Broadcast input data to all ranks */
  MPI_Bcast(&order,         1, MPI_LONG, root, MPI_COMM_WORLD);
  MPI_Bcast(&iterations,    1, MPI_INT,  root, MPI_COMM_WORLD);
  MPI_Bcast(&Tile_order,    1, MPI_INT,  root, MPI_COMM_WORLD);
  MPI_Bcast(&spare_ranks,   1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&kill_ranks,    1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&kill_period,   1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&checkpointing, 1, MPI_INT, root, MPI_COMM_WORLD);

  if (my_ID == root) {
    printf("Number of ranks       = %d\n", Num_procs);
    printf("Matrix order          = %ld\n", order);
    printf("Number of iterations  = %d\n", iterations);
    if ((Tile_order > 0) && (Tile_order < order))
          printf("Tile size             = %d\n", Tile_order);
    else  printf("Untiled\n");
#if !SYNCHRONOUS
    printf("Non-");
#endif
    printf("Blocking messages\n");
    printf("Number of spare ranks = %d\n", spare_ranks);
    printf("Kill set size         = %d\n", kill_ranks);
    printf("Fault period          = %d\n", kill_period);
    if (checkpointing)
      printf("Data recovery         = Fenix checkpointing\n");
    else
      printf("Data recovery         = analytical\n");
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
    error = 1;
  }
  else if(my_ID==root) printf("Total injected failures  = %d times %d errors\n", 
                           num_fenix_init-1, kill_ranks);
  bail_out(error);
  if ((num_fenix_init-1)*kill_ranks>=Num_procs-spare_ranks) if (my_ID==root)
  printf("WARNING: All active ranks will be replaced by recovered ranks; timings not valid\n");

  fail_iter = (int *) prk_malloc(sizeof(int)*num_fenix_init);
  if (!fail_iter) {
    printf("ERROR: Rank %d could not allocate space for array fail_iter\n", my_ID);
    error = 1;
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

/*********************************************************************
** Create the column block of the test matrix, the row block of the
** transposed matrix, and workspace (workspace only if #procs>1)
*********************************************************************/
  if (fenix_status != FENIX_ROLE_SURVIVOR_RANK) {
    A_p = (double *)prk_malloc(Colblock_size*sizeof(double));
    if (A_p == NULL){
      printf(" Error allocating space for original matrix on node %d\n",my_ID);
      error = 1;
    }
  }
  bail_out(error);

  if (fenix_status != FENIX_ROLE_SURVIVOR_RANK) {
    B_p = (double *)prk_malloc(Colblock_size*sizeof(double));
    if (B_p == NULL){
      printf(" Error allocating space for transpose matrix on node %d\n",my_ID);
      error = 1;
    }
  }
  bail_out(error);

  if (fenix_status != FENIX_ROLE_SURVIVOR_RANK && Num_procs>1) {
    Work_in_p   = (double *)prk_malloc(2*Block_size*sizeof(double));
    if (Work_in_p == NULL){
      printf(" Error allocating space for work on node %d\n",my_ID);
      error = 1;
    }
    Work_out_p = Work_in_p + Block_size;
  }
  bail_out(error);

  /* Fill the original column matrix                                                */
  /* intialize the input and output arrays, note that if we use the analytical
     solution to initialize, one might be tempted to skip this step for survivor
     ranks, because they already have the correct (interim) values. That would
     be wrong for two reasons: It is possible for ranks to be in different time
     steps at the same time, and it is possible that error signal delivery to
     a rank is delayed                                                         */
  if (checkpointing) {
    init_add = 0.0;
    addit    = 0.0;
  }
  else {
    init_add = (double) iter;
    addit = ((double)(iter-1) * (double) (iter))/2.0;
  }
  istart = 0;
  for (j=0;j<Block_order;j++)
    for (i=0;i<order; i++)  {
      A(i,j) = (double) (order*(j+colstart) + i) + init_add;
      B(i,j) = ((double) ((j+colstart) + order*i)*iter + addit);
  }

  for (; iter<=iterations; iter++){

    /* start timer after a warmup iteration                                        */
    if (iter == 1) {
      MPI_Barrier(MPI_COMM_WORLD);
      transpose_time = wtime();
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

    time_step(Block_order, Block_size, Colblock_size, Tile_order, tiling,
              Num_procs, order, my_ID, colstart, A_p, B_p, Work_in_p, Work_out_p);

  } /* end of iterations */

  MPI_Barrier(MPI_COMM_WORLD);
  transpose_time = wtime() - transpose_time;;

  abserr = 0.0;
  istart = 0;
  addit = ((double)(iterations+1) * (double) (iterations))/2.0;
  for (j=0;j<Block_order;j++) for (i=0;i<order; i++) {
      abserr += ABS(B(i,j) - (double)((order*i + j+colstart)*(iterations+1)+addit));
  }

  root = Num_procs-1;
  MPI_Reduce(&abserr, &abserr_tot, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

  if (my_ID == root) {
    if (abserr_tot < epsilon) {
      printf("Solution validates\n");
      avgtime = transpose_time/(double)iterations;
      printf("Rate (MB/s): %lf Avg time (s): %lf\n",1.0E-06*bytes/avgtime, avgtime);
#if VERBOSE
      printf("Summed errors: %f \n", abserr);
#endif
    }
    else {
      printf("ERROR: Aggregate squared error %lf exceeds threshold %e\n", abserr, epsilon);
      error = 1;
    }
  }

  bail_out(error);

  Fenix_Finalize();
  MPI_Finalize();
  exit(EXIT_SUCCESS);

}  /* end of main */

