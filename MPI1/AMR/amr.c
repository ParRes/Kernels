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

NAME:    AMR

PURPOSE: This program tests the efficiency with which a space-invariant,
         linear, symmetric filter (stencil) can be applied to a square
         grid or image, with periodic introduction and removal of
         subgrids.
  
USAGE:   Type ./amr for full list of parameters

FUNCTIONS CALLED:

         Other than standard C functions, the following functions are used in 
         this program:
         wtime()

HISTORY: - Written by Rob Van der Wijngaart, February September 2016.
         - RvdW: Removed unrolling pragmas for clarity;
           added constant to array "in" at end of each iteration to force 
           refreshing of neighbor data in parallel versions; August 2013
  
**********************************************************************************/

#include <par-res-kern_general.h>
#include <par-res-kern_mpi.h>

#if DOUBLE
  #define DTYPE     double
  #define MPI_DTYPE MPI_DOUBLE
  #define EPSILON   1.e-8
  #define COEFX     1.0
  #define COEFY     1.0
  #define FSTR      "%10.4lf"
#else
  #define DTYPE     float
  #define MPI_DTYPE MPI_FLOAT
  #define EPSILON   0.0001f
  #define COEFX     1.0f
  #define COEFY     1.0f
  #define FSTR      "%10.4f"
#endif

/* define shorthand for indexing multi-dimensional arrays                       */
#define INDEXIN(i,j)     (i+RADIUS+(j+RADIUS)*(L_width_bg+2*RADIUS))
/* need to add offset of RADIUS to j to account for ghost points                */
#define IN(i,j)          in_bg[INDEXIN(i-L_istart_bg,j-L_jstart_bg)]
#define INDEXIN_R(g,i,j) (i+RADIUS+(j+RADIUS)*(L_width_r_true_gross[g]+2*RADIUS))
#define INDEXIN_RG(i,j)  (i+RADIUS+(j+RADIUS)*(L_width_r_true_gross+2*RADIUS))
#define IN_R(g,i,j)      in_r[g][INDEXIN_R(g,i-L_istart_r_true_gross[g],j-L_jstart_r_true_gross[g])]
#define ING_R(i,j)       ing_r[INDEXIN_RG(i-L_istart_r_true_gross,j-L_jstart_r_true_gross)]
#define INDEXOUT(i,j)    (i+(j)*(L_width_bg))
#define OUT(i,j)         out_bg[INDEXOUT(i-L_istart_bg,j-L_jstart_bg)]
#define INDEXOUT_R(i,j)  (i+(j)*L_width_r_true_gross[g])
#define OUT_R(g,i,j)     out_r[g][INDEXOUT_R(i-L_istart_r_true_gross[g],j-L_jstart_r_true_gross[g])]
#define WEIGHT(ii,jj)    weight[ii+RADIUS][jj+RADIUS]
#define WEIGHT_R(ii,jj)  weight_r[ii+RADIUS][jj+RADIUS]

#define undefined        1111
#define fine_grain       9797
#define no_talk          1212
#define high_water       3232

int main(int argc, char ** argv) {

  int    Num_procs;         /* number of ranks                                     */
  int    Num_procs_bg;      /* number of ranks in BG                               */
  int    Num_procs_bgx, Num_procs_bgy; /* number of ranks in each coord direction  */
  int    Num_procs_r[4];    /* number of ranks in refinements                      */
  int    Num_procs_rx[4], Num_procs_ry[4];
  int    my_ID;             /* MPI rank                                            */
  int    my_ID_bg;          /* MPI rank on BG grid (-1 if not present)             */
  int    my_ID_bgx, my_ID_bgy;/* coordinates of rank in BG rank grid               */
  int    my_ID_r[4];        /* rank within refinement                              */
  int    my_ID_rx[4], my_ID_ry[4];/* coordinates of rank in refinement             */
  int    right_nbr_bg;      /* global rank of right neighboring BG tile            */
  int    left_nbr_bg;       /* global rank of left neighboring BG tile             */
  int    top_nbr_bg;        /* global rank of top neighboring BG tile              */
  int    bottom_nbr_bg;     /* global rank of bottom neighboring BG tile           */
  int    right_nbr_r[4];    /* global rank of right neighboring ref tile           */
  int    left_nbr_r[4];     /* global rank of left neighboring ref tile            */
  int    top_nbr_r[4];      /* global rank of top neighboring ref tile             */
  int    bottom_nbr_r[4];   /* global rank of bottom neighboring ref tile          */
  DTYPE  *top_buf_out_bg;   /* BG communication buffer                             */
  DTYPE  *top_buf_in_bg;    /* "     "         "                                   */
  DTYPE  *bottom_buf_out_bg;/* "     "         "                                   */
  DTYPE  *bottom_buf_in_bg; /* "     "         "                                   */
  DTYPE  *right_buf_out_bg; /* "     "         "                                   */
  DTYPE  *right_buf_in_bg;  /* "     "         "                                   */
  DTYPE  *left_buf_out_bg;  /* "     "         "                                   */
  DTYPE  *left_buf_in_bg;   /* "     "         "                                   */
  DTYPE  *top_buf_out_r[4]; /* refinement communication buffer                     */
  DTYPE  *top_buf_in_r[4];  /*       "         "          "                        */
  DTYPE  *bottom_buf_out_r[4];/*     "         "          "                        */
  DTYPE  *bottom_buf_in_r[4];/*      "         "          "                        */
  DTYPE  *right_buf_out_r[4];/*      "         "          "                        */
  DTYPE  *right_buf_in_r[4];/*       "         "          "                        */
  DTYPE  *left_buf_out_r[4];/*       "         "          "                        */
  DTYPE  *left_buf_in_r[4]; /*       "         "          "                        */
  int    root = 0;
  long   n;                 /* linear grid dimension                               */
  int    refine_level;      /* refinement level                                    */
  long   G_istart_r[4];     /* global left boundaries of refinements               */
  long   G_iend_r[4];       /* global right boundaries of refinements              */
  long   G_jstart_r[4];     /* global bottom boundaries of refinements             */
  long   G_jend_r[4];       /* global top boundaries of refinements                */
  long   L_istart_bg, L_iend_bg;/* bounds of BG tile assigned to calling rank      */
  long   L_jstart_bg, L_jend_bg;/* bounds of BG tile assigned to calling rank      */
  long   L_width_bg, L_height_bg;/* local BG dimensions                            */
  long   L_istart_r[4], L_iend_r[4];/* bounds of refinement tile for calling rank  */
  long   L_jstart_r[4], L_jend_r[4];/* bounds of refinement tile for calling rank  */
  long   L_istart_r_gross[4], L_iend_r_gross[4]; /* see implemenation_details.md   */
  long   L_jstart_r_gross[4], L_jend_r_gross[4]; /*             "                  */
  long   L_istart_r_true_gross[4], L_iend_r_true_gross[4]; /*   "                  */
  long   L_jstart_r_true_gross[4], L_jend_r_true_gross[4]; /*   "                  */
  long   L_istart_r_true[4], L_iend_r_true[4]; /*               "                  */
  long   L_jstart_r_true[4], L_jend_r_true[4]; /*               "                  */
  long   L_width_r[4], L_height_r[4]; /* local refinement dimensions               */
  long   L_width_r_true_gross[4], L_height_r_true_gross[4];/* "            "       */
  long   L_width_r_true[4], L_height_r_true[4];/*             "            "       */
  int    g;                 /* refinement grid index                               */
  long   n_r;               /* linear refinement size in bg grid points            */
  long   n_r_true;          /* linear refinement size                              */
  long   expand;            /* number of refinement cells per background cell      */
  int    period;            /* refinement period                                   */
  int    duration;          /* lifetime of a refinement                            */
  int    sub_iterations;    /* number of sub-iterations on refinement              */
  long   i, j, ii, jj, it, jt, l, leftover; /* dummies                             */
  int    iter, sub_iter;    /* dummies                                             */
  DTYPE  norm, local_norm,  /* L1 norm of solution on background grid              */
         reference_norm;
  DTYPE  norm_in,           /* L1 norm of input field on background grid           */
         local_norm_in,
         reference_norm_in;
  DTYPE  norm_r[4],         /* L1 norm of solution on refinements                  */
         local_norm_r[4],
         reference_norm_r[4];
  DTYPE  norm_in_r[4],      /* L1 norm of input field on refinements               */
         local_norm_in_r[4],
         reference_norm_in_r[4];
  DTYPE  h_r;               /* mesh spacing of refinement                          */
  DTYPE  f_active_points_bg;/* interior of grid with respect to stencil            */
  DTYPE  f_active_points_r; /* interior of refinement with respect to stencil      */
  DTYPE  flops;             /* total floating point ops       `                    */
  int    iterations;        /* number of times to run the algorithm                */
  int    iterations_r[4];   /* number of iterations on each refinement             */
  int    full_cycles;       /* number of full cycles all refinement grids appear   */
  int    leftover_iterations;/* number of iterations in last partial AMR cycle     */
  int    num_interpolations;/* total number of timed interpolations                */
  int    bg_updates;        /* # background grid updates before last interpolation */
  int    r_updates;         /* # refinement updates since last interpolation       */ 
  double stencil_time,      /* timing parameters                                   */
         local_stencil_time,
         avgtime;
  int    stencil_size;      /* number of points in stencil                         */
  DTYPE  * RESTRICT in_bg;  /* background grid input values                        */
  DTYPE  * RESTRICT out_bg; /* background grid output values                       */
  DTYPE  * RESTRICT in_r[4];/* refinement grid input values                        */
  DTYPE  * RESTRICT out_r[4];/* refinement grid output values                      */
  long   total_length_in;   /* total required length for bg grid values in         */
  long   total_length_out;  /* total required length for bg grid values out        */
  long   total_length_in_r[4]; /* total required length for refinement values in   */
  long   total_length_out_r[4];/* total required length for refinement values out  */
  DTYPE  weight[2*RADIUS+1][2*RADIUS+1]; /* weights of points in the stencil       */
  DTYPE  weight_r[2*RADIUS+1][2*RADIUS+1]; /* weights of points in the stencil     */
  int    error=0;           /* error flag                                          */
  int    validate=1;        /* tracks correct solution on all grids                */
  char   *c_load_balance;   /* input string defining load balancing                */
  int    load_balance;      /* integer defining load balancing                     */
  MPI_Request request_bg[8];
  MPI_Request request_r[4][8];
  MPI_Comm comm_r[4];       /* communicators for refinements                       */
  MPI_Comm comm_bg;         /* communicator for BG                                 */
  int    color_r;           /* color used to create refinement communicators       */
  int    color_bg;          /* color used to create BG communicator                */
  int    rank_spread;       /* number of ranks for refinement in fine_grain        */

  /*********************************************************************************
  ** Initialize the MPI environment
  **********************************************************************************/
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_ID);
  MPI_Comm_size(MPI_COMM_WORLD, &Num_procs);

  /*********************************************************************************
  ** process, test, and broadcast input parameters    
  **********************************************************************************/
 
  if (my_ID == root) {
    printf("Parallel Research Kernels Version %s\n", PRKVERSION);
    printf("MPI AMR stencil execution on 2D grid\n");

#if !STAR
    printf("ERROR: Compact stencil not supported\n");
    error = 1;
    goto ENDOFINPUTTESTS;
#endif

    if (argc != 9 && argc != 10){
      printf("Usage: %s <# iterations> <background grid size> <refinement size>\n",
             *argv);
      printf("       <refinement level> <refinement period> <refinement duration>\n");
      printf("       <refinement sub-iterations> <load balancer> \n");
      printf("       load balancer: FINE_GRAIN [refinement rank spread]\n");
      printf("                      NO_TALK\n");
      printf("                      HIGH_WATER\n");
      error = 1;
      goto ENDOFINPUTTESTS;
    }

    iterations  = atoi(*++argv); 
    if (iterations < 1){
      printf("ERROR: iterations must be >= 1 : %d \n",iterations);
      error = 1;
      goto ENDOFINPUTTESTS;
    }

    n  = atol(*++argv);

    if (n < 2){
      printf("ERROR: grid must have at least one cell: %ld\n", n);
      error = 1;
      goto ENDOFINPUTTESTS;
    }

    n_r = atol(*++argv);
    if (n_r < 2) {
      printf("ERROR: refinements must have at least one cell: %ld\n", n_r);
      error = 1;
      goto ENDOFINPUTTESTS;
    }
    if (n_r>n) {
      printf("ERROR: refinements must be contained in background grid: %ld\n", n_r);
      error = 1;
      goto ENDOFINPUTTESTS;
    }

    refine_level = atoi(*++argv);
    if (refine_level < 0) {
      printf("ERROR: refinement levels must be >= 0 : %d\n", refine_level);
      error = 1;
      goto ENDOFINPUTTESTS;
    }

    period = atoi(*++argv);
    if (period < 1) {
      printf("ERROR: refinement period must be at least one: %d\n", period);
      error = 1;
      goto ENDOFINPUTTESTS;
    }
  
    duration = atoi(*++argv);
    if (duration < 1 || duration > period) {
      printf("ERROR: refinement duration must be positive, no greater than period: %d\n",
             duration);
      error = 1;
      goto ENDOFINPUTTESTS;
    }
 
    sub_iterations = atoi(*++argv);
    if (sub_iterations < 1) {
      printf("ERROR: refinement sub-iterations must be positive: %d\n", sub_iterations);
      error = 1;
      goto ENDOFINPUTTESTS;
    }

    c_load_balance = *++argv;
    if      (!strcmp("FINE_GRAIN", c_load_balance)) load_balance=fine_grain;
    else if (!strcmp("NO_TALK",    c_load_balance)) load_balance=no_talk;
    else if (!strcmp("HIGH_WATER", c_load_balance)) load_balance=high_water;
    else                                            load_balance=undefined;
    if (load_balance==undefined) {
      printf("ERROR: invalid load balancer %s\n", c_load_balance);
      error = 1;
      goto ENDOFINPUTTESTS;
    }

    if (load_balance == high_water && Num_procs==1) {
      printf("ERROR: Load balancer HIGH_WATER requires more than one rank\n");
      error = 1;
      goto ENDOFINPUTTESTS;
    }

    if (load_balance==fine_grain && argc==10) {
      rank_spread = atoi(*++argv);
      if (rank_spread<1 || rank_spread>Num_procs) {
	printf("ERROR: Invalid number of ranks to spread refinement work: %d\n", rank_spread);
	error = 1;
	goto ENDOFINPUTTESTS;
      }
    } else rank_spread = Num_procs;

    if (RADIUS < 1) {
      printf("ERROR: Stencil radius %d should be positive\n", RADIUS);
      error = 1;
      goto ENDOFINPUTTESTS;
    }

    if (2*RADIUS+1 > n) {
      printf("ERROR: Stencil radius %d exceeds grid size %ld\n", RADIUS, n);
      error = 1;
      goto ENDOFINPUTTESTS;
    }

    /* calculate refinement mesh spacing plus ratio of mesh spacings */
    h_r = (DTYPE)1.0; expand = 1;
    for (l=0; l<refine_level; l++) {
      h_r /= (DTYPE)2.0;
      expand *= 2;
    }
    n_r_true = (n_r-1)*expand+1;
    if (2*RADIUS+1 > n_r_true) {
      printf("ERROR: Stencil radius %d exceeds refinement size %ld\n", RADIUS, n_r_true);
      error = 1;
      goto ENDOFINPUTTESTS;
    }

    ENDOFINPUTTESTS:;  
  }
  bail_out(error);

  MPI_Bcast(&n,              1, MPI_LONG,  root, MPI_COMM_WORLD);
  MPI_Bcast(&n_r,            1, MPI_LONG,  root, MPI_COMM_WORLD);
  MPI_Bcast(&h_r,            1, MPI_DTYPE, root, MPI_COMM_WORLD);
  MPI_Bcast(&n_r_true,       1, MPI_LONG,  root, MPI_COMM_WORLD);
  MPI_Bcast(&period,         1, MPI_INT,   root, MPI_COMM_WORLD);
  MPI_Bcast(&duration,       1, MPI_INT,   root, MPI_COMM_WORLD);
  MPI_Bcast(&refine_level,   1, MPI_INT,   root, MPI_COMM_WORLD);
  MPI_Bcast(&iterations,     1, MPI_INT,   root, MPI_COMM_WORLD);
  MPI_Bcast(&sub_iterations, 1, MPI_INT,   root, MPI_COMM_WORLD);
  MPI_Bcast(&load_balance,   1, MPI_INT,   root, MPI_COMM_WORLD);
  MPI_Bcast(&rank_spread,    1, MPI_INT,   root, MPI_COMM_WORLD);
  MPI_Bcast(&expand,         1, MPI_LONG,  root, MPI_COMM_WORLD);

  /* depending on the load balancing strategy chosen, we determine the 
     partitions of BG (background grid) and the refinements                  */
  float bg_size, total_size, Frac_procs_bg; // used for HIGH_WATER
                   
  switch (load_balance) {
  case fine_grain: MPI_Comm_dup(MPI_COMM_WORLD, &comm_bg);
                   Num_procs_bg = Num_procs;
                   my_ID_bg = my_ID;
                   for (g=0; g<4; g++) {
                     if (my_ID < rank_spread) color_r = 1;
                     else                     color_r = MPI_UNDEFINED;
                     MPI_Comm_split(MPI_COMM_WORLD, color_r, my_ID, &comm_r[g]);
                     if (comm_r[g] != MPI_COMM_NULL) {
                       MPI_Comm_size(comm_r[g], &Num_procs_r[g]);
                       MPI_Comm_rank(comm_r[g], &my_ID_r[g]);
		     }
                   }
                   break;
  case no_talk:    MPI_Comm_dup(MPI_COMM_WORLD, &comm_bg);
                   Num_procs_bg = Num_procs;
                   my_ID_bg = my_ID;
                   break;
  case high_water: bg_size=n*n; 
                   total_size = n*n+n_r_true*n_r_true;
                   Frac_procs_bg;
                   Frac_procs_bg = (float) Num_procs * bg_size/total_size;
                   Num_procs_bg = MIN(Num_procs-1,MAX(1,ceil(Frac_procs_bg)));

		   /* Adjust number of BG procs to avoid pathological aspect ratios */
		   int Num_procs_R = Num_procs-Num_procs_bg;
		   optimize_split(&Num_procs_bg, &Num_procs_R, 3);

                   if (my_ID>=Num_procs_bg) {color_bg = MPI_UNDEFINED; color_r = 1;}
                   else                     {color_bg = 1; color_r = MPI_UNDEFINED;}
                   MPI_Comm_split(MPI_COMM_WORLD, color_bg, my_ID, &comm_bg);
		   if (comm_bg != MPI_COMM_NULL) {
                     MPI_Comm_size(comm_bg, &Num_procs_bg);
                     MPI_Comm_rank(comm_bg, &my_ID_bg);
		   }
                   for (g=0; g<4; g++) {
                     MPI_Comm_split(MPI_COMM_WORLD, color_r, my_ID, &comm_r[g]);
		     if (comm_r[g] != MPI_COMM_NULL) {
                       MPI_Comm_size(comm_r[g], &Num_procs_r[g]);
                       MPI_Comm_rank(comm_r[g], &my_ID_r[g]);
		     } 
                     else {
                       Num_procs_r[g] = Num_procs - Num_procs_bg;
                     }
                   }
		   if (comm_bg == MPI_COMM_NULL) Num_procs_bg = Num_procs - Num_procs_r[0];
                   break;
  }

  /* do bookkeeping for background grid                                       */
  if (comm_bg != MPI_COMM_NULL) {
    /* determine best way to create a 2D grid of ranks (closest to square)    */
    factor(Num_procs_bg, &Num_procs_bgx, &Num_procs_bgy);

    /* communication neighbors on BG are computed for all who own part of it  */
    my_ID_bgx = my_ID_bg%Num_procs_bgx;
    my_ID_bgy = my_ID_bg/Num_procs_bgx;
    /* compute neighbors; catch dropping off edges of grid                    */
    right_nbr_bg = left_nbr_bg = top_nbr_bg = bottom_nbr_bg = -1;
    if (my_ID_bgx < Num_procs_bgx-1) right_nbr_bg  = my_ID+1;
    if (my_ID_bgx > 0)               left_nbr_bg   = my_ID-1;
    if (my_ID_bgy < Num_procs_bgy-1) top_nbr_bg    = my_ID+Num_procs_bgx;
    if (my_ID_bgy > 0)               bottom_nbr_bg = my_ID-Num_procs_bgx;

    /* create decomposition and reserve space for BG input/output fields      */
    L_width_bg = n/Num_procs_bgx;
    leftover = n%Num_procs_bgx;

   if (my_ID_bgx<leftover) {
      L_istart_bg = (L_width_bg+1) * my_ID_bgx; 
      L_iend_bg = L_istart_bg + L_width_bg;
    }
    else {
      L_istart_bg = (L_width_bg+1) * leftover + L_width_bg * (my_ID_bgx-leftover);
      L_iend_bg = L_istart_bg + L_width_bg - 1;
    }
    
    L_width_bg = L_iend_bg - L_istart_bg + 1;
    if (L_width_bg == 0) {
      printf("ERROR: rank %d has no work to do\n", my_ID);
      error = 1;
      goto ENDOFBG;
    }
  
    L_height_bg = n/Num_procs_bgy;
    leftover = n%Num_procs_bgy;

    if (my_ID_bgy<leftover) {
      L_jstart_bg = (L_height_bg+1) * my_ID_bgy; 
      L_jend_bg = L_jstart_bg + L_height_bg;
    }
    else {
      L_jstart_bg = (L_height_bg+1) * leftover + L_height_bg * (my_ID_bgy-leftover);
      L_jend_bg = L_jstart_bg + L_height_bg - 1;
    }
    
    L_height_bg = L_jend_bg - L_jstart_bg + 1;
    if (L_height_bg == 0) {
      printf("ERROR: rank %d has no work to do\n", my_ID);
      error = 1;
      goto ENDOFBG;
    }

    if (L_width_bg < RADIUS || L_height_bg < RADIUS) {
      printf("ERROR: rank %d's BG work tile smaller than stencil radius: %ld\n",
             my_ID, MIN(L_width_bg, L_height_bg));
      error = 1;
      goto ENDOFBG;
    }

    total_length_in  = (long) (L_width_bg+2*RADIUS)*(long) (L_height_bg+2*RADIUS);
    total_length_out = (long) L_width_bg* (long) L_height_bg;

    in_bg  = (DTYPE *) prk_malloc(total_length_in*sizeof(DTYPE));
    out_bg = (DTYPE *) prk_malloc(total_length_out*sizeof(DTYPE));
    if (!in_bg || !out_bg) {
      printf("ERROR: rank %d could not allocate space for input/output array\n",
              my_ID);
      error = 1;
      goto ENDOFBG;
    }
    ENDOFBG:;
  }
  else { // bogus empty patch
    L_istart_bg =  0;
    L_iend_bg   = -1;
    L_jstart_bg =  0;;
    L_jend_bg   = -1;
  }
  bail_out(error);
  
  /* compute global layout of refinements                                      */
  G_istart_r[0] = G_istart_r[2] = 0;
  G_iend_r[0]   = G_iend_r[2]   = n_r-1;
  G_istart_r[1] = G_istart_r[3] = n-n_r;
  G_iend_r[1]   = G_iend_r[3]   = n-1;
  G_jstart_r[0] = G_jstart_r[3] = 0;
  G_jend_r[0]   = G_jend_r[3]   = n_r-1;
  G_jstart_r[1] = G_jstart_r[2] = n-n_r;
  G_jend_r[1]   = G_jend_r[2]   = n-1;
  
  /* compute tiling of refinements                                             */
  switch(load_balance) {
  case no_talk:    // check if calling rank's BG patch overlaps with refinement*/
                   for (g=0; g<4; g++) {
		     L_istart_r[g] = MAX(L_istart_bg,G_istart_r[g]);
		     L_iend_r[g]   = MIN(L_iend_bg,  G_iend_r[g]);		     
		     L_jstart_r[g] = MAX(L_jstart_bg,G_jstart_r[g]);
		     L_jend_r[g]   = MIN(L_jend_bg,  G_jend_r[g]);
                     if (L_istart_r[g]<=L_iend_r[g] &&
			 L_jstart_r[g]<=L_jend_r[g]) color_r = 1;
		     else                            color_r = MPI_UNDEFINED;
		     MPI_Comm_split(MPI_COMM_WORLD, color_r, my_ID, &comm_r[g]);
		     if (comm_r[g] != MPI_COMM_NULL) {
                       MPI_Comm_size(comm_r[g], &Num_procs_r[g]);
                       MPI_Comm_rank(comm_r[g], &my_ID_r[g]);
		       // determine layout of subset
		       long ilow, ihigh, jlow, jhigh;
                       MPI_Allreduce(&my_ID_bgx,&ilow ,1,MPI_LONG,MPI_MIN,comm_r[g]);
                       MPI_Allreduce(&my_ID_bgx,&ihigh,1,MPI_LONG,MPI_MAX,comm_r[g]);
                       MPI_Allreduce(&my_ID_bgy,&jlow ,1,MPI_LONG,MPI_MIN,comm_r[g]);
                       MPI_Allreduce(&my_ID_bgy,&jhigh,1,MPI_LONG,MPI_MAX,comm_r[g]);
		       Num_procs_rx[g] = ihigh-ilow+1;
		       Num_procs_ry[g] = jhigh-jlow+1;
		     }
		   }
                   break;
  case fine_grain: 
  case high_water: // refinements are partitioned independently, but similar to BG
                   for (g=0; g<4; g++) if (comm_r[g] != MPI_COMM_NULL) {
		     factor(Num_procs_r[g], &Num_procs_rx[g], &Num_procs_ry[g]);
		   }
                   break;
  }

  /* compute communication neighbors on refinements                           */
  for (g=0; g<4; g++) if (comm_r[g] != MPI_COMM_NULL) {
    my_ID_rx[g] = my_ID_r[g]%Num_procs_rx[g];
    my_ID_ry[g] = my_ID_r[g]/Num_procs_rx[g];
    /* compute neighbors; catch dropping off edges of grid                    */
    right_nbr_r[g] = left_nbr_r[g] = top_nbr_r[g] = bottom_nbr_r[g] = -1;
    if (my_ID_rx[g] < Num_procs_rx[g]-1) right_nbr_r[g]  = my_ID_r[g]+1;
    if (my_ID_rx[g] > 0)                 left_nbr_r[g]   = my_ID_r[g]-1;
    if (my_ID_ry[g] < Num_procs_ry[g]-1) top_nbr_r[g]    = my_ID_r[g]+Num_procs_rx[g];
    if (my_ID_ry[g] > 0)                 bottom_nbr_r[g] = my_ID_r[g]-Num_procs_rx[g];
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (my_ID == root) {
    printf("Number of ranks                 = %d\n", Num_procs);
    printf("Background grid size            = %ld\n", n);
    printf("Radius of stencil               = %d\n", RADIUS);
    printf("Tiles in x/y-direction on BG    = %d/%d\n", Num_procs_bgx, Num_procs_bgy);
  }
  for (g=0; g<4; g++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if ((comm_r[g] != MPI_COMM_NULL) && (my_ID_r[g]==root))
      printf("Tiles in x/y-direction on ref %d = %d/%d\n",
	     g, Num_procs_rx[g], Num_procs_ry[g]);
    prk_pause(0.001); // wait for a short while to ensure proper I/O ordering
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (my_ID == root) {
    printf("Type of stencil                 = star\n");
#if DOUBLE
    printf("Data type                       = double precision\n");
#else
    printf("Data type                       = single precision\n");
#endif
#if LOOPGEN
    printf("Script used to expand stencil loop body\n");
#else
    printf("Compact representation of stencil loop body\n");
#endif
    printf("Number of iterations            = %d\n", iterations);
    printf("Load balancer                   = %s\n", c_load_balance);
    if (load_balance==fine_grain)
      printf("Refinement rank spread          = %d\n", rank_spread);
    printf("Refinements:\n");
    printf("   Background grid points       = %ld\n", n_r);
    printf("   Grid size                    = %ld\n", n_r_true);
    printf("   Refinement level             = %d\n", refine_level);
    printf("   Period                       = %d\n", period);
    printf("   Duration                     = %d\n", duration);
    printf("   Sub-iterations               = %d\n", sub_iterations);
  }

  /* reserve space for refinement input/output fields; first compute extents */

  /* we partition the refinement in terms of BG indices, so that we know 
     for the fine_grain balancer that a rank's refinement partitition does 
     not need BG data beyond the boundary of the refinement as input to the 
     interpolation                                                           */
  for (g=0; g<4; g++) if (comm_r[g] != MPI_COMM_NULL) {
    if (load_balance==fine_grain || load_balance==high_water) {

      L_width_r[g] = n_r/Num_procs_rx[g];
      leftover =   n_r%Num_procs_rx[g];

      if (my_ID_rx[g]<leftover) {
        L_istart_r[g] = (L_width_r[g]+1) * my_ID_rx[g]; 
        L_iend_r[g]   = L_istart_r[g] + L_width_r[g];
      }
      else {
        L_istart_r[g] = (L_width_r[g]+1) * leftover + L_width_r[g] * (my_ID_rx[g]-leftover);
        L_iend_r[g]   = L_istart_r[g] + L_width_r[g] - 1;
      }
  
      L_height_r[g] = n_r/Num_procs_ry[g];
      leftover = n_r%Num_procs_ry[g];

      if (my_ID_ry[g]<leftover) {
        L_jstart_r[g] = (L_height_r[g]+1) * my_ID_ry[g]; 
        L_jend_r[g]   = L_jstart_r[g] + L_height_r[g];
      }
      else {
        L_jstart_r[g] = (L_height_r[g]+1) * leftover + L_height_r[g] * (my_ID_ry[g]-leftover);
        L_jend_r[g]   = L_jstart_r[g] + L_height_r[g] - 1;
      }

      /* now do the same for the actually expanded refinements                              */
      L_width_r_true[g] = n_r_true/Num_procs_rx[g];
      leftover =   n_r_true%Num_procs_rx[g];

      if (my_ID_rx[g]<leftover) {
        L_istart_r_true[g] = (L_width_r_true[g]+1) * my_ID_rx[g]; 
        L_iend_r_true[g]   = L_istart_r_true[g] + L_width_r_true[g];
      }
      else {
        L_istart_r_true[g] = (L_width_r_true[g]+1) * leftover + L_width_r_true[g] * (my_ID_rx[g]-leftover);
        L_iend_r_true[g]   = L_istart_r_true[g] + L_width_r_true[g] - 1;
      }
  
      L_height_r_true[g] = n_r_true/Num_procs_ry[g];
      leftover = n_r_true%Num_procs_ry[g];

      if (my_ID_ry[g]<leftover) {
        L_jstart_r_true[g] = (L_height_r_true[g]+1) * my_ID_ry[g]; 
        L_jend_r_true[g]   = L_jstart_r_true[g] + L_height_r_true[g];
      }
      else {
        L_jstart_r_true[g] = (L_height_r_true[g]+1) * leftover + L_height_r_true[g] * (my_ID_ry[g]-leftover);
        L_jend_r_true[g]   = L_jstart_r_true[g] + L_height_r_true[g] - 1;
      }

      /* shift refinement patch boundaries to BG coordinates                                */
      L_istart_r[g] += G_istart_r[g]; L_iend_r[g] += G_istart_r[g];
      L_jstart_r[g] += G_jstart_r[g]; L_jend_r[g] += G_jstart_r[g];
    }
    else if (load_balance == no_talk) { // already computed refinement partition boundaries
      L_istart_r_true[g] = (L_istart_r[g] - G_istart_r[g])*expand;
      if (my_ID_rx[g]>0) L_istart_r_true[g] -= expand/2;
      L_iend_r_true[g]   = (L_iend_r[g]   - G_istart_r[g])*expand;
      if (my_ID_rx[g] < Num_procs_rx[g]-1) L_iend_r_true[g] += (expand-1)/2;
      L_jstart_r_true[g] = (L_jstart_r[g] - G_jstart_r[g])*expand;
      if (my_ID_ry[g]>0) L_jstart_r_true[g] -= expand/2;
      L_jend_r_true[g]   = (L_jend_r[g]   - G_jstart_r[g])*expand;
      if (my_ID_ry[g] < Num_procs_ry[g]-1) L_jend_r_true[g] += (expand-1)/2;
    }

    /* make sure that the gross boundaries of the patch coincide with BG points           */
    L_istart_r_true_gross[g] = (L_istart_r_true[g]/expand)*expand;
    L_iend_r_true_gross[g]   = (L_iend_r_true[g]/expand+1)*expand;
    L_jstart_r_true_gross[g] = (L_jstart_r_true[g]/expand)*expand;
    L_jend_r_true_gross[g]   = (L_jend_r_true[g]/expand+1)*expand;
    L_istart_r_gross[g]      = L_istart_r_true_gross[g]/expand;
    L_iend_r_gross[g]        = L_iend_r_true_gross[g]/expand;
    L_jstart_r_gross[g]      = L_jstart_r_true_gross[g]/expand;
    L_jend_r_gross[g]        = L_jend_r_true_gross[g]/expand;

    /* shift unexpanded gross refinement patch boundaries to global BG coordinates        */
    L_istart_r_gross[g] += G_istart_r[g]; L_iend_r_gross[g] += G_istart_r[g];
    L_jstart_r_gross[g] += G_jstart_r[g]; L_jend_r_gross[g] += G_jstart_r[g];

    L_height_r[g]            = L_jend_r[g] -            L_jstart_r[g] + 1;
    L_width_r[g]             = L_iend_r[g] -            L_istart_r[g] + 1;
    L_height_r_true_gross[g] = L_jend_r_true_gross[g] - L_jstart_r_true_gross[g] + 1;
    L_width_r_true_gross[g]  = L_iend_r_true_gross[g] - L_istart_r_true_gross[g] + 1;
    L_height_r_true[g]       = L_jend_r_true[g] -       L_jstart_r_true[g] + 1;
    L_width_r_true[g]        = L_iend_r_true[g] -       L_istart_r_true[g] + 1;

    if (L_height_r_true[g] == 0 || L_width_r_true[g] == 0)  {
      printf("ERROR: rank %d has no work to do on refinement %d\n", my_ID, g);
      error = 1;
    }

    /* FIX THIS; don't want to bail out, just because a rank doesn't have a large
       enough refinement tile to work with. Can merge until tile is large enough */
    if (L_width_r_true[g] < RADIUS || L_height_r_true[g] < RADIUS) {
      printf("ERROR: rank %d's work tile %d smaller than stencil radius: %ld\n",
	     my_ID, g, MIN(L_width_r_true[g],L_height_r_true[g]));
      error = 1;
    }

    total_length_in_r[g]  = (L_width_r_true_gross[g]+2*RADIUS)*
                            (L_height_r_true_gross[g]+2*RADIUS);
    total_length_out_r[g] = L_width_r_true_gross[g] * L_height_r_true_gross[g];
    in_r[g]  = (DTYPE *) prk_malloc(sizeof(DTYPE)*total_length_in_r[g]);
    out_r[g] = (DTYPE *) prk_malloc(sizeof(DTYPE)*total_length_out_r[g]);
    if (!in_r[g] || !out_r[g]) {
      printf("ERROR: could not allocate space for refinement input or output arrays\n");
      error=1;
    }
  }
  else {//Bogus patch
    L_istart_r_gross[g] =  0;
    L_iend_r_gross[g]   = -1;
    L_jstart_r_gross[g] =  0;
    L_jend_r_gross[g]   = -1;
  }
  bail_out(error);

  /* fill the stencil weights to reflect a discrete divergence operator     */
  for (jj=-RADIUS; jj<=RADIUS; jj++) for (ii=-RADIUS; ii<=RADIUS; ii++) 
    WEIGHT(ii,jj) = (DTYPE) 0.0;

  stencil_size = 4*RADIUS+1;
  for (ii=1; ii<=RADIUS; ii++) {
    WEIGHT(0, ii) = WEIGHT( ii,0) =  (DTYPE) (1.0/(2.0*ii*RADIUS));
    WEIGHT(0,-ii) = WEIGHT(-ii,0) = -(DTYPE) (1.0/(2.0*ii*RADIUS));
  }

  /* weights for the refinement have to be scaled with the mesh spacing   */
  for (jj=-RADIUS; jj<=RADIUS; jj++) for (ii=-RADIUS; ii<=RADIUS; ii++)
    WEIGHT_R(ii,jj) = WEIGHT(ii,jj)*(DTYPE)expand;
  
  f_active_points_bg = (DTYPE) (n-2*RADIUS)*(DTYPE) (n-2*RADIUS);
  f_active_points_r  = (DTYPE) (n_r_true-2*RADIUS)*(DTYPE) (n_r_true-2*RADIUS);

  /* intialize the input and output arrays                                     */
  if (comm_bg != MPI_COMM_NULL)
  for (j=L_jstart_bg; j<=L_jend_bg; j++) for (i=L_istart_bg; i<=L_iend_bg; i++) {
    IN(i,j)  = COEFX*i+COEFY*j;
    OUT(i,j) = (DTYPE)0.0;
  }

  if (comm_bg != MPI_COMM_NULL) {
    /* allocate communication buffers for halo values                          */
    top_buf_out_bg = (DTYPE *) prk_malloc(4*sizeof(DTYPE)*RADIUS*L_width_bg);
    if (!top_buf_out_bg) {
      printf("ERROR: Rank %d could not allocate comm buffers for y-direction\n", my_ID);
      error = 1;
    } 
    top_buf_in_bg     = top_buf_out_bg +   RADIUS*L_width_bg;
    bottom_buf_out_bg = top_buf_out_bg + 2*RADIUS*L_width_bg;
    bottom_buf_in_bg  = top_buf_out_bg + 3*RADIUS*L_width_bg;

    /* add 1 on each side of the ghost point buffers for communication in the
       horizontal direction, to enable the NO_TALK scenario. See implementation details */
    right_buf_out_bg  = (DTYPE *) prk_malloc(4*sizeof(DTYPE)*RADIUS*(L_height_bg+2));
    if (!right_buf_out_bg) {
      printf("ERROR: Rank %d could not allocate comm buffers for x-direction\n", my_ID);
      error = 1;
    }
    right_buf_in_bg   = right_buf_out_bg +   RADIUS*(L_height_bg+2);
    left_buf_out_bg   = right_buf_out_bg + 2*RADIUS*(L_height_bg+2);
    left_buf_in_bg    = right_buf_out_bg + 3*RADIUS*(L_height_bg+2);
  }
  bail_out(error);

  /* intialize the refinement arrays                                           */
  for (g=0; g<4; g++) if (comm_r[g] != MPI_COMM_NULL) {
    for (j=L_jstart_r_true[g]; j<=L_jend_r_true[g]; j++) 
    for (i=L_istart_r_true[g]; i<=L_iend_r_true[g]; i++) {
      IN_R(g,i,j)  = (DTYPE)0.0;
      OUT_R(g,i,j) = (DTYPE)0.0;
    }
  }

  for (g=0; g<4; g++) if (comm_r[g] != MPI_COMM_NULL) {
    /* allocate communication buffers for halo values                          */
    top_buf_out_r[g] = (DTYPE *) prk_malloc(4*sizeof(DTYPE)*RADIUS*L_width_r_true[g]);
    if (!top_buf_out_r[g]) {
      printf("ERROR: Rank %d could not allocate comm buffers for y-direction for r=%d\n", 
             my_ID, g);
      error = 1;
    }
    top_buf_in_r[g]     = top_buf_out_r[g] +   RADIUS*L_width_r_true[g];
    bottom_buf_out_r[g] = top_buf_out_r[g] + 2*RADIUS*L_width_r_true[g];
    bottom_buf_in_r[g]  = top_buf_out_r[g] + 3*RADIUS*L_width_r_true[g];

    right_buf_out_r[g]  = (DTYPE *) prk_malloc(4*sizeof(DTYPE)*RADIUS*L_height_r_true[g]);
    if (!right_buf_out_r[g]) {
      printf("ERROR: Rank %d could not allocate comm buffers for x-direction for r=%d\n", my_ID, g);
      error = 1;
    }
    right_buf_in_r[g]   = right_buf_out_r[g] +   RADIUS*L_height_r_true[g];
    left_buf_out_r[g]   = right_buf_out_r[g] + 2*RADIUS*L_height_r_true[g];
    left_buf_in_r[g]    = right_buf_out_r[g] + 3*RADIUS*L_height_r_true[g];
  }
  bail_out(error);

  local_stencil_time = 0.0; /* silence compiler warning */

  num_interpolations = 0;
  
  for (iter = 0; iter<=iterations; iter++){

    /* start timer after a warmup iteration */
    if (iter == 1) {
      MPI_Barrier(MPI_COMM_WORLD);
      local_stencil_time = wtime();
    }

     time_step(Num_procs, Num_procs_bg, Num_procs_bgx, Num_procs_bgy,
	       Num_procs_r, Num_procs_rx, Num_procs_ry,
	       my_ID, my_ID_bg, my_ID_bgx, my_ID_bgy, my_ID_r, my_ID_rx, my_ID_ry,
	       right_nbr_bg, left_nbr_bg, top_nbr_bg, bottom_nbr_bg,
	       right_nbr_r, left_nbr_r, top_nbr_r, bottom_nbr_r,
	       top_buf_out_bg, top_buf_in_bg, bottom_buf_out_bg, bottom_buf_in_bg,
	       right_buf_out_bg, right_buf_in_bg, left_buf_out_bg, left_buf_in_bg,
	       top_buf_out_r, top_buf_in_r, bottom_buf_out_r, bottom_buf_in_r,
	       right_buf_out_r, right_buf_in_r, left_buf_out_r, left_buf_in_r,
	       n, refine_level, G_istart_r, G_iend_r, G_jstart_r, G_jend_r,
	       L_istart_bg, L_iend_bg, L_jstart_bg, L_jend_bg, L_width_bg, L_height_bg,
	       L_istart_r, L_iend_r, L_jstart_r, L_jend_r,
	       L_istart_r_gross, L_iend_r_gross, L_jstart_r_gross, L_jend_r_gross,
	       L_istart_r_true_gross, L_iend_r_true_gross,
	       L_jstart_r_true_gross, L_jend_r_true_gross,
	       L_istart_r_true, L_iend_r_true, L_jstart_r_true, L_jend_r_true,
	       L_width_r, L_height_r, L_width_r_true_gross, L_height_r_true_gross, 
	       L_width_r_true, L_height_r_true,
	       n_r, n_r_true, expand, period, duration, sub_iterations, iter, h_r,
	       num_interpolations, in_bg, out_bg, in_r, out_r, weight, weight_r,
	       load_balance, request_bg, request_r, comm_r, comm_bg);


  } /* end of iterations                                                         */

  local_stencil_time = wtime() - local_stencil_time;
  MPI_Reduce(&local_stencil_time, &stencil_time, 1, MPI_DOUBLE, MPI_MAX, root,
             MPI_COMM_WORLD);

  /* compute normalized L1 solution norm on background grid                      */
  local_norm = (DTYPE) 0.0;
  if (comm_bg != MPI_COMM_NULL) 
  for (int j=MAX(L_jstart_bg,RADIUS); j<=MIN(n-RADIUS-1,L_jend_bg); j++) {
    for (int i=MAX(L_istart_bg,RADIUS); i<=MIN(n-RADIUS-1,L_iend_bg); i++) {
      local_norm += (DTYPE)ABS(OUT(i,j));
    }
  }

  MPI_Reduce(&local_norm, &norm, 1, MPI_DTYPE, MPI_SUM, root, MPI_COMM_WORLD);
  if (my_ID == root) norm /= f_active_points_bg;

  /* compute normalized L1 input field norm on background grid                   */
  local_norm_in = (DTYPE) 0.0;
  if (comm_bg != MPI_COMM_NULL) 
  for (j=L_jstart_bg; j<=L_jend_bg; j++) for (i=L_istart_bg; i<=L_iend_bg; i++) {
    local_norm_in += (DTYPE)ABS(IN(i,j));
  }
  MPI_Reduce(&local_norm_in, &norm_in, 1, MPI_DTYPE, MPI_SUM, root, MPI_COMM_WORLD);
  if (my_ID == root) norm_in /= n*n;
  
  for (g=0; g<4; g++) {
    local_norm_r[g] = local_norm_in_r[g] = (DTYPE) 0.0;
    /* compute normalized L1 solution norm on refinements                        */
    if (comm_r[g] != MPI_COMM_NULL)
    for (j=MAX(L_jstart_r_true[g],RADIUS); j<=MIN(n_r_true-RADIUS-1,L_jend_r_true[g]); j++) 
      for (i=MAX(L_istart_r_true[g],RADIUS); i<=MIN(n_r_true-RADIUS-1,L_iend_r_true[g]); i++) {
        local_norm_r[g] += (DTYPE)ABS(OUT_R(g,i,j));
    }
    MPI_Reduce(&local_norm_r[g], &norm_r[g], 1, MPI_DTYPE, MPI_SUM, root, MPI_COMM_WORLD);
    if (my_ID == root) norm_r[g] /= f_active_points_r;

    /* compute normalized L1 input field norms on refinements                    */
    if (comm_r[g] != MPI_COMM_NULL)
    for (j=L_jstart_r_true[g]; j<=L_jend_r_true[g]; j++) 
      for (i=L_istart_r_true[g]; i<=L_iend_r_true[g]; i++) {
	local_norm_in_r[g] += (DTYPE)ABS(IN_R(g,i,j)); 
    }
    MPI_Reduce(&local_norm_in_r[g], &norm_in_r[g], 1, MPI_DTYPE, MPI_SUM, root, MPI_COMM_WORLD);
    if (my_ID == root) norm_in_r[g] /=  n_r_true*n_r_true;
  }


  /*******************************************************************************
  ** Analyze and output results.
  ********************************************************************************/

  if (my_ID == root) {
    /* verify correctness of background grid solution and input field            */
    reference_norm = (DTYPE) (iterations+1) * (COEFX + COEFY);
    reference_norm_in = (COEFX+COEFY)*(DTYPE)((n-1)/2.0)+iterations+1;
    if (ABS(norm-reference_norm) > EPSILON) {
      printf("ERROR: L1 norm = "FSTR", Reference L1 norm = "FSTR"\n",
             norm, reference_norm);
      validate = 0;
    }
    else {
#if VERBOSE
      printf("SUCCESS: Reference L1 norm         = "FSTR", L1 norm         = "FSTR"\n", 
             reference_norm, norm);
#endif
    }
 
    if (ABS(norm_in-reference_norm_in) > EPSILON) {
      printf("ERROR: L1 input norm         = "FSTR", Reference L1 input norm = "FSTR"\n",
             norm_in, reference_norm_in);
      validate = 0;
    }
    else {
#if VERBOSE
      printf("SUCCESS: Reference L1 input norm   = "FSTR", L1 input norm   = "FSTR"\n", 
             reference_norm_in, norm_in);
#endif
    }
    
    /* verify correctness of refinement grid solutions and input fields          */
    full_cycles = ((iterations+1)/(period*4));
    leftover_iterations = (iterations+1)%(period*4);
    for (g=0; g<4; g++) {
      iterations_r[g] = sub_iterations*(full_cycles*duration+
                        MIN(MAX(0,leftover_iterations-g*period),duration));
      reference_norm_r[g] = (DTYPE) (iterations_r[g]) * (COEFX + COEFY);
      if (iterations_r[g]==0) {
        reference_norm_in_r[g] = 0;
      }
      else {
        bg_updates = (full_cycles*4 + g)*period;
        r_updates  = MIN(MAX(0,leftover_iterations-g*period),duration) *
                         sub_iterations;
        if (bg_updates > iterations) {
          /* if this refinement not active in last AMR cycle, it completed the
             previous one completely                                            */
          bg_updates -= 4*period;
          r_updates = sub_iterations*duration;
        }
        reference_norm_in_r[g] = 
          /* initial input field value at bottom left corner of refinement      */
          (COEFX*G_istart_r[g] + COEFY*G_jstart_r[g]) +
          /* variable part                                                      */
          (COEFX+COEFY)*(n_r-1)/2.0 +
          /* number of times unity was added to background grid input field 
             before interpolation onto this refinement                          */
          (DTYPE) bg_updates +
          /* number of actual updates on this refinement since interpolation    */
          (DTYPE) r_updates;
      }
 
      if (ABS(norm_r[g]-reference_norm_r[g]) > EPSILON) {
        printf("ERROR: L1 norm %d       = "FSTR", Reference L1 norm %d = "FSTR"\n",
               g, norm_r[g], g, reference_norm_r[g]);
        validate = 0;
      }
      else {
#if VERBOSE
        printf("SUCCESS: Reference L1 norm %d       = "FSTR", L1 norm         = "FSTR"\n", g,
               reference_norm_r[g], norm_r[g]);
#endif
      }

      if (ABS(norm_in_r[g]-reference_norm_in_r[g]) > EPSILON) {
        printf("ERROR: L1 input norm %d = "FSTR", Reference L1 input norm %d = "FSTR"\n",
               g, norm_in_r[g], g, reference_norm_in_r[g]);
        validate = 0;
      }
      else {
#if VERBOSE
        printf("SUCCESS: Reference L1 input norm %d = "FSTR", L1 input norm %d = "FSTR"\n", 
               g, reference_norm_in_r[g], g, norm_in_r[g]);
#endif
      }
    }
 
    if (!validate) {
      printf("Solution does not validate\n");
    }
    else {
      printf("Solution validates\n");
 
      flops = f_active_points_bg * iterations;
      /* subtract one untimed iteration from refinement 0                          */
      iterations_r[0]--;
      for (g=0; g<4; g++) flops += f_active_points_r * iterations_r[g];
      flops *= (DTYPE) (2*stencil_size+1);
      /* add interpolation flops, if applicable                                    */
      if (refine_level>0) {
        /* subtract one interpolation (not timed)                                  */
        num_interpolations--;
        flops += n_r_true*(num_interpolations)*3*(n_r_true+n_r);
      }
      avgtime = stencil_time/iterations;
      printf("Rate (MFlops/s): "FSTR"  Avg time (s): %lf\n",
             1.0E-06 * flops/stencil_time, avgtime);
    }
  }

  MPI_Finalize();
  return(MPI_SUCCESS);
}
