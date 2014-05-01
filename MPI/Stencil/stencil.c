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

NAME:    Stencil

PURPOSE: This program tests the efficiency with which a space-invariant,
         linear, symmetric filter (stencil) can be applied to a square
         grid or image.
  
USAGE:   The program takes as input the linear dimension of the grid, the
         number of iterations on the grid, and optionally the tile size

               <progname> <# iterations> <grid size> [<tile size>]
  
         The output consists of diagnostics to make sure the 
         algorithm worked, and of timing statistics.

FUNCTIONS CALLED:

         Other than MPI or standard C functions, the following 
         functions are used in this program:

         wtime()
         bail_out()

HISTORY: - Written by Rob Van der Wijngaart, November 2006.
           RvdW: , August 2013
         - RvdW: Removed unrolling pragmas for clarity;
           fixed bug in compuation of width of strip assigned to 
           each rank;
           added constant to array "in" at end of each iteration to force 
           refreshing of neighbor data in parallel versions; August 2013
  
*********************************************************************************/

#include <par-res-kern_general.h>
#include <par-res-kern_mpi.h>

#ifndef RADIUS
  #define RADIUS 2
#endif

#ifdef DOUBLE
  #define DTYPE     double
  #define MPI_DTYPE MPI_DOUBLE
  #define EPSILON   1.e-8
  #define COEFX     1.0
  #define COEFY     1.0
  #define FSTR      "%lf"
#else
  #define DTYPE     float
  #define MPI_DTYPE MPI_FLOAT
  #define EPSILON   0.0001f
  #define COEFX     1.0f
  #define COEFY     1.0f
  #define FSTR      "%f"
#endif

/* define shorthand for indexing multi-dimensional arrays with offsets           */
#define INDEXIN(i,j)  (i+(j+RADIUS)*(n))
/* need to add offset of RADIUS to j to account for ghost points                 */
#define IN(i,j)       in[INDEXIN(i,j)]
#define INDEXOUT(i,j) (i+(j)*(n))
#define OUT(i,j)      out[INDEXOUT(i,j)]
#define WEIGHT(ii,jj) weight[ii+RADIUS][jj+RADIUS]

int main(int argc, char ** argv) {

  int    Num_procs;       /* number of processes                                 */
  int    my_ID;           /* MPI rank                                            */
  int    root = 0;
  int    n, nloc;         /* linear global and local grid dimension              */
  int    i, j, ii, jj, it, jt, iter, leftover;  /* dummies                       */
  int    jlow, jup;       /* bounds of grid strip assigned to calling process    */
  DTYPE  norm,            /* L1 norm of solution                                 */
         local_norm,      /* contribution of calling process to L1 norm          */
         reference_norm;
  DTYPE  f_active_points; /* interior of grid with respect to stencil            */
  DTYPE  flops;           /* floating point ops per iteration                    */
  int    iterations;      /* number of times to run the algorithm                */
  double stencil_time,    /* timing parameters                                   */
         avgtime = 0.0, 
         maxtime = 0.0, 
         mintime = 366.0*24.0*3600.0; /* set the minimum time to a large 
                             value; one leap year should be enough               */
  int    stencil_size;    /* number of points in stencil                         */
  int    tile_size;       /* grid block factor                                   */
  DTYPE  * RESTRICT in;   /* input grid values                                   */
  DTYPE  * RESTRICT out;  /* output grid values                                  */
  int    total_length_in; /* total required length to store input array          */
  int    total_length_out;/* total required length to store output array         */
  int    error=0;         /* error flag                                          */
  DTYPE  weight[2*RADIUS+1][2*RADIUS+1]; /* weights of points in the stencil     */
  MPI_Request request[4];
  MPI_Status  status[4];

  /*******************************************************************************
  ** Initialize the MPI environment
  ********************************************************************************/
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_ID);
  MPI_Comm_size(MPI_COMM_WORLD, &Num_procs);

  /*******************************************************************************
  ** process, test, and broadcast input parameters    
  ********************************************************************************/

  if (my_ID == root) {
    if (argc != 3 && argc != 4){
      printf("Usage: %s <# iterations> <array dimension> <tile size>\n", 
             *argv);
      error = 1;
      goto ENDOFTESTS;
    }

    iterations  = atoi(*++argv); 
    if (iterations < 1){
      printf("ERROR: iterations must be >= 1 : %d \n",iterations);
      error = 1;
      goto ENDOFTESTS;  
    }

    n  = atoi(*++argv);
    if (n < Num_procs){
      printf("ERROR: grid dimension must be at least # processes: %d\n", n);
      error = 1;
      goto ENDOFTESTS;
    }

    if (RADIUS < 0) {
      printf("ERROR: Stencil radius %d should be non-negative\n", RADIUS);
      error = 1;
      goto ENDOFTESTS;  
    }

    if (2*RADIUS +1 > n) {
      printf("ERROR: Stencil radius %d exceeds grid size %d\n", RADIUS, n);
      error = 1;
      goto ENDOFTESTS;  
    }

    if (argc == 4) {
      tile_size = atoi(*++argv);
      if (tile_size < 1) {
        printf("ERROR: tile size must be positive : %d\n", tile_size);
        error = 1;
        goto ENDOFTESTS;
      }
    }
    else tile_size = n;

    ENDOFTESTS:;  
  }
  bail_out(error);

  if (my_ID == root) {
    printf("MPI stencil execution on 2D grid\n");
    printf("Number of processes  = %d\n", Num_procs);
    printf("Grid size            = %d\n", n);
    printf("Radius of stencil    = %d\n", RADIUS);
    if (tile_size <n-2*RADIUS) 
      printf("Tile size            = %d\n", tile_size);
    else
      printf("Grid not tiled\n");
#ifdef STAR
    printf("Type of stencil      = star\n");
#else
    printf("Type of stencil      = compact\n");
#endif
#ifdef DOUBLE
    printf("Data type            = double precision\n");
#else
    printf("Data type            = single precision\n");
#endif
    printf("Number of iterations = %d\n", iterations);
  }

  MPI_Bcast(&n,          1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&tile_size,  1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&iterations, 1, MPI_INT, root, MPI_COMM_WORLD);

  /* compute amount of space required for input and solution arrays             */
  nloc = n/Num_procs;
  leftover = n%Num_procs;
  if (my_ID<leftover) {
    jlow = (nloc+1) * my_ID; 
    jup = jlow + nloc + 1;
  }
  else {
    jlow = (nloc+1) * leftover + nloc * (my_ID-leftover);
    jup = jlow + nloc;
  }
  
  nloc = jup - jlow;
  if (nloc == 0) {
    printf("ERROR: Process %d has no work to do\n", my_ID);
    error = 1;
  }
  bail_out(error);

  if (nloc < RADIUS) {
    printf("ERROR: Process %d has work strip smaller then stencil radius\n",
	   my_ID);
    error = 1;
  }
  bail_out(error);

  total_length_in = (nloc+2*RADIUS)*n*sizeof(DTYPE);
  if (total_length_in/(nloc+2*RADIUS) != n*sizeof(DTYPE)) {
    printf("ERROR: Space for %d x %d input array cannot be represented\n", 
           nloc+2*RADIUS, n);
    error = 1;
  }
  bail_out(error);

  total_length_out = nloc*n*sizeof(DTYPE);

  in  = (DTYPE *) malloc(total_length_in);
  out = (DTYPE *) malloc(total_length_out);
  if (!in || !out) {
    printf("ERROR: process %d could not allocate space for input/output array\n",
            my_ID);
    error = 1;
  }
  bail_out(error);

  /* fill the stencil weights to reflect a discrete divergence operator         */
  for (jj=-RADIUS; jj<=RADIUS; jj++) for (ii=-RADIUS; ii<=RADIUS; ii++)
    WEIGHT(ii,jj) = (DTYPE) 0.0;
#ifdef STAR
  stencil_size = 4*RADIUS+1;
  for (ii=1; ii<=RADIUS; ii++) {
    WEIGHT(0, ii) = WEIGHT( ii,0) =  (DTYPE) (1.0/(2.0*ii*RADIUS));
    WEIGHT(0,-ii) = WEIGHT(-ii,0) = -(DTYPE) (1.0/(2.0*ii*RADIUS));
  }
#else
  stencil_size = (2*RADIUS+1)*(2*RADIUS+1);
  for (jj=1; jj<=RADIUS; jj++) {
    for (ii=-jj+1; ii<jj; ii++) {
      WEIGHT(ii,jj)  =  (DTYPE) (1.0/(4.0*jj*(2.0*jj-1)*RADIUS));
      WEIGHT(ii,-jj) = -(DTYPE) (1.0/(4.0*jj*(2.0*jj-1)*RADIUS));
      WEIGHT(jj,ii)  =  (DTYPE) (1.0/(4.0*jj*(2.0*jj-1)*RADIUS));
      WEIGHT(-jj,ii) = -(DTYPE) (1.0/(4.0*jj*(2.0*jj-1)*RADIUS));      
    }
    WEIGHT(jj,jj)    =  (DTYPE) (1.0/(4.0*jj*RADIUS));
    WEIGHT(-jj,-jj)  = -(DTYPE) (1.0/(4.0*jj*RADIUS));
  }
#endif  

  norm = (DTYPE) 0.0;
  f_active_points = (DTYPE) (n-2*RADIUS)*(DTYPE) (n-2*RADIUS);
  /* intialize the input and output arrays                                     */
  for (j=jlow; j<jup; j++) for (i=0; i<n; i++) {
    IN(i,j-jlow)  = COEFX*i+COEFY*j;
    OUT(i,j-jlow) = (DTYPE)0.0;
  }

  for (iter = 0; iter<iterations; iter++){

    MPI_Barrier(MPI_COMM_WORLD);
    stencil_time = wtime();

    /* need to fetch ghost point data from neighbors                           */
    if (my_ID < Num_procs-1) {
      MPI_Isend(&(IN(0,jup-jlow-RADIUS)),RADIUS*n,MPI_DTYPE, my_ID+1, 99, 
                MPI_COMM_WORLD, &(request[0]));
      MPI_Irecv(&(IN(0,jup-jlow)), RADIUS*n, MPI_DTYPE, my_ID+1, 101,
                MPI_COMM_WORLD, &(request[1]));
    }
    if (my_ID > 0) {
      MPI_Isend(&(IN(0,0)), RADIUS*n,MPI_DTYPE, my_ID-1, 101,
                MPI_COMM_WORLD, &(request[2]));
      MPI_Irecv(&(IN(0,-RADIUS)),RADIUS*n, MPI_DTYPE, my_ID-1, 99, 
                MPI_COMM_WORLD, &(request[3]));
    }
    if (my_ID < Num_procs-1) {
      MPI_Wait(&(request[0]), &(status[0]));
      MPI_Wait(&(request[1]), &(status[1]));
    }
    if (my_ID > 0) {
      MPI_Wait(&(request[2]), &(status[2]));
      MPI_Wait(&(request[3]), &(status[3]));
    }

    /* Apply the stencil operator; only use tiling if the tile size is smaller
       than the iterior part of the grid                                       */
    if (tile_size < n-2*RADIUS) { 
      for (j=MAX(jlow,RADIUS); j<MIN(n-RADIUS,jup); j+=tile_size) {
        for (i=RADIUS; i<n-RADIUS; i+=tile_size) {
          for (jt=j; jt<MIN(jup,j+tile_size); jt++) {
            for (it=i; it<MIN(n-RADIUS,i+tile_size); it++) { 
#ifdef STAR
              for (jj=-RADIUS; jj<=RADIUS; jj++)  
                OUT(it,jt-jlow) += WEIGHT(0,jj)*IN(it,jt-jlow+jj);
              for (ii=-RADIUS; ii<0; ii++)        
                OUT(it,jt-jlow) += WEIGHT(ii,0)*IN(it+ii,jt-jlow);
              for (ii=1; ii<=RADIUS; ii++)        
                OUT(it,jt-jlow) += WEIGHT(ii,0)*IN(it+ii,jt-jlow);
#else
              /* would like to be able to unroll this loop, but compiler will ignore  */
              for (jj=-RADIUS; jj<=RADIUS; jj++) 
              for (ii=-RADIUS; ii<=RADIUS; ii++) {
                OUT(it,jt-jlow) += WEIGHT(ii,jj)*IN(it+ii,jt-jlow+jj);
              }
#endif
            }
          }
        }
      }
    }
    else {
      for (j=MAX(jlow,RADIUS); j<MIN(n-RADIUS,jup); j++) {
        for (i=RADIUS; i<n-RADIUS; i++) {
#ifdef STAR
          for (jj=-RADIUS; jj<=RADIUS; jj++)  
            OUT(i,j-jlow) += WEIGHT(0,jj)*IN(i,j-jlow+jj);
          for (ii=-RADIUS; ii<0; ii++)        
            OUT(i,j-jlow) += WEIGHT(ii,0)*IN(i+ii,j-jlow);
          for (ii=1; ii<=RADIUS; ii++)        
            OUT(i,j-jlow) += WEIGHT(ii,0)*IN(i+ii,j-jlow);
#else
          /* would like to be able to unroll this loop, but compiler will ignore  */
          for (jj=-RADIUS; jj<=RADIUS; jj++) 
          for (ii=-RADIUS; ii<=RADIUS; ii++) {
            OUT(i,j-jlow) += WEIGHT(ii,jj)*IN(i+ii,j-jlow+jj);
          }
#endif
        }
      }
    }
    stencil_time = wtime() - stencil_time;
    if (iter>0 || iterations==1) { /* skip the first iteration                   */
      avgtime = avgtime + stencil_time;
      mintime = MIN(mintime, stencil_time);
      maxtime = MAX(maxtime, stencil_time);
    }
    /* add constant to solution to force refresh of neighbor data, if any         */
    for (j=jlow; j<jup; j++) for (i=0; i<n; i++) IN(i,j-jlow)+= 1.0;

  }
  
  /* compute L1 norm in parallel                                                */
  local_norm = (DTYPE) 0.0;
  for (j=MAX(jlow,RADIUS); j<MIN(n-RADIUS,jup); j++) {
    for (i=RADIUS; i<n-RADIUS; i++) {
      local_norm += (DTYPE)ABS(OUT(i,j-jlow));
    }
  }

  MPI_Reduce(&local_norm, &norm, 1, MPI_DTYPE, MPI_SUM, root, MPI_COMM_WORLD);

  /*******************************************************************************
  ** Analyze and output results.
  ********************************************************************************/

/* verify correctness                                                            */
  if (my_ID == root) {
    norm /= f_active_points;
    if (RADIUS > 0) {
      reference_norm = (DTYPE) iterations * (COEFX + COEFY);
    }
    else {
      reference_norm = (DTYPE) 0.0;
    }
    if (ABS(norm-reference_norm) > EPSILON) {
      printf("ERROR: L1 norm = "FSTR", Reference L1 norm = "FSTR"\n",
             norm, reference_norm);
      error = 1;
    }
    else {
      printf("Solution validates\n");
#ifdef VERBOSE
      printf("Reference L1 norm = "FSTR", L1 norm = "FSTR"\n", 
             reference_norm, norm);
#endif
    }
  }
  bail_out(error);

  if (my_ID == root) {
    flops = (DTYPE) (2*stencil_size-1) * f_active_points;
    avgtime = avgtime/(double)(MAX(iterations-1,1));
    printf("Rate (MFlops/s): "FSTR",  Avg time (s): %lf,  Min time (s): %lf",
           1.0E-06 * flops/mintime, avgtime, mintime);
    printf(", Max time (s): %lf\n", maxtime);
  }

  MPI_Finalize();
  exit(EXIT_SUCCESS);
}
