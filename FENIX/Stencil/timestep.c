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

/* define shorthand for indexing multi-dimensional arrays with offsets           */
#define INDEXIN(i,j)  (i+RADIUS+(j+RADIUS)*(width+2*RADIUS))
/* need to add offset of RADIUS to j to account for ghost points                 */
#include <signal.h>
#include <sys/types.h>
#include <par-res-kern_general.h>
#include <par-res-kern_fenix.h>
#include <random_draw.h>
#include <unistd.h>

#if DOUBLE
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
#define INDEXIN(i,j)  (i+RADIUS+(j+RADIUS)*(width+2*RADIUS))
/* need to add offset of RADIUS to j to account for ghost points                 */
#define IN(i,j)       in[INDEXIN(i-istart,j-jstart)]
#define INDEXOUT(i,j) (i+(j)*(width))
#define OUT(i,j)      out[INDEXOUT(i-istart,j-jstart)]
#define WEIGHT(ii,jj) weight[ii+RADIUS][jj+RADIUS]

void time_step(int    Num_procsx, int Num_procsy,
	       int    my_IDx, int my_IDy,
	       int    right_nbr,
	       int    left_nbr,
	       int    top_nbr,
	       int    bottom_nbr,
	       DTYPE *top_buf_out,
	       DTYPE *top_buf_in,
	       DTYPE *bottom_buf_out,
	       DTYPE *bottom_buf_in,
	       DTYPE *right_buf_out,
	       DTYPE *right_buf_in,
	       DTYPE *left_buf_out,
	       DTYPE *left_buf_in,
	       int    n, int width, int height,
	       int    istart, int iend,
	       int    jstart, int jend,
	       DTYPE  * RESTRICT in,
	       DTYPE  * RESTRICT out,
	       DTYPE  weight[2*RADIUS+1][2*RADIUS+1],
	       MPI_Request request[8])
{

  /* need to fetch ghost point data from neighbors in y-direction                 */
  if (my_IDy < Num_procsy-1) {
    MPI_Irecv(top_buf_in, RADIUS*width, MPI_DTYPE, top_nbr, 101,
              MPI_COMM_WORLD, &(request[1]));
    for (int kk=0,j=jend-RADIUS+1; j<=jend; j++) for (int i=istart; i<=iend; i++) {
        top_buf_out[kk++]= IN(i,j);
    }
    MPI_Isend(top_buf_out, RADIUS*width,MPI_DTYPE, top_nbr, 99,
              MPI_COMM_WORLD, &(request[0]));
  }
  if (my_IDy > 0) {
    MPI_Irecv(bottom_buf_in,RADIUS*width, MPI_DTYPE, bottom_nbr, 99,
              MPI_COMM_WORLD, &(request[3]));
    for (int kk=0,j=jstart; j<=jstart+RADIUS-1; j++) for (int i=istart; i<=iend; i++) {
        bottom_buf_out[kk++]= IN(i,j);
    }
    MPI_Isend(bottom_buf_out, RADIUS*width,MPI_DTYPE, bottom_nbr, 101,
              MPI_COMM_WORLD, &(request[2]));
  }
  if (my_IDy < Num_procsy-1) {
    MPI_Wait(&(request[0]), MPI_STATUS_IGNORE);
    MPI_Wait(&(request[1]), MPI_STATUS_IGNORE);
    for (int kk=0,j=jend+1; j<=jend+RADIUS; j++) for (int i=istart; i<=iend; i++) {
        IN(i,j) = top_buf_in[kk++];
    }
  }
  if (my_IDy > 0) {
    MPI_Wait(&(request[2]), MPI_STATUS_IGNORE);
    MPI_Wait(&(request[3]), MPI_STATUS_IGNORE);
    for (int kk=0,j=jstart-RADIUS; j<=jstart-1; j++) for (int i=istart; i<=iend; i++) {
        IN(i,j) = bottom_buf_in[kk++];
    }
  }

  /* need to fetch ghost point data from neighbors in x-direction                 */
  if (my_IDx < Num_procsx-1) {
    MPI_Irecv(right_buf_in, RADIUS*height, MPI_DTYPE, right_nbr, 1010,
              MPI_COMM_WORLD, &(request[1+4]));
    for (int kk=0,j=jstart; j<=jend; j++) for (int i=iend-RADIUS+1; i<=iend; i++) {
        right_buf_out[kk++]= IN(i,j);
    }
    MPI_Isend(right_buf_out, RADIUS*height, MPI_DTYPE, right_nbr, 990,
              MPI_COMM_WORLD, &(request[0+4]));
  }
  if (my_IDx > 0) {
    MPI_Irecv(left_buf_in, RADIUS*height, MPI_DTYPE, left_nbr, 990,
              MPI_COMM_WORLD, &(request[3+4]));
    for (int kk=0,j=jstart; j<=jend; j++) for (int i=istart; i<=istart+RADIUS-1; i++) {
        left_buf_out[kk++]= IN(i,j);
    }
    MPI_Isend(left_buf_out, RADIUS*height, MPI_DTYPE, left_nbr, 1010,
              MPI_COMM_WORLD, &(request[2+4]));
  }
  if (my_IDx < Num_procsx-1) {
    MPI_Wait(&(request[0+4]), MPI_STATUS_IGNORE);
    MPI_Wait(&(request[1+4]), MPI_STATUS_IGNORE);
    for (int kk=0,j=jstart; j<=jend; j++) for (int i=iend+1; i<=iend+RADIUS; i++) {
        IN(i,j) = right_buf_in[kk++];
    }
  }
  if (my_IDx > 0) {
    MPI_Wait(&(request[2+4]), MPI_STATUS_IGNORE);
    MPI_Wait(&(request[3+4]), MPI_STATUS_IGNORE);
    for (int kk=0,j=jstart; j<=jend; j++) for (int i=istart-RADIUS; i<=istart-1; i++) {
        IN(i,j) = left_buf_in[kk++];
    }
  }

  /* Apply the stencil operator */
  for (int j=MAX(jstart,RADIUS); j<=MIN(n-RADIUS-1,jend); j++) {
    for (int i=MAX(istart,RADIUS); i<=MIN(n-RADIUS-1,iend); i++) {
      #if LOOPGEN
        #include "loop_body_star.incl"
      #else
        for (int jj=-RADIUS; jj<=RADIUS; jj++) OUT(i,j) += WEIGHT(0,jj)*IN(i,j+jj);
        for (int ii=-RADIUS; ii<0; ii++)       OUT(i,j) += WEIGHT(ii,0)*IN(i+ii,j);
        for (int ii=1; ii<=RADIUS; ii++)       OUT(i,j) += WEIGHT(ii,0)*IN(i+ii,j);
      #endif
    }
  }

  /* add constant to solution to force refresh of neighbor data, if any */
  for (int j=jstart; j<=jend; j++) for (int i=istart; i<=iend; i++) IN(i,j)+= 1.0;
}
