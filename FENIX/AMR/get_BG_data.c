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

/* before interpolating from the background grid, we need to gather that BG data
   from wherever it resides and copy it to the right locations of the refinement */
void get_BG_data(int load_balance, DTYPE *in_bg, DTYPE *ing_r, int my_ID, long expand,
                 int Num_procs, long L_width_bg, 
                 long L_istart_bg, long L_iend_bg, long L_jstart_bg, long L_jend_bg,
                 long L_istart_r, long L_iend_r, long L_jstart_r, long L_jend_r,
                 long G_istart_r, long G_jstart_r, MPI_Comm comm_bg, MPI_Comm comm_r,
                 long L_istart_r_gross, long L_iend_r_gross, 
                 long L_jstart_r_gross, long L_jend_r_gross, 
                 long L_width_r_true_gross, long L_istart_r_true_gross, long L_iend_r_true_gross,
                 long L_jstart_r_true_gross, long L_jend_r_true_gross, int g) {

  long send_vec[8], *recv_vec, offset, i, j, p, acc_send, acc_recv;
  int *recv_offset, *recv_count, *send_offset, *send_count;
  DTYPE *recv_buf, *send_buf;

  if (load_balance == no_talk) {
    /* in case of no_talk we just copy the in-rank data from BG to refinement     */
    if (comm_r != MPI_COMM_NULL) {
      for (j=L_jstart_r_gross; j<=L_jend_r_gross; j++) 
      for (i=L_istart_r_gross; i<=L_iend_r_gross; i++) {
	int ir = i-G_istart_r, jr = j-G_jstart_r;
	ING_R(ir*expand,jr*expand) = IN(i,j);
      }
    }
  }
  else {
    recv_vec    = (long *)  prk_malloc(sizeof(long)*Num_procs*8);
    recv_count  = (int *)   prk_malloc(sizeof(int)*Num_procs);
    recv_offset = (int *)   prk_malloc(sizeof(int)*Num_procs);
    send_count  = (int *)   prk_malloc(sizeof(int)*Num_procs);
    send_offset = (int *)   prk_malloc(sizeof(int)*Num_procs);
    if (!recv_vec || !recv_count || !recv_offset || !send_count || !send_offset){
      printf("ERROR: Could not allocate space for Allgather on rank %d\n", my_ID);
      MPI_Abort(MPI_COMM_WORLD, 66); // no graceful exit in timed code
    }

    /* ask all other ranks what chunk of BG they have, and what chunk of the 
       refinement (one of the two will be nil for high_water)                     */
    
    send_vec[0] = L_istart_bg;
    send_vec[1] = L_iend_bg;
    send_vec[2] = L_jstart_bg;
    send_vec[3] = L_jend_bg;
    
    send_vec[4] = L_istart_r_gross;
    send_vec[5] = L_iend_r_gross;
    send_vec[6] = L_jstart_r_gross;
    send_vec[7] = L_jend_r_gross;
    
    MPI_Allgather(send_vec, 8, MPI_LONG, recv_vec, 8, MPI_LONG, MPI_COMM_WORLD);

    acc_recv = 0;
    for (acc_recv=0,p=0; p<Num_procs; p++) {
      /* Compute intersection of calling rank's gross refinement patch with each remote
         BG chunk,  which is the data they need to receive                        */
      recv_vec[p*8+0] = MAX(recv_vec[p*8+0], L_istart_r_gross); 
      recv_vec[p*8+1] = MIN(recv_vec[p*8+1], L_iend_r_gross);
      recv_vec[p*8+2] = MAX(recv_vec[p*8+2], L_jstart_r_gross);
      recv_vec[p*8+3] = MIN(recv_vec[p*8+3], L_jend_r_gross);
      /* now they determine how much data they are going to receive from each rank*/
      recv_count[p] = MAX(0,(recv_vec[p*8+1]-recv_vec[p*8+0]+1)) *
                      MAX(0,(recv_vec[p*8+3]-recv_vec[p*8+2]+1));
      acc_recv += recv_count[p];
    }
    if (acc_recv) {
      recv_buf = (DTYPE *) prk_malloc(sizeof(DTYPE)*acc_recv);
      if (!recv_buf) {
        printf("ERROR: Could not allocate space for recv_buf on rank %d\n", my_ID);
        MPI_Abort(MPI_COMM_WORLD, 66); // no graceful exit in timed code
      }
    }
      
    for (acc_send=0,p=0; p<Num_procs; p++) {
      /* compute intersection of calling rank BG with each refinement chunk, which 
         is the data they need to send                                            */
      recv_vec[p*8+4] = MAX(recv_vec[p*8+4], L_istart_bg);
      recv_vec[p*8+5] = MIN(recv_vec[p*8+5], L_iend_bg);
      recv_vec[p*8+6] = MAX(recv_vec[p*8+6], L_jstart_bg);
      recv_vec[p*8+7] = MIN(recv_vec[p*8+7], L_jend_bg);
      /* now they determine how much data they are going to send to each rank     */
      send_count[p] = MAX(0,(recv_vec[p*8+5]-recv_vec[p*8+4]+1)) *
                      MAX(0,(recv_vec[p*8+7]-recv_vec[p*8+6]+1));
      acc_send += send_count[p]; 
    }
    if (acc_send) {
      send_buf    = (DTYPE *) prk_malloc(sizeof(DTYPE)*acc_send);
      if (!send_buf) {
        printf("ERROR: Could not allocate space for send_buf on rank %d\n", my_ID);
        MPI_Abort(MPI_COMM_WORLD, 66); // no graceful exit in timed code
      }
    }

    recv_offset[0] =  send_offset[0] = 0;
    for (p=1; p<Num_procs; p++) {
      recv_offset[p] = recv_offset[p-1]+recv_count[p-1];
      send_offset[p] = send_offset[p-1]+send_count[p-1];
    }
    /* fill send buffer with BG data to all other ranks who need it               */
    offset = 0;
    if (comm_bg != MPI_COMM_NULL) for (p=0; p<Num_procs; p++){
      if (recv_vec[p*8+4]<=recv_vec[p*8+5]) { //test for non-empty inner loop
        for (j=recv_vec[p*8+6]; j<=recv_vec[p*8+7]; j++) {
          for (i=recv_vec[p*8+4]; i<=recv_vec[p*8+5]; i++){
            send_buf[offset++] = IN(i,j);
          }
	}
      }
    }

    MPI_Alltoallv(send_buf, send_count, send_offset, MPI_DTYPE, 
                  recv_buf, recv_count, recv_offset, MPI_DTYPE, MPI_COMM_WORLD);

    /* drain receive buffer with BG data from all other ranks who supplied it     */
    offset = 0;
    if (comm_r != MPI_COMM_NULL) for (p=0; p<Num_procs; p++) {
      if (recv_vec[p*8+0]<=recv_vec[p*8+1]) { //test for non-empty inner loop
        for (j=recv_vec[p*8+2]-G_jstart_r; j<=recv_vec[p*8+3]-G_jstart_r; j++) {
          for (i=recv_vec[p*8+0]-G_istart_r; i<=recv_vec[p*8+1]-G_istart_r; i++) {
            ING_R(i*expand,j*expand) = recv_buf[offset++];
	  }
	}
      }
    }

    prk_free(recv_vec);
    prk_free(recv_count);
    prk_free(recv_offset);
    prk_free(send_count);
    prk_free(send_offset);
    if (acc_recv) prk_free(recv_buf);
    if (acc_send) prk_free(send_buf);
  }
}

