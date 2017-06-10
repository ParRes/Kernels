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

#include <par-res-kern_general.h>
#include <par-res-kern_fenix.h>

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

/* use two-stage, bi-linear interpolation of BG values to refinement. BG values
   have already been copied to the refinement                                   */
void interpolate(DTYPE *ing_r, long L_width_r_true_gross,
                 long L_istart_r_true_gross, long L_iend_r_true_gross,
                 long L_jstart_r_true_gross, long L_jend_r_true_gross, 
                 long L_istart_r_true, long L_iend_r_true,
                 long L_jstart_r_true, long L_jend_r_true, 
                 long expand, DTYPE h_r, int g, int Num_procs, int my_ID) {

  long ir, jr, ib, jrb, jrb1, jb;
  DTYPE xr, xb, yr, yb;

  if (expand==1) return; /* nothing to do anymore                                  */

  /* First, interpolate in x-direction                                             */
  for (jr=L_jstart_r_true_gross; jr<=L_jend_r_true_gross; jr+=expand) {
    for (ir=L_istart_r_true_gross; ir<L_iend_r_true_gross; ir++) {
      xr = h_r*(DTYPE)ir;
      ib = (long)xr;
      xb = (DTYPE)ib;
      ING_R(ir,jr) = ING_R((ib+1)*expand,jr)*(xr-xb) +
	             ING_R(ib*expand,jr)*(xb+(DTYPE)1.0-xr);
    }
  }

  /* Next, interpolate in y-direction                                              */
  for (jr=L_jstart_r_true; jr<=L_jend_r_true; jr++) {
    yr = h_r*(DTYPE)jr;
    jb = (long)yr;
    jrb = jb*expand;
    jrb1 = (jb+1)*expand;
    yb = (DTYPE)jb;
    for (ir=L_istart_r_true; ir<=L_iend_r_true; ir++) {
      ING_R(ir,jr) = ING_R(ir,jrb1)*(yr-yb) + ING_R(ir,jrb)*(yb+(DTYPE)1.0-yr);
    }
    /* note that (yr-yb) and (yb+(DTYPE)1.0-yr) can be hoisted out of the loop,
       so in the performance computation we assign 3 flops per point               */
  }
}


void time_step(int    Num_procs,
	       int    Num_procs_bg,
	       int    Num_procs_bgx, int Num_procs_bgy,
	       int    Num_procs_r[4],
	       int    Num_procs_rx[4], int Num_procs_ry[4],
	       int    my_ID,
	       int    my_ID_bg,
	       int    my_ID_bgx, int my_ID_bgy,
	       int    my_ID_r[4],
	       int    my_ID_rx[4], int my_ID_ry[4],
	       int    right_nbr_bg,
	       int    left_nbr_bg,
	       int    top_nbr_bg,
	       int    bottom_nbr_bg,
	       int    right_nbr_r[4],
	       int    left_nbr_r[4],
	       int    top_nbr_r[4],
	       int    bottom_nbr_r[4],
	       DTYPE  *top_buf_out_bg,
	       DTYPE  *top_buf_in_bg,
	       DTYPE  *bottom_buf_out_bg,
	       DTYPE  *bottom_buf_in_bg,
	       DTYPE  *right_buf_out_bg,
	       DTYPE  *right_buf_in_bg,
	       DTYPE  *left_buf_out_bg,
	       DTYPE  *left_buf_in_bg,
	       DTYPE  *top_buf_out_r[4],
	       DTYPE  *top_buf_in_r[4],
	       DTYPE  *bottom_buf_out_r[4],
	       DTYPE  *bottom_buf_in_r[4],
	       DTYPE  *right_buf_out_r[4],
	       DTYPE  *right_buf_in_r[4],
	       DTYPE  *left_buf_out_r[4],
	       DTYPE  *left_buf_in_r[4],
	       long   n,
	       int    refine_level,
	       long   G_istart_r[4],
	       long   G_iend_r[4],
	       long   G_jstart_r[4],
	       long   G_jend_r[4],
	       long   L_istart_bg, long L_iend_bg,
	       long   L_jstart_bg, long L_jend_bg,
	       long   L_width_bg, long L_height_bg,
	       long   L_istart_r[4], long L_iend_r[4],
	       long   L_jstart_r[4], long L_jend_r[4],
	       long   L_istart_r_gross[4], long L_iend_r_gross[4],
	       long   L_jstart_r_gross[4], long L_jend_r_gross[4],
	       long   L_istart_r_true_gross[4], long L_iend_r_true_gross[4],
	       long   L_jstart_r_true_gross[4], long L_jend_r_true_gross[4],
	       long   L_istart_r_true[4], long L_iend_r_true[4],
	       long   L_jstart_r_true[4], long L_jend_r_true[4],
	       long   L_width_r[4], long L_height_r[4],
	       long   L_width_r_true_gross[4], long L_height_r_true_gross[4], 
	       long   L_width_r_true[4], long L_height_r_true[4],
	       long   n_r,
	       long   n_r_true,
	       long   expand,
	       int    period,
	       int    duration,
	       int    sub_iterations, 
               int    iter,
	       DTYPE  h_r,
               int    num_interpolations,
	       DTYPE  * RESTRICT in_bg,
	       DTYPE  * RESTRICT out_bg,
	       DTYPE  * RESTRICT in_r[4],
	       DTYPE  * RESTRICT out_r[4],
	       DTYPE  weight[2*RADIUS+1][2*RADIUS+1],
	       DTYPE  weight_r[2*RADIUS+1][2*RADIUS+1],
               int    load_balance,
	       MPI_Request request_bg[8],
	       MPI_Request request_r[4][8],
	       MPI_Comm comm_r[4],
	       MPI_Comm comm_bg,
               int    first_through) {

  int g, i, j, ii, jj, kk, sub_iter;

  /* first complete communication on background grid to help no_talk balancer     */
  if (comm_bg != MPI_COMM_NULL) {
    /* need to fetch ghost point data from neighbors in y-direction                 */
    if (my_ID_bgy < Num_procs_bgy-1) {
      MPI_Irecv(top_buf_in_bg, RADIUS*L_width_bg, MPI_DTYPE, top_nbr_bg, 101,
                comm_bg, &(request_bg[1]));
      for (int kk=0,j=L_jend_bg-RADIUS+1; j<=L_jend_bg; j++) 
      for (int i=L_istart_bg; i<=L_iend_bg; i++) {
          top_buf_out_bg[kk++]= IN(i,j);
      }
      MPI_Isend(top_buf_out_bg, RADIUS*L_width_bg,MPI_DTYPE, top_nbr_bg, 99,
                comm_bg, &(request_bg[0]));
    }
    if (my_ID_bgy > 0) {
      MPI_Irecv(bottom_buf_in_bg,RADIUS*L_width_bg, MPI_DTYPE, bottom_nbr_bg, 99,
                comm_bg, &(request_bg[3]));
      for (int kk=0,j=L_jstart_bg; j<=L_jstart_bg+RADIUS-1; j++) 
      for (int i=L_istart_bg; i<=L_iend_bg; i++) {
            bottom_buf_out_bg[kk++]= IN(i,j);
      }
      MPI_Isend(bottom_buf_out_bg, RADIUS*L_width_bg,MPI_DTYPE, bottom_nbr_bg, 101,
                comm_bg, &(request_bg[2]));
    }
    if (my_ID_bgy < Num_procs_bgy-1) {
      MPI_Wait(&(request_bg[0]), MPI_STATUS_IGNORE);
      MPI_Wait(&(request_bg[1]), MPI_STATUS_IGNORE);
      for (int kk=0,j=L_jend_bg+1; j<=L_jend_bg+RADIUS; j++) 
      for (int i=L_istart_bg; i<=L_iend_bg; i++) {
          IN(i,j) = top_buf_in_bg[kk++];
      }
    }

    if (my_ID_bgy > 0) {
      MPI_Wait(&(request_bg[2]), MPI_STATUS_IGNORE);
      MPI_Wait(&(request_bg[3]), MPI_STATUS_IGNORE);
      for (int kk=0,j=L_jstart_bg-RADIUS; j<=L_jstart_bg-1; j++) 
      for (int i=L_istart_bg; i<=L_iend_bg; i++) {
          IN(i,j) = bottom_buf_in_bg[kk++];
      }
    }

    /* need to fetch ghost point data from neighbors in x-direction; take into account
       the load balancer; NO_TALK needs wider copy                                    */
    if (my_ID_bgx < Num_procs_bgx-1) {
      MPI_Irecv(right_buf_in_bg, RADIUS*(L_height_bg+2), MPI_DTYPE, right_nbr_bg, 1010,
                comm_bg, &(request_bg[1+4]));
      for (int kk=0,j=L_jstart_bg-1; j<=L_jend_bg+1; j++) 
      for (int i=L_iend_bg-RADIUS+1; i<=L_iend_bg; i++) {
          right_buf_out_bg[kk++]= IN(i,j);
      }
      MPI_Isend(right_buf_out_bg, RADIUS*(L_height_bg+2), MPI_DTYPE, right_nbr_bg, 990,
              comm_bg, &(request_bg[0+4]));
    }
    if (my_ID_bgx > 0) {
      MPI_Irecv(left_buf_in_bg, RADIUS*(L_height_bg+2), MPI_DTYPE, left_nbr_bg, 990,
                comm_bg, &(request_bg[3+4]));
      for (int kk=0,j=L_jstart_bg-1; j<=L_jend_bg+1; j++) 
      for (int i=L_istart_bg; i<=L_istart_bg+RADIUS-1; i++) {
        left_buf_out_bg[kk++]= IN(i,j);
      }
      MPI_Isend(left_buf_out_bg, RADIUS*(L_height_bg+2), MPI_DTYPE, left_nbr_bg, 1010,
                comm_bg, &(request_bg[2+4]));
    }
    if (my_ID_bgx < Num_procs_bgx-1) {
      MPI_Wait(&(request_bg[0+4]), MPI_STATUS_IGNORE);
      MPI_Wait(&(request_bg[1+4]), MPI_STATUS_IGNORE);
      for (int kk=0,j=L_jstart_bg-1; j<=L_jend_bg+1; j++) 
      for (int i=L_iend_bg+1; i<=L_iend_bg+RADIUS; i++) {
          IN(i,j) = right_buf_in_bg[kk++];
      }
    }
  
    if (my_ID_bgx > 0) {
      MPI_Wait(&(request_bg[2+4]), MPI_STATUS_IGNORE);
      MPI_Wait(&(request_bg[3+4]), MPI_STATUS_IGNORE);
      for (int kk=0,j=L_jstart_bg-1; j<=L_jend_bg+1; j++) 
      for (int i=L_istart_bg-RADIUS; i<=L_istart_bg-1; i++) {
          IN(i,j) = left_buf_in_bg[kk++];
      }
    }
  }

  if (!(iter%period) || first_through) {
    /* a specific refinement has come to life                                */
    g=(iter/period)%4; first_through=0;

    get_BG_data(load_balance, in_bg, in_r[g], my_ID, expand, Num_procs,
                L_width_bg, L_istart_bg, L_iend_bg, L_jstart_bg, L_jend_bg,
                L_istart_r[g], L_iend_r[g], L_jstart_r[g], L_jend_r[g],
                G_istart_r[g], G_jstart_r[g], comm_bg, comm_r[g],
                L_istart_r_gross[g], L_iend_r_gross[g], 
                L_jstart_r_gross[g], L_jend_r_gross[g], 
                L_width_r_true_gross[g], L_istart_r_true_gross[g], L_iend_r_true_gross[g],
                L_jstart_r_true_gross[g], L_jend_r_true_gross[g], g);
      
    if (comm_r[g] != MPI_COMM_NULL) {
      interpolate(in_r[g], L_width_r_true_gross[g], 
                  L_istart_r_true_gross[g], L_iend_r_true_gross[g],
                  L_jstart_r_true_gross[g], L_jend_r_true_gross[g], 
                  L_istart_r_true[g], L_iend_r_true[g],
                  L_jstart_r_true[g], L_jend_r_true[g], 
                  expand, h_r, g, Num_procs, my_ID);
    }
    /* even though this rank may not interpolate, some just did, so we keep track   */
    num_interpolations++;

  } // end of initialization of refinement g

  /* if we have recovered from a failure and we aren't at a RG activation point,
     RG index g must be set to the same value as that of the survivor ranks         */
  g=(iter/period)%4;

  if (comm_r[g] != MPI_COMM_NULL) if ((iter%period) < duration) {

    /* if within an active refinement epoch, first communicate within refinement    */

    for (sub_iter=0; sub_iter<sub_iterations; sub_iter++) {
      /* need to communicate within each sub-iteration                              */
      /* need to fetch ghost point data from neighbors in y-direction               */
      if (top_nbr_r[g] != -1) {
        MPI_Irecv(top_buf_in_r[g], RADIUS*L_width_r_true[g], MPI_DTYPE, top_nbr_r[g], 
                  101, comm_r[g], &(request_r[g][1]));
        for (int kk=0,j=L_jend_r_true[g]-RADIUS+1; j<=L_jend_r_true[g]; j++) 
        for (int i=L_istart_r_true[g]; i<=L_iend_r_true[g]; i++) {
          top_buf_out_r[g][kk++]= IN_R(g,i,j);
        }
        MPI_Isend(top_buf_out_r[g], RADIUS*L_width_r_true[g],MPI_DTYPE, top_nbr_r[g], 
                  99, comm_r[g], &(request_r[g][0]));
      }

      if (bottom_nbr_r[g] != -1) {
        MPI_Irecv(bottom_buf_in_r[g], RADIUS*L_width_r_true[g], MPI_DTYPE, bottom_nbr_r[g], 
                  99, comm_r[g], &(request_r[g][3]));
        for (int kk=0,j=L_jstart_r_true[g]; j<=L_jstart_r_true[g]+RADIUS-1; j++) 
        for (int i=L_istart_r_true[g]; i<=L_iend_r_true[g]; i++) {
          bottom_buf_out_r[g][kk++]= IN_R(g,i,j);
        }
        MPI_Isend(bottom_buf_out_r[g], RADIUS*L_width_r_true[g],MPI_DTYPE, bottom_nbr_r[g], 
                  101, comm_r[g], &(request_r[g][2]));
      }

      if (top_nbr_r[g] != -1) {
        MPI_Wait(&(request_r[g][0]), MPI_STATUS_IGNORE);
        MPI_Wait(&(request_r[g][1]), MPI_STATUS_IGNORE);
        for (int kk=0,j=L_jend_r_true[g]+1; j<=L_jend_r_true[g]+RADIUS; j++) 
        for (int i=L_istart_r_true[g]; i<=L_iend_r_true[g]; i++) {
          IN_R(g,i,j) = top_buf_in_r[g][kk++];
        }
      }

      if (bottom_nbr_r[g] != -1) {
        MPI_Wait(&(request_r[g][2]), MPI_STATUS_IGNORE);
        MPI_Wait(&(request_r[g][3]), MPI_STATUS_IGNORE);
        for (int kk=0,j=L_jstart_r_true[g]-RADIUS; j<=L_jstart_r_true[g]-1; j++) 
        for (int i=L_istart_r_true[g]; i<=L_iend_r_true[g]; i++) {
          IN_R(g,i,j) = bottom_buf_in_r[g][kk++];
        }
      }

      /* need to fetch ghost point data from neighbors in x-direction                 */
      if (right_nbr_r[g] != -1) {
        MPI_Irecv(right_buf_in_r[g], RADIUS*L_height_r_true[g], MPI_DTYPE, right_nbr_r[g], 
                  1010, comm_r[g], &(request_r[g][1+4]));
        for (int kk=0,j=L_jstart_r_true[g]; j<=L_jend_r_true[g]; j++) {
          for (int i=L_iend_r_true[g]-RADIUS+1; i<=L_iend_r_true[g]; i++) {
            right_buf_out_r[g][kk++]= IN_R(g,i,j);
          }
        }
        MPI_Isend(right_buf_out_r[g], RADIUS*L_height_r_true[g], MPI_DTYPE, right_nbr_r[g], 
                  990, comm_r[g], &(request_r[g][0+4]));
      }

      if (left_nbr_r[g] != -1) {
        MPI_Irecv(left_buf_in_r[g], RADIUS*L_height_r_true[g], MPI_DTYPE, left_nbr_r[g], 
                  990, comm_r[g], &(request_r[g][3+4]));
        for (int kk=0,j=L_jstart_r_true[g]; j<=L_jend_r_true[g]; j++) {
          for (int i=L_istart_r_true[g]; i<=L_istart_r_true[g]+RADIUS-1; i++) {
            left_buf_out_r[g][kk++]= IN_R(g,i,j);
          }
        }
        MPI_Isend(left_buf_out_r[g], RADIUS*L_height_r_true[g], MPI_DTYPE, left_nbr_r[g], 
                  1010, comm_r[g], &(request_r[g][2+4]));
      }

      if (right_nbr_r[g] != -1) {
        MPI_Wait(&(request_r[g][0+4]), MPI_STATUS_IGNORE);
        MPI_Wait(&(request_r[g][1+4]), MPI_STATUS_IGNORE);
        for (int kk=0,j=L_jstart_r_true[g]; j<=L_jend_r_true[g]; j++) {
          for (int i=L_iend_r_true[g]+1; i<=L_iend_r_true[g]+RADIUS; i++) {
            IN_R(g,i,j) = right_buf_in_r[g][kk++];
          }
        }
      }

      if (left_nbr_r[g] != -1) {
        MPI_Wait(&(request_r[g][2+4]), MPI_STATUS_IGNORE);
        MPI_Wait(&(request_r[g][3+4]), MPI_STATUS_IGNORE);
        for (int kk=0,j=L_jstart_r_true[g]; j<=L_jend_r_true[g]; j++) {
          for (int i=L_istart_r_true[g]-RADIUS; i<=L_istart_r_true[g]-1; i++) {
            IN_R(g,i,j) = left_buf_in_r[g][kk++];
          }
        }
      }

      for (j=MAX(RADIUS,L_jstart_r_true[g]); j<=MIN(n_r_true-RADIUS-1,L_jend_r_true[g]); j++) {
        for (i=MAX(RADIUS,L_istart_r_true[g]); i<=MIN(n_r_true-RADIUS-1,L_iend_r_true[g]); i++) {
          #if LOOPGEN
            #include "loop_body_star_amr.incl"
          #else
            for (jj=-RADIUS; jj<=RADIUS; jj++)  OUT_R(g,i,j) += WEIGHT_R(0,jj)*IN_R(g,i,j+jj);
            for (ii=-RADIUS; ii<0; ii++)        OUT_R(g,i,j) += WEIGHT_R(ii,0)*IN_R(g,i+ii,j);
            for (ii=1; ii<=RADIUS; ii++)        OUT_R(g,i,j) += WEIGHT_R(ii,0)*IN_R(g,i+ii,j);
          #endif
        }
      }

      /* add constant to solution to force refresh of neighbor data, if any        */
      for (j=L_jstart_r_true[g]; j<=L_jend_r_true[g]; j++) 
        for (i=L_istart_r_true[g]; i<=L_iend_r_true[g]; i++) IN_R(g,i,j)+= (DTYPE)1.0;
    }
  }

  /* Apply the stencil operator to background grid                                 */
  if (comm_bg != MPI_COMM_NULL) {
    for (int j=MAX(L_jstart_bg,RADIUS); j<=MIN(n-RADIUS-1,L_jend_bg); j++) {
      for (int i=MAX(L_istart_bg,RADIUS); i<=MIN(n-RADIUS-1,L_iend_bg); i++) {
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
    for (int j=L_jstart_bg; j<=L_jend_bg; j++)
    for (int i=L_istart_bg; i<=L_iend_bg; i++) IN(i,j)+= 1.0;
  }

} 
