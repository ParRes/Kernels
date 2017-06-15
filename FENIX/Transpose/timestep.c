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
	       double * RESTRICT Work_out_p)
{
  int i, j, it, jt, istart, phase;
  int send_to, recv_from;
#if !SYNCHRONOUS
    MPI_Request send_req, recv_req;
#endif

  /* do the local transpose                                                     */
  istart = colstart;
  if (!tiling) {
    for (i=0; i<Block_order; i++)
      for (j=0; j<Block_order; j++) {
        B(j,i) += A(i,j);
        A(i,j) += 1.0;
      }
  }
  else {
    for (i=0; i<Block_order; i+=Tile_order)
      for (j=0; j<Block_order; j+=Tile_order)
        for (it=i; it<MIN(Block_order,i+Tile_order); it++)
          for (jt=j; jt<MIN(Block_order,j+Tile_order);jt++) {
            B(jt,it) += A(it,jt);
            A(it,jt) += 1.0;
          }
  }

  for (phase=1; phase<Num_procs; phase++){
    recv_from = (my_ID + phase            )%Num_procs;
    send_to   = (my_ID - phase + Num_procs)%Num_procs;

#if !SYNCHRONOUS
    MPI_Irecv(Work_in_p, Block_size, MPI_DOUBLE,
              recv_from, phase, MPI_COMM_WORLD, &recv_req);
#endif

    istart = send_to*Block_order;
    if (!tiling) {
      for (i=0; i<Block_order; i++)
        for (j=0; j<Block_order; j++){
          Work_out(j,i) = A(i,j);
          A(i,j) += 1.0;
        }
    }
    else {
      for (i=0; i<Block_order; i+=Tile_order)
        for (j=0; j<Block_order; j+=Tile_order)
          for (it=i; it<MIN(Block_order,i+Tile_order); it++)
            for (jt=j; jt<MIN(Block_order,j+Tile_order);jt++) {
              Work_out(jt,it) = A(it,jt);
              A(it,jt) += 1.0;
            }
    }

#if !SYNCHRONOUS
    MPI_Isend(Work_out_p, Block_size, MPI_DOUBLE, send_to,
              phase, MPI_COMM_WORLD, &send_req);
    MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
    MPI_Wait(&send_req, MPI_STATUS_IGNORE);
#else
    MPI_Sendrecv(Work_out_p, Block_size, MPI_DOUBLE, send_to, phase,
                 Work_in_p, Block_size, MPI_DOUBLE,
                 recv_from, phase, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif

    istart = recv_from*Block_order;
    /* scatter received block to transposed matrix; no need to tile */
    for (j=0; j<Block_order; j++)
      for (i=0; i<Block_order; i++)
        B(i,j) += Work_in(i,j);
  }  /* end of phase loop  */
}
