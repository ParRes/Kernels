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

#define ARRAY(i,j) vector[i+1+(j)*(segment_size+1)]

void time_step(int    my_ID,
               int    root, int final,
               long   m, long n,
               long   start, long end,
               long   segment_size,
               int    Num_procs,
               int    grp,
               double * RESTRICT vector,
               double *inbuf, double *outbuf)
{
    double corner_val;
    int i, j, jj, jjsize;

    /* execute pipeline algorithm for grid lines 1 through n-1 (skip bottom line) */
    if (grp==1) for (j=1; j<n; j++) { /* special case for no grouping             */

      /* if I am not at the left boundary, I need to wait for my left neighbor to
         send data                                                                */
      if (my_ID > 0) {
        MPI_Recv(&(ARRAY(start-1,j)), 1, MPI_DOUBLE, my_ID-1, j, 
                                  MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
      }

      for (i=start; i<= end; i++) {
        ARRAY(i,j) = ARRAY(i-1,j) + ARRAY(i,j-1) - ARRAY(i-1,j-1);
      }

      /* if I am not on the right boundary, send data to my right neighbor        */  
      if (my_ID < Num_procs-1) {
        MPI_Send(&(ARRAY(end,j)), 1, MPI_DOUBLE, my_ID+1, j, MPI_COMM_WORLD);
      }
    }
    else for (j=1; j<n; j+=grp) { /* apply grouping                               */

      jjsize = MIN(grp, n-j);
      /* if I am not at the left boundary, I need to wait for my left neighbor to
         send data                                                                */
      if (my_ID > 0) {
        MPI_Recv(inbuf, jjsize, MPI_DOUBLE, my_ID-1, j, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        for (jj=0; jj<jjsize; jj++) {
          ARRAY(start-1,jj+j) = inbuf[jj];
	}
      }

      for (jj=0; jj<jjsize; jj++) for (i=start; i<= end; i++) {
        ARRAY(i,jj+j) = ARRAY(i-1,jj+j) + ARRAY(i,jj+j-1) - ARRAY(i-1,jj+j-1);
      }

      /* if I am not on the right boundary, send data to my right neighbor        */  
      if (my_ID < Num_procs-1) {
        for (jj=0; jj<jjsize; jj++) {
          outbuf[jj] = ARRAY(end,jj+j);
        }
        MPI_Send(outbuf, jjsize, MPI_DOUBLE, my_ID+1, j, MPI_COMM_WORLD);
      }

    }

    /* copy top right corner value to bottom left corner to create dependency     */
    if (Num_procs >1) {
      if (my_ID==final) {
        corner_val = -ARRAY(end,n-1);
        MPI_Send(&corner_val,1,MPI_DOUBLE,root,888,MPI_COMM_WORLD);
      }
      if (my_ID==root) {
        MPI_Recv(&(ARRAY(0,0)),1,MPI_DOUBLE,final,888,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
      }
    }
    else ARRAY(0,0)= -ARRAY(end,n-1);
}
