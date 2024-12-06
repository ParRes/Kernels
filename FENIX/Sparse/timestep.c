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

void time_step(int               Num_procs,
               int               my_ID,
               int               stencil_size,
               int               nrows,
               double * RESTRICT matrix,
               double * RESTRICT vector,
               double * RESTRICT result,  
               s64Int * RESTRICT colIndex)
{

  int row_offset;
  s64Int row, col, first, last; 
  double temp;

  /* fill vector                                                                */
  row_offset = nrows*my_ID;
  for (row=row_offset; row<nrows+row_offset; row++) vector[row] += (double) (row+1);

  /* replicate vector on all ranks                                              */
  MPI_Allgather(MPI_IN_PLACE, nrows, MPI_DOUBLE, vector, nrows, MPI_DOUBLE,
                MPI_COMM_WORLD);

  /* do the actual matrix multiplication                                        */
  for (row=0; row<nrows; row++) {
    first = stencil_size*row; last = first+stencil_size-1;
    for (temp=0.0,col=first; col<=last; col++) {
      temp += matrix[col]*vector[colIndex[col]];
    }
    result[row] += temp;
  }
} 
