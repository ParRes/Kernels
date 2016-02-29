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

NAME:      bail_out

PURPOSE:   Exit gracefully when an SHMEM process has encountered an error
  
Arguments: error code, work space  

Returns:   nothing, but the program terminates with a nonzero exit status

Notes:     This function must be called by all participating processes

HISTORY: - Written by Gabriele Jost, March 2015.
  
**********************************************************************************/

#include <par-res-kern_general.h>
#include <par-res-kern_shmem.h>

void bail_out (int error) {
   long *global_error;
   long *local_error;
   long *pWrk;
   long *pSync_local;

   int i;
   global_error = prk_shmem_align(prk_get_alignment(),sizeof(long));
   local_error = prk_shmem_align(prk_get_alignment(),sizeof(long));
   pWrk = prk_shmem_align(prk_get_alignment(),sizeof(long)*PRK_SHMEM_REDUCE_MIN_WRKDATA_SIZE);
   pSync_local = prk_shmem_align(prk_get_alignment(),sizeof(long)*PRK_SHMEM_REDUCE_SYNC_SIZE);
   for (i = 0; i < PRK_SHMEM_REDUCE_SYNC_SIZE; i += 1) {
    pSync_local[i] = PRK_SHMEM_SYNC_VALUE;
   }
   local_error [0] = error;
   shmem_barrier_all ();
   shmem_long_max_to_all (global_error, local_error, 1, 0, 0, prk_shmem_n_pes(), pWrk, pSync_local); 
   if (global_error [0] > 0) {
     prk_shmem_finalize ();
     exit (1);
  }
  return;
}

