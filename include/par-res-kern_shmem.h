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

#include <shmem.h>

extern void bail_out(int);

#if defined(SHMEM_MAJOR_VERSION) && defined(SHMEM_MINOR_VERSION)
/* OpenSHMEM 1.3 and later */
# if (SHMEM_MAJOR_VERSION>1) || ((SHMEM_MAJOR_VERSION == 1) && (SHMEM_MINOR_VERSION >= 3))
#  define PRK_SHMEM_BCAST_SYNC_SIZE SHMEM_BCAST_SYNC_SIZE
#  define PRK_SHMEM_REDUCE_SYNC_SIZE SHMEM_REDUCE_SYNC_SIZE
#  define PRK_SHMEM_SYNC_VALUE SHMEM_SYNC_VALUE
#  define PRK_SHMEM_REDUCE_MIN_WRKDATA_SIZE SHMEM_REDUCE_MIN_WRKDATA_SIZE
/* OpenSHMEM 1.2 and earlier, or other SHMEM that defines VERSION symbols */
# else
#  define PRK_SHMEM_BCAST_SYNC_SIZE _SHMEM_BCAST_SYNC_SIZE
#  define PRK_SHMEM_REDUCE_SYNC_SIZE _SHMEM_REDUCE_SYNC_SIZE
#  define PRK_SHMEM_SYNC_VALUE _SHMEM_SYNC_VALUE
#  define PRK_SHMEM_REDUCE_MIN_WRKDATA_SIZE _SHMEM_REDUCE_MIN_WRKDATA_SIZE
# endif
/* no VERSION symbols */
#else
/* PRK_SHMEM_BCAST_SYNC_SIZE */
# if defined(SHMEM_BCAST_SYNC_SIZE)
#  define PRK_SHMEM_BCAST_SYNC_SIZE SHMEM_BCAST_SYNC_SIZE
# elif defined(_SHMEM_BCAST_SYNC_SIZE)
#  define PRK_SHMEM_BCAST_SYNC_SIZE _SHMEM_BCAST_SYNC_SIZE
# else
#  error No preprocesor definition of SHMEM_BCAST_SYNC_SIZE!
# endif
/* PRK_SHMEM_REDUCE_SYNC_SIZE */
# if defined(SHMEM_REDUCE_SYNC_SIZE)
#  define PRK_SHMEM_REDUCE_SYNC_SIZE SHMEM_REDUCE_SYNC_SIZE
# elif defined(_SHMEM_REDUCE_SYNC_SIZE)
#  define PRK_SHMEM_REDUCE_SYNC_SIZE _SHMEM_REDUCE_SYNC_SIZE
# else
#  error No preprocesor definition of SHMEM_REDUCE_SYNC_SIZE!
# endif
/* PRK_SHMEM_SYNC_VALUE */
# if defined(SHMEM_SYNC_VALUE)
#  define PRK_SHMEM_SYNC_VALUE SHMEM_SYNC_VALUE
# elif defined(SHMEM_SYNC_VALUE)
#  define PRK_SHMEM_SYNC_VALUE _SHMEM_SYNC_VALUE
# else
#  error No preprocesor definition of SHMEM_SYNC_VALUE!
# endif
/* PRK_SHMEM_REDUCE_MIN_WRKDATA_SIZE */
# if defined(SHMEM_REDUCE_MIN_WRKDATA_SIZE)
#  define PRK_SHMEM_REDUCE_MIN_WRKDATA_SIZE SHMEM_REDUCE_MIN_WRKDATA_SIZE
# elif defined(SHMEM_REDUCE_MIN_WRKDATA_SIZE)
#  define PRK_SHMEM_REDUCE_MIN_WRKDATA_SIZE _SHMEM_REDUCE_MIN_WRKDATA_SIZE
# else
#  error No preprocesor definition of SHMEM_REDUCE_MIN_WRKDATA_SIZE!
# endif
/* end manual definition detection */
#endif // VERSION symbols




