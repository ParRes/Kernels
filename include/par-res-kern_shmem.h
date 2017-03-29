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

#ifndef PRK_SHMEM_H
#define PRK_SHMEM_H

#include "par-res-kern_general.h"

#include <shmem.h>

extern void bail_out(int);

#if defined(SHMEM_MAJOR_VERSION) && defined(SHMEM_MINOR_VERSION)
#if ((SHMEM_MAJOR_VERSION>1) || ((SHMEM_MAJOR_VERSION == 1) && (SHMEM_MINOR_VERSION >= 2)))
#define PRK_HAVE_OPENSHMEM_1_2
#endif
/* Cray SHMEM provides some but not all of the changes in OpenSHMEM 1.2. */
#elif defined(CRAY_SHMEM_NUMVERSION)
#define PRK_HAVE_CRAY_SHMEM
#endif

static void prk_shmem_init(void) {
#if defined(PRK_HAVE_OPENSHMEM_1_2) || defined(PRK_HAVE_CRAY_SHMEM)
    shmem_init();
#else
    start_pes(0);
#endif
}

static void prk_shmem_finalize(void) {
#if defined(PRK_HAVE_OPENSHMEM_1_2) || defined(PRK_HAVE_CRAY_SHMEM)
    shmem_finalize();
#else
    /* shmem_finalize contains implicit barrier, so we include this
     * to be consistent, even though there is no matching call to start_pes(). */
    shmem_barrier_all();
#endif
}

static int prk_shmem_my_pe(void) {
#if defined(PRK_HAVE_OPENSHMEM_1_2) || defined(PRK_HAVE_CRAY_SHMEM)
    return shmem_my_pe();
#else
    return _my_pe();
#endif
}

static int prk_shmem_n_pes(void) {
#if defined(PRK_HAVE_OPENSHMEM_1_2) || defined(PRK_HAVE_CRAY_SHMEM)
    return shmem_n_pes();
#else
    return _num_pes();
#endif
}

static void * prk_shmem_malloc(size_t n) {
#ifdef PRK_HAVE_OPENSHMEM_1_2
    return shmem_malloc(n);
#elif defined(PRK_HAVE_CRAY_SHMEM)
    /* Cray SHMEM has not had barrier semantics for shmalloc in the past. */
    void * ptr = shmalloc(n);
    shmem_barrier_all();
    return ptr;
#else
    return shmalloc(n);
#endif
}

static void * prk_shmem_realloc(void *ptr, size_t size) {
#ifdef PRK_HAVE_OPENSHMEM_1_2
    return shmem_realloc(ptr,size);
#else
    return shrealloc(ptr,size);
#endif
}

static void * prk_shmem_align(size_t alignment, size_t size) {
#ifdef PRK_HAVE_OPENSHMEM_1_2
    return shmem_align(alignment,size);
#else
    /* It seems that shmalign did not exist. */
    return shmalloc(size);
#endif
}

static void prk_shmem_free(void * ptr) {
#ifdef PRK_HAVE_OPENSHMEM_1_2
    shmem_free(ptr);
#else
    shfree(ptr);
#endif
}

/* The SHMEM_* constants were added in OpenSHMEM 1.2
 * and the _SHMEM* constants were deprecated in OpenSHMEM 1.3. */
#ifdef PRK_HAVE_OPENSHMEM_1_2
#  define PRK_SHMEM_BCAST_SYNC_SIZE SHMEM_BCAST_SYNC_SIZE
#  define PRK_SHMEM_REDUCE_SYNC_SIZE SHMEM_REDUCE_SYNC_SIZE
#  define PRK_SHMEM_SYNC_VALUE SHMEM_SYNC_VALUE
#  define PRK_SHMEM_REDUCE_MIN_WRKDATA_SIZE SHMEM_REDUCE_MIN_WRKDATA_SIZE
#else
#  define PRK_SHMEM_BCAST_SYNC_SIZE _SHMEM_BCAST_SYNC_SIZE
#  define PRK_SHMEM_REDUCE_SYNC_SIZE _SHMEM_REDUCE_SYNC_SIZE
#  define PRK_SHMEM_SYNC_VALUE _SHMEM_SYNC_VALUE
#  define PRK_SHMEM_REDUCE_MIN_WRKDATA_SIZE _SHMEM_REDUCE_MIN_WRKDATA_SIZE
#endif

#endif /* PRK_SHMEM_H */
