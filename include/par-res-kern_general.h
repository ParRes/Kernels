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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <unistd.h>

#define PRKVERSION "2.17"
#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif
#ifndef MAX
#define MAX(x,y) ((x)>(y)?(x):(y))
#endif
#ifndef ABS
#define ABS(a) ((a) >= 0 ? (a) : -(a))
#endif

#if RESTRICT_KEYWORD
  #ifdef __GNUC__
    #define RESTRICT __restrict__
  #else
    #define RESTRICT restrict
  #endif
#else
  #define RESTRICT
#endif

/* Define 64-bit types and corresponding format strings for printf() */
#ifdef LONG_IS_64BITS
  typedef unsigned long      u64Int;
  typedef long               s64Int;
  #define FSTR64             "%16ld"
  #define FSTR64U            "%16lu"
#else
  typedef unsigned long long u64Int;
  typedef long long          s64Int;
  #define FSTR64             "%16ll"
  #define FSTR64U            "%16llu"
#endif

extern double wtime(void);

/*  We cannot use C11 aligned_alloc because of this GCC 5.3.0 bug:
 *  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=69680 */
#if 0 && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#define PRK_HAS_C11 1
#endif

/* This function is separate from prk_malloc() because
 * we need it when calling prk_shmem_align(..)           */
static inline int prk_get_alignment(void)
{
    /* a := alignment */
# ifdef PRK_ALIGNMENT
    int a = PRK_ALIGNMENT;
# else
    char* temp = getenv("PRK_ALIGNMENT");
    int a = (temp!=NULL) ? atoi(temp) : 64;
    if (a < 8) a = 8;
    assert( (a & (~a+1)) == a );
#endif
    return a;
}

/* There are a variety of reasons why this function is not declared by stdlib.h. */
#if defined(__UPC__)
int posix_memalign(void **memptr, size_t alignment, size_t size);
#endif

static inline void* prk_malloc(size_t bytes)
{
#ifndef PRK_USE_MALLOC
    int alignment = prk_get_alignment();
#endif

/* Berkeley UPC throws warnings related to this function for no obvious reason... */
#if !defined(__UPC__) && defined(__INTEL_COMPILER) && !defined(PRK_USE_POSIX_MEMALIGN)
    return (void*)_mm_malloc(bytes,alignment);
#elif defined(PRK_HAS_C11)
/* From ISO C11:
 *
 * "The aligned_alloc function allocates space for an object
 *  whose alignment is specified by alignment, whose size is
 *  specified by size, and whose value is indeterminate.
 *  The value of alignment shall be a valid alignment supported
 *  by the implementation and the value of size shall be an
 *  integral multiple of alignment."
 *
 *  Thus, if we do not round up the bytes to be a multiple
 *  of the alignment, we violate ISO C.
 */
    size_t padded = bytes;
    size_t excess = bytes % alignment;
    if (excess>0) padded += (alignment - excess);
    return aligned_alloc(alignment,padded);
#elif defined(PRK_USE_MALLOC)
#warning PRK_USE_MALLOC prevents the use of alignmed memory.
    return prk_malloc(bytes);
#else /* if defined(PRK_USE_POSIX_MEMALIGN) */
    void * ptr = NULL;
    posix_memalign(&ptr,alignment,bytes);
    return ptr;
#endif
}

static inline void prk_free(void* p)
{
#if defined(__INTEL_COMPILER) && !defined(PRK_USE_POSIX_MEMALIGN)
    _mm_free(p);
#else
    free(p);
#endif
}
