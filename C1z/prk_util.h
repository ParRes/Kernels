///
/// Copyright (c) 2013, Intel Corporation
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions
/// are met:
///
/// * Redistributions of source code must retain the above copyright
///       notice, this list of conditions and the following disclaimer.
/// * Redistributions in binary form must reproduce the above
///       copyright notice, this list of conditions and the following
///       disclaimer in the documentation and/or other materials provided
///       with the distribution.
/// * Neither the name of Intel Corporation nor the names of its
///       contributors may be used to endorse or promote products
///       derived from this software without specific prior written
///       permission.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
/// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
/// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
/// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
/// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
/// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
/// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
/// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
/// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
/// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
/// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
/// POSSIBILITY OF SUCH DAMAGE.

#ifndef PRK_UTIL_H
#define PRK_UTIL_H

#if !(defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L))
# error You need a C99+ compiler.
#endif

#define PRAGMA(x) _Pragma(#x)

#include <stdio.h>   // atoi
#include <stdlib.h>  // getenv

int posix_memalign(void **memptr, size_t alignment, size_t size);

#include <stdint.h>
#if defined(__PGIC__)
typedef _Bool bool;
const bool true=1;
const bool false=0;
#else
#include <stdbool.h> // bool
#endif
#include <string.h>
#include <limits.h>
#include <math.h>    // fabs
#include <time.h>    // clock_gettime, timespec_get
#include <assert.h>
#include <errno.h>

#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif
#ifndef MAX
#define MAX(x,y) ((x)>(y)?(x):(y))
#endif
#ifndef ABS
#define ABS(a) ((a) >= 0 ? (a) : -(a))
#endif

#ifdef _OPENMP
# include <omp.h>
# define OMP(x) PRAGMA(omp x)
# define OMP_PARALLEL(x) PRAGMA(omp parallel x)
# define OMP_PARALLEL_FOR_REDUCE(x) PRAGMA(omp parallel for reduction (x) )
# define OMP_MASTER PRAGMA(omp master)
# define OMP_BARRIER PRAGMA(omp barrier)
# define OMP_FOR(x) PRAGMA(omp for x)
# define OMP_FOR_REDUCE(x) PRAGMA(omp for reduction (x) )
# if (_OPENMP >= 201300) || (__ibmxl_version__ >= 16)
#  define OMP_SIMD PRAGMA(omp simd)
#  define OMP_FOR_SIMD(x) PRAGMA(omp for simd x)
#  define OMP_TASK(x) PRAGMA(omp task x)
#  define OMP_TASKLOOP(x) PRAGMA(omp taskloop x )
#  define OMP_TASKWAIT PRAGMA(omp taskwait)
#  define OMP_ORDERED(x) PRAGMA(omp ordered x)
#  define OMP_TARGET(x) PRAGMA(omp target x)
# else
#  warning No OpenMP 4+ features!
#  define OMP_SIMD
#  define OMP_FOR_SIMD(x) PRAGMA(omp for x)
#  define OMP_TASK(x)
#  define OMP_TASKLOOP(x)
#  define OMP_TASKWAIT
#  define OMP_ORDERED(x)
#  define OMP_TARGET(x)
# endif
#else
# define OMP(x)
# define OMP_PARALLEL(x)
# define OMP_PARALLEL_FOR_REDUCE(x)
# define OMP_MASTER
# define OMP_BARRIER
# define OMP_FOR(x)
# define OMP_FOR_REDUCE(x)
# define OMP_SIMD
# define OMP_FOR_SIMD(x)
# define OMP_TASK(x)
# define OMP_TASKLOOP(x)
# define OMP_TASKWAIT
# define OMP_ORDERED(x)
# define OMP_TARGET(x)
#endif

#ifdef __cilk
# include <cilk/cilk.h>
// Not defined in the header but documented at https://www.cilkplus.org/.
int __cilkrts_get_nworkers(void);
#endif

#if defined(__INTEL_COMPILER)
# define PRAGMA_SIMD PRAGMA(vector)
#elif defined(__GNUC__) && defined(__GNUC_MINOR__) && ( ( (__GNUC__ == 4) && (__GNUC_MINOR__ == 9) ) || (__GNUC__ >= 5) )
# define PRAGMA_SIMD PRAGMA(GCC ivdep)
#elif defined(__clang__)
# define PRAGMA_SIMD PRAGMA(clang loop vectorize(enable))
#else
# define PRAGMA_SIMD
#endif

#ifdef __linux__
#include <features.h>
#endif

// If we are on Linux and we are not using GLIBC, attempt to
// use C11 threads, because this means we are using MUSL.
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L) && \
   !defined(__STDC_NO_THREADS__) && \
   ( defined(USE_C11_THREADS) || \
     ( defined(__linux__) && !defined(__GNU_LIBRARY__) && !defined(__GLIBC__) ) \
   )
# define HAVE_C11_THREADS
# include <threads.h>
#else
# define HAVE_PTHREADS
# include <pthread.h>
#endif

#if defined(_OPENMP)

#include <omp.h>

// OpenMP has its own timer and is desirable since it will
// not overcount by measuring processor time.
static inline double prk_wtime(void)
{
    return omp_get_wtime();
}

// Apple does not have C11 support in the C standard library...
#elif defined(__APPLE__)

#include <mach/mach.h>
#include <mach/mach_time.h>

static inline double prk_wtime(void)
{
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    uint64_t at = mach_absolute_time();
    return ( 1.e-9 * at * info.numer / info.denom );
}

// gettimeofday is the worst timer, but should work everywhere.
// This addresses issues with clock_gettime and timespec_get
// library support in Travis.
#elif defined(PRK_USE_GETTIMEOFDAY)

#include <sys/time.h>

static inline double prk_wtime(void)
{
  struct timeval tv;
  gettimeofday( &tv, NULL);
  double t  = (double) tv.tv_sec + (double) tv.tv_usec * 1.0e-6;
  return t;
}

// GCC claims to be C11 without knowing if glibc is compliant.
// glibc added support for timespec_get in version 2.16.
// (https://gcc.gnu.org/wiki/C11Status)
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L) && \
      defined(__GLIBC__) && defined(__GLIBC_MINOR__) && \
      (((__GLIBC__ == 2) && (__GLIBC_MINOR__ >= 16)) || (__GLIBC__ > 2))

static inline double prk_wtime(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    time_t s  = ts.tv_sec;
    long   ns = ts.tv_nsec;
    double t  = (double)s + 1.e-9 * (double)ns;
    return t;
}

// clock_gettime is not supported everywhere, or requires explicit librt.
#elif defined(CLOCK_PROCESS_CPUTIME_ID)

static inline double prk_wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    time_t s  = ts.tv_sec;
    long   ns = ts.tv_nsec;
    double t  = (double)s + 1.e-9 * (double)ns;
    return t;
}

#else

#include <sys/time.h>

static inline double prk_wtime(void)
{
  struct timeval tv;
  gettimeofday( &tv, NULL);
  double t  = (double) tv.tv_sec + (double) tv.tv_usec * 1.0e-6;
  return t;
}

#endif // timers

static inline int prk_get_alignment(void)
{
    /* a := alignment */
#ifdef PRK_ALIGNMENT
    int a = PRK_ALIGNMENT;
#else
    char* temp = getenv("PRK_ALIGNMENT");
    int a = (temp!=NULL) ? atoi(temp) : 64;
    if (a < 8) a = 8;
    assert( (a & (~a+1)) == a ); /* is power of 2? */
#endif
    return a;
}

void * prk_malloc(size_t bytes)
{
    int alignment = prk_get_alignment();

#if defined(__INTEL_COMPILER)

    return (void*)_mm_malloc( bytes, alignment);

// We cannot use C11 aligned_alloc on Mac.
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=69680 */
// GCC claims to be C11 without knowing if glibc is compliant...
#elif !defined(__GNUC__) && \
      !defined(__APPLE__) && \
       defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)

// From ISO C11:
//
// "The aligned_alloc function allocates space for an object
//  whose alignment is specified by alignment, whose size is
//  specified by size, and whose value is indeterminate.
//  The value of alignment shall be a valid alignment supported
//  by the implementation and the value of size shall be an
//  integral multiple of alignment."
//
//  Thus, if we do not round up the bytes to be a multiple
//  of the alignment, we violate ISO C.

    size_t padded = bytes;
    size_t excess = bytes % alignment;
    if (excess>0) padded += (alignment - excess);
    return aligned_alloc(alignment,padded);

#else

    void * ptr = NULL;
    int ret = posix_memalign(&ptr,alignment,bytes);
    if (ret!=0) ptr = NULL;
    return ptr;

#endif

}

static inline void prk_free(void * p)
{
#if defined(__INTEL_COMPILER)
    _mm_free(p);
#else
    free(p);
#endif
}

static inline void prk_lookup_posix_error(int e, char * n, int l)
{
    switch (e) {
        case EACCES:
            strncpy(n,"EACCES",l);
            break;
        case EAGAIN:
            strncpy(n,"EAGAIN",l);
            break;
        case EBADF:
            strncpy(n,"EBADF",l);
            break;
        case EEXIST:
            strncpy(n,"EEXIST",l);
            break;
        case EINVAL:
            strncpy(n,"EINVAL",l);
            break;
        case ENFILE:
            strncpy(n,"ENFILE",l);
            break;
        case ENODEV:
            strncpy(n,"ENODEV",l);
            break;
        case ENOMEM:
            strncpy(n,"ENOMEM",l);
            break;
        case EPERM:
            strncpy(n,"EPERM",l);
            break;
        case ETXTBSY:
            strncpy(n,"ETXTBSY",l);
            break;
        case EOPNOTSUPP:
            strncpy(n,"EOPNOTSUPP",l);
            break;
        /*
        case E:
            strncpy(n,"E",l);
            break;
        */
        default:
            printf("error code %d unknown\n", e);
            strncpy(n,"UNKNOWN",l);
            break;
    }
}

#endif /* PRK_UTIL_H */
