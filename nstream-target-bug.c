//
// icx -std=c11 -pthread -g -O3 -xHOST nstream-target-bug.c -fiopenmp -fopenmp-targets=spir64
//
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

#define PRAGMA(x) _Pragma(#x)

#include <omp.h>
#define OMP(x) PRAGMA(omp x)
#define OMP_PARALLEL(x) PRAGMA(omp parallel x)
#define OMP_PARALLEL_FOR_REDUCE(x) PRAGMA(omp parallel for reduction (x) )
#define OMP_MASTER PRAGMA(omp master)
#define OMP_BARRIER PRAGMA(omp barrier)
#define OMP_FOR(x) PRAGMA(omp for x)
#define OMP_FOR_REDUCE(x) PRAGMA(omp for reduction (x) )
#define OMP_SIMD PRAGMA(omp simd)
#define OMP_FOR_SIMD(x) PRAGMA(omp for simd x)
#define OMP_TASK(x) PRAGMA(omp task x)
#define OMP_TASKLOOP(x) PRAGMA(omp taskloop x )
#define OMP_TASKWAIT PRAGMA(omp taskwait)
#define OMP_ORDERED(x) PRAGMA(omp ordered x)
#define OMP_TARGET(x) PRAGMA(omp target x)

int main(int argc, char * argv[])
{
  printf("Parallel Research Kernels version %d\n", 0 );
  printf("C11/OpenMP TARGET STREAM triad: A = B + scalar * C\n");

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  if (argc < 3) {
    printf("Usage: <# iterations> <vector length>\n");
    return 1;
  }

  int iterations = atoi(argv[1]);
  if (iterations < 1) {
    printf("ERROR: iterations must be >= 1\n");
    return 1;
  }

  // length of a the vector
  size_t length = atol(argv[2]);
  if (length <= 0) {
    printf("ERROR: Vector length must be greater than 0\n");
    return 1;
  }

#ifdef _OPENMP
  printf("Number of threads    = %d\n", omp_get_max_threads());
#endif
  printf("Number of iterations = %d\n", iterations);
  printf("Vector length        = %zu\n", length);
  //printf("Offset               = %d\n", offset);

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double nstream_time = 0.0;

  size_t bytes = length*sizeof(double);
  double * restrict A = prk_malloc(bytes);
  double * restrict B = prk_malloc(bytes);
  double * restrict C = prk_malloc(bytes);

  double scalar = 3.0;

  // HOST
  OMP_PARALLEL()
  {
    OMP_FOR_SIMD()
    for (size_t i=0; i<length; i++) {
      A[i] = 0.0;
      B[i] = 2.0;
      C[i] = 2.0;
    }
  }

  // DEVICE
  OMP_TARGET( data map(tofrom: A[0:length]) map(to: B[0:length], C[0:length]) )
  {
    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) nstream_time = prk_wtime();

      OMP_TARGET( teams distribute parallel for simd schedule(static,1) )
      for (size_t i=0; i<length; i++) {
          A[i] += B[i] + scalar * C[i];
      }
    }
    nstream_time = prk_wtime() - nstream_time;
  }

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  double ar = 0.0;
  double br = 2.0;
  double cr = 2.0;
  for (int i=0; i<=iterations; i++) {
      ar += br + scalar * cr;
  }

  ar *= length;

  double asum = 0.0;
  OMP_PARALLEL_FOR_REDUCE( +:asum )
  for (size_t i=0; i<length; i++) {
      asum += fabs(A[i]);
  }

  double epsilon=1.e-8;
  if (fabs(ar-asum)/asum > epsilon) {
      printf("Failed Validation on output array\n"
             "       Expected checksum: %lf\n"
             "       Observed checksum: %lf\n"
             "ERROR: solution did not validate\n", ar, asum);
      return 1;
  } else {
      printf("Solution validates\n");
      double avgtime = nstream_time/iterations;
      double nbytes = 4.0 * length * sizeof(double);
      printf("Rate (MB/s): %lf Avg time (s): %lf\n", 1.e-6*nbytes/avgtime, avgtime);
  }

  return 0;
}


