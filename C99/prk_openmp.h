#ifndef PRK_OPENMP_H
#define PRK_OPENMP_H

/* TODO Need to check for OpenMP 4 to use "omp simd",
 *      and fallback to compiler-specific alternatives if not available. */

#if defined(_OPENMP) && (( __STDC_VERSION__ >= 199901L ) || (__cplusplus >= 201103L ))

#define PRAGMA(x) _Pragma(#x)

#define OMP_PARALLEL(a) PRAGMA(omp parallel a)
#define OMP_FOR(a) PRAGMA(omp for schedule(static) a)
#define OMP_SIMD(a) PRAGMA(omp simd a)
#define OMP_BARRIER PRAGMA(omp barrier)
#define OMP_MASTER PRAGMA(omp master)

#else

#define OMP_PARALLEL(a)
#define OMP_FOR(a)
#define OMP_SIMD(a)
#define OMP_BARRIER
#define OMP_MASTER

#endif

#endif // PRK_OPENMP_H
