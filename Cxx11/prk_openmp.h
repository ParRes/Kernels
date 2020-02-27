///
/// Copyright (c) 2018, Intel Corporation
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

#ifndef PRK_OPENMP_H
#define PRK_OPENMP_H

#define PRAGMA(x) _Pragma(#x)

#ifdef _OPENMP
# include <omp.h>
# define OMP(x) PRAGMA(omp x)
# define OMP_PARALLEL(x) PRAGMA(omp parallel x)
# define OMP_PARALLEL_FOR_REDUCE(x) PRAGMA(omp parallel for reduction (x) )
# define OMP_MASTER PRAGMA(omp master)
# define OMP_BARRIER PRAGMA(omp barrier)
# define OMP_FOR(x) PRAGMA(omp for x)
# define OMP_FOR_REDUCE(x) PRAGMA(omp for reduction (x) )
// OpenMP SIMD if supported, else not.
# if (_OPENMP >= 201300) || (__ibmxl_version__ >= 16)
#  define OMP_SIMD PRAGMA(omp simd)
#  define OMP_FOR_SIMD PRAGMA(omp for simd)
#  define OMP_TASK(x) PRAGMA(omp task x)
#  define OMP_TASKLOOP(x) PRAGMA(omp taskloop x )
#  if defined(__INTEL_COMPILER)
#   define OMP_TASKLOOP_COLLAPSE(n,x) PRAGMA(omp taskloop x )
#  else
#   define OMP_TASKLOOP_COLLAPSE(n,x) PRAGMA(omp taskloop collapse(n) x )
#  endif
#  define OMP_TASKWAIT PRAGMA(omp taskwait)
#  define OMP_ORDERED(x) PRAGMA(omp ordered x)
#  define OMP_TARGET(x) PRAGMA(omp target x)
#  define OMP_DECLARE_TARGET PRAGMA(omp declare target)
#  define OMP_END_DECLARE_TARGET PRAGMA(omp end declare target)
# else
#  define OMP_SIMD
#  define OMP_FOR_SIMD PRAGMA(omp for)
#  define OMP_TASK(x)
#  define OMP_TASKLOOP(x)
#  define OMP_TASKLOOP_COLLAPSE(n,x)
#  define OMP_TASKWAIT
#  define OMP_ORDERED(x)
#  define OMP_TARGET(x)
#  define OMP_DECLARE_TARGET
#  define OMP_END_DECLARE_TARGET
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
# define OMP_FOR_SIMD
# define OMP_TASK(x)
# define OMP_TASKLOOP(x)
# define OMP_TASKLOOP_COLLAPSE(n,x)
# define OMP_TASKWAIT
# define OMP_ORDERED(x)
# define OMP_TARGET(x)
# define OMP_DECLARE_TARGET
# define OMP_END_DECLARE_TARGET
#endif

// used in OpenMP target code because std::min etc are not declare target
#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif
#ifndef MAX
#define MAX(x,y) ((x)>(y)?(x):(y))
#endif
#ifndef ABS
#define ABS(a) ((a) >= 0 ? (a) : -(a))
#endif

#endif /* PRK_OPENMP_H */
