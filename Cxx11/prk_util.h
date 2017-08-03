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

#if !(defined(__cplusplus) && (__cplusplus >= 201103L))
# error You need a C++11 compiler.
#endif

#define PRAGMA(x) _Pragma(#x)

#include <cstdio>  // atoi
#include <cstdlib> // getenv
#include <cstdint>
#include <climits>
#include <cmath>   // fabs
#include <cassert>

#include <string>
#include <iostream>
#include <iomanip> // std::setprecision
#include <exception>
#include <chrono>
#include <random>

#include <list>
#include <vector>
#include <valarray>
#include <array>
#include <thread>
#include <future>
#include <atomic>
#include <numeric>
#include <algorithm>

#ifdef _OPENMP
# include <omp.h>
# define OMP_PARALLEL(x) PRAGMA(omp parallel x)
# define OMP_PARALLEL_FOR_REDUCE(x) PRAGMA(omp parallel for reduction (x) )
# define OMP_MASTER PRAGMA(omp master)
# define OMP_BARRIER PRAGMA(omp barrier)
# define OMP_FOR(x) PRAGMA(omp for x)
# define OMP_FOR_REDUCE(x) PRAGMA(omp for reduction (x) )
// OpenMP SIMD if supported, else not.
# if (_OPENMP >= 201300)
#  define OMP_SIMD PRAGMA(omp simd)
#  define OMP_FOR_SIMD PRAGMA(omp for simd)
#  define OMP_TASK(x) PRAGMA(omp task x)
#  define OMP_TASKLOOP(x) PRAGMA(omp taskloop x )
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
#  define OMP_TASKWAIT
#  define OMP_ORDERED(x)
#  define OMP_TARGET(x)
#  define OMP_DECLARE_TARGET
#  define OMP_END_DECLARE_TARGET
# endif
#else
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
# define OMP_TASKWAIT
# define OMP_ORDERED(x)
# define OMP_TARGET(x)
# define OMP_DECLARE_TARGET
# define OMP_END_DECLARE_TARGET
#endif

#ifdef __cilk
# include <cilk/cilk.h>
#endif

#if defined(__INTEL_COMPILER) && !defined(PRAGMA_OMP_SIMD)
# define PRAGMA_SIMD PRAGMA(simd)
#else
# define PRAGMA_SIMD
#endif

#ifdef USE_TBB
# include <tbb/tbb.h>
# include <tbb/parallel_for.h>
# include <tbb/blocked_range.h>
#endif

#ifdef USE_BOOST
# include <boost/range/irange.hpp>
#endif

#ifdef USE_PSTL
# if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1800)
#  include <pstl/execution>
#  include <pstl/algorithm>
#  include <pstl/numeric>
#  include <pstl/memory>
# elif defined(__GNUC__) && defined(__GNUC_MINOR__) && \
       ( (__GNUC__ >= 8) || (__GNUC__ == 7) && (__GNUC_MINOR__ >= 2) )
#  include <parallel/algorithm>
#  include <parallel/numeric>
# endif
#endif

#ifdef USE_KOKKOS
# include <typeinfo>
# include <Kokkos_Core.hpp>
#endif

#ifdef USE_RAJA
# define RAJA_ENABLE_NESTED 1
# include "RAJA/RAJA.hpp"
#endif

#define RESTRICT __restrict__

namespace prk {

    static inline double wtime(void)
    {
#ifdef _OPENMP
        return omp_get_wtime();
#else
        using t = std::chrono::high_resolution_clock;
        auto c = t::now().time_since_epoch().count();
        auto n = t::period::num;
        auto d = t::period::den;
        double r = static_cast<double>(c)/static_cast<double>(d)*static_cast<double>(n);
        return r;
#endif
    }

} // namespace prk

#endif /* PRK_UTIL_H */
