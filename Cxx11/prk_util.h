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

#include <cstdio>
#include <cstdlib> // atoi, getenv
#include <cstdint>
#include <climits>
#include <cmath>   // abs, fabs
#include <cassert>

// Test standard library _after_ standard headers have been included...
#if !defined(__NVCC__) && (defined(__GLIBCXX__) || defined(_GLIBCXX_RELEASE) ) && !defined(_GLIBCXX_USE_CXX11_ABI)
# error You are using an ancient version GNU libstdc++.  Either upgrade your GCC or tell ICC to use a newer version via the -gxx-name= option.
#endif

#if !(defined(__cplusplus) && (__cplusplus >= 201103L))
# error You need a C++11 compiler or a newer C++ standard library.
#endif

#include <string>
#include <iostream>
#include <iomanip> // std::setprecision
#include <exception>
#include <list>
#include <vector>
#include <valarray>

#include <chrono>
#include <random>
#include <typeinfo>
#include <array>
#include <atomic>
#include <numeric>
#include <algorithm>

template<class I, class T>
const T prk_reduce(I first, I last, T init) {
#if (defined(__cplusplus) && (__cplusplus >= 201703L)) && !defined(__GNUC__)
    return std::reduce(first, last, init);
#elif (defined(__cplusplus) && (__cplusplus >= 201103L))
    return std::accumulate(first, last, init);
#else
    // unreachable, but preserved as reference implementation
    T r(0);
    for (I i=first; i!=last; ++i) {
        r += *i;
    }
    return r;
#endif
}

// These headers are busted with NVCC and GCC 5.4.0
// The <future> header is busted with Cray C++ 8.6.1.
#if !defined(__NVCC__) && !defined(_CRAYC)
#include <thread>
#include <future>
#endif

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
# if (_OPENMP >= 201300)
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

#if defined(__INTEL_COMPILER)
# define PRAGMA_SIMD PRAGMA(vector) PRAGMA(ivdep)
// According to https://github.com/LLNL/RAJA/pull/310, this improves lambda performance
# define PRAGMA_INLINE PRAGMA(forceinline recursive)
#elif defined(__GNUC__) && defined(__GNUC_MINOR__) && ( ( (__GNUC__ == 4) && (__GNUC_MINOR__ == 9) ) || (__GNUC__ >= 5) )
# define PRAGMA_SIMD PRAGMA(GCC ivdep)
# define PRAGMA_INLINE PRAGMA(inline)
#elif defined(__clang__)
# define PRAGMA_SIMD PRAGMA(clang loop vectorize(assume_safety))
# define PRAGMA_INLINE
#else
# define PRAGMA_SIMD
# define PRAGMA_INLINE
#endif

#ifdef USE_TBB
# include <tbb/tbb.h>
# include <tbb/parallel_for.h>
# include <tbb/blocked_range.h>
# if ( PRK_TBB_PARTITIONER == 1)
//#  warning STATIC
   tbb::static_partitioner tbb_partitioner;
# elif ( PRK_TBB_PARTITIONER == 2)
//#  warning AFFINITY
   tbb::affinity_partitioner tbb_partitioner;
# elif ( PRK_TBB_PARTITIONER == 3)
//#  warning SIMPLE
   tbb::simple_partitioner tbb_partitioner;
# else
//#  warning AUTO
   tbb::auto_partitioner tbb_partitioner;
# endif
#endif

#if defined(USE_BOOST)
# include "boost/range/irange.hpp"
#endif

#if defined(USE_BOOST_COMPUTE)
# include "boost/compute.hpp"
# include "boost/compute/container/valarray.hpp"
#endif

#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1800)
#define USE_INTEL_PSTL
#endif

#ifdef USE_PSTL
# ifdef USE_INTEL_PSTL
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
# include <Kokkos_Core.hpp>
# include <Kokkos_Concepts.hpp>
# include <Kokkos_MemoryTraits.hpp>
#endif

#ifdef USE_RAJA
# define RAJA_ENABLE_NESTED 1
# include "RAJA/RAJA.hpp"
#endif

#ifdef USE_SYCL
# include "CL/sycl.hpp"
#endif

#ifdef USE_OCCA
# include "occa.hpp"
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

    template <class T1, class T2>
    static inline auto divceil(T1 numerator, T2 denominator) -> decltype(numerator / denominator) {
        return ( numerator / denominator + (numerator % denominator > 0) );
    }

} // namespace prk

#endif /* PRK_UTIL_H */
