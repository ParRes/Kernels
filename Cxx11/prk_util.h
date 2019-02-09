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

#ifndef PRK_UTIL_H
#define PRK_UTIL_H

#include <cstdio>
#include <cstdlib> // atoi, getenv
#include <cstdint>
#include <climits>
#include <cmath>   // abs, fabs
#include <cassert>

// Test standard library _after_ standard headers have been included...
#if !defined(__NVCC__) && !defined(__PGI) && (defined(__GLIBCXX__) || defined(_GLIBCXX_RELEASE) ) && !defined(_GLIBCXX_USE_CXX11_ABI)
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

#include "prk_simd.h"

#ifdef USE_RANGES
# include "prk_ranges.h"
#endif

#ifdef USE_OPENMP
# include "prk_openmp.h"
#endif

#define RESTRICT __restrict__

#if (defined(__cplusplus) && (__cplusplus >= 201703L))
#define PRK_UNUSED [[maybe_unused]]
#else
#define PRK_UNUSED
#endif

namespace prk {

    template<class I, class T>
    const T reduce(I first, I last, T init) {
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

    static inline double wtime(void)
    {
#if defined(USE_OPENMP) && defined(_OPENMP)
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
