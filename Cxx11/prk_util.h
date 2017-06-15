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

#include <numeric>
#include <algorithm>

#if !(defined(__cplusplus) && (__cplusplus >= 201103L))
#error You need a C++11 compiler.
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __cilk
#include <cilk/cilk.h>
#endif

#ifdef USE_TBB
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
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

    /* This function is separate from prk_malloc() because
     * we need it when calling prk_shmem_align(..)           */
    static inline int get_alignment(void)
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

} // namespace prk
