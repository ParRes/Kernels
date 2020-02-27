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
#if !defined(__NVCC__) && !defined(__PGI) && !defined(__ibmxl__) && (defined(__GLIBCXX__) || defined(_GLIBCXX_RELEASE) ) && !defined(_GLIBCXX_USE_CXX11_ABI)
# warning You are using an ancient version GNU libstdc++.  Either upgrade your GCC or tell ICC to use a newer version via the -gxx-name= option.
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
//#include <valarray>

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

    int get_alignment(void)
    {
        /* a := alignment */
#ifdef PRK_ALIGNMENT
        int a = PRK_ALIGNMENT;
#else
        const char* temp = std::getenv("PRK_ALIGNMENT");
        int a = (temp!=nullptr) ? std::atoi(temp) : 64;
        if (a < 8) a = 8;
        assert( (a & (~a+1)) == a ); /* is power of 2? */
#endif
        return a;
    }

#if defined(__INTEL_COMPILER)

    template <typename T>
    T * malloc(size_t n)
    {
        const int alignment = prk::get_alignment();
        const size_t bytes = n * sizeof(T);
        return (T*)_mm_malloc( bytes, alignment);
    }

    template <typename T>
    void free(T * p)
    {
        _mm_free(p);
        p = nullptr;
    }

#else // !__INTEL_COMPILER

    template <typename T>
    T * malloc(size_t n)
    {
        const int alignment = prk::get_alignment();
        const size_t bytes = n * sizeof(T);

        // We cannot use C11 aligned_alloc on Mac.
        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=69680 */
        // GCC claims to be C11 without knowing if glibc is compliant...
#if !defined(__GNUC__) && \
    !defined(__APPLE__) && \
     defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L) && 0 \

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

        const size_t padded = bytes;
        const size_t excess = bytes % alignment;
        if (excess>0) padded += (alignment - excess);
        return aligned_alloc(alignment,padded);

#else

        T * ptr = nullptr;
        const int ret = posix_memalign((void**)&ptr,alignment,bytes);
        if (ret!=0) ptr = nullptr;
        return ptr;

#endif

    }

    template <typename T>
    void free(T * p)
    {
        std::free(p);
        p = nullptr;
    }

#endif // __INTEL_COMPILER

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

    template <typename T>
    class vector {

        private:
            T * data_;
            size_t size_;

        public:

            vector(size_t n) {
                //this->data_ = new T[n];
                this->data_ = prk::malloc<T>(n);
                this->size_ = n;
            }

            vector(size_t n, T v) {
                //this->data_ = new T[n];
                this->data_ = prk::malloc<T>(n);
                for (size_t i=0; i<n; ++i) this->data_[i] = v;
                this->size_ = n;
            }

            ~vector() {
                //delete[] this->data_;
                prk::free<T>(this->data_);
            }

            void operator~() {
                this->~vector();
            }

            T * data() {
                return this->data_;
            }

            size_t size() {
                return this->size_;
            }

#if 0
            T const & operator[] (int n) const {
                return this->data_[n];
            }

            T & operator[] (int n) {
                return this->data_[n];
            }
#endif

            T const & operator[] (size_t n) const {
                return this->data_[n];
            }

            T & operator[] (size_t n) {
                return this->data_[n];
            }

            T * begin() {
                return &(this->data_[0]);
            }

            T * end() {
                return &(this->data_[this->size_]);
            }

#if 0
            T & begin() {
                return this->data_[0];
            }

            T & end() {
                return this->data_[this->size_];
            }
#endif
    };

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

    template<typename T>
    T * alloc(size_t bytes)
    {
        int alignment = ::prk::get_alignment();
#if defined(__INTEL_COMPILER)
        return (void*)_mm_malloc(bytes,alignment);
#else
        T * ptr = nullptr;
        int ret = posix_memalign((void**)&ptr,alignment,bytes);
        if (ret!=0) ptr = NULL;
        return ptr;
#endif

    }

    template<typename T>
    void dealloc(T * p)
    {
#if defined(__INTEL_COMPILER)
        _mm_free((void*)p);
#else
        free((void*)p);
#endif
    }

} // namespace prk

#endif /* PRK_UTIL_H */
