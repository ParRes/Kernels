//
// icpx -std=c++17 -pthread -g -O3 -xHOST -fiopenmp -fopenmp-targets=spir64 nstream-openmp-target-bug.cc
//
#include <cstdio>
#include <cstdlib> // atoi, getenv
#include <cstdint>
#include <cfloat>  // FLT_MIN
#include <climits>
#include <cmath>   // abs, fabs
#include <cassert>

#include <string>
#include <iostream>
#include <iomanip> // std::setprecision
#include <exception>
#include <list>
#include <vector>

#include <chrono>
#include <random>
#include <typeinfo>
#include <array>
#include <atomic>
#include <numeric>
#include <algorithm>

#define PRAGMA(x) _Pragma(#x)

#if defined(__INTEL_COMPILER)
# define PRAGMA_SIMD PRAGMA(vector) PRAGMA(ivdep)
// According to https://github.com/LLNL/RAJA/pull/310, this improves lambda performance
# define PRAGMA_INLINE PRAGMA(forceinline recursive)
#elif defined(__PGI)
# define PRAGMA_SIMD PRAGMA(vector) PRAGMA(ivdep)
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

#include <omp.h>

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

    bool parse_boolean(const std::string & s)
    {
        if (s=="t" || s=="T" || s=="y" || s=="Y" || s=="1") {
            return true;
        } else {
            return false;
        }

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

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << std::endl;
  std::cout << "C++11/OpenMP TARGET STREAM triad: A = B + scalar * C" << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations, offset;
  size_t length;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <vector length>";
      }

      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      length = std::atol(argv[2]);
      if (length <= 0) {
        throw "ERROR: vector length must be positive";
      }

      offset = (argc>3) ? std::atoi(argv[3]) : 0;
      if (length <= 0) {
        throw "ERROR: offset must be nonnegative";
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of threads    = " << omp_get_max_threads() << std::endl;
  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Vector length        = " << length << std::endl;
  std::cout << "Offset               = " << offset << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  auto nstream_time = 0.0;

  double * RESTRICT A = new double[length];
  double * RESTRICT B = new double[length];
  double * RESTRICT C = new double[length];

  double scalar = 3.0;

  // HOST
  {
    for (size_t i=0; i<length; i++) {
      A[i] = 0.0;
      B[i] = 2.0;
      C[i] = 2.0;
    }
  }

  // DEVICE
#pragma omp target data map(tofrom: A[0:length]) map(to: B[0:length], C[0:length])
  {
    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) nstream_time = prk::wtime();

#pragma omp target teams distribute parallel for // simd schedule(static,1)
      for (size_t i=0; i<length; i++) {
          A[i] += B[i] + scalar * C[i];
      }
    }
    nstream_time = prk::wtime() - nstream_time;
  }

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  double ar(0);
  double br(2);
  double cr(2);
  for (int i=0; i<=iterations; i++) {
      ar += br + scalar * cr;
  }

  ar *= length;

  double asum(0);
  for (size_t i=0; i<length; i++) {
      asum += std::fabs(A[i]);
  }

  double epsilon=1.e-8;
  if (std::fabs(ar-asum)/asum > epsilon) {
      std::cout << "Failed Validation on output array\n"
                << std::setprecision(16)
                << "       Expected checksum: " << ar << "\n"
                << "       Observed checksum: " << asum << std::endl;
      std::cout << "ERROR: solution did not validate" << std::endl;
      return 1;
  } else {
      std::cout << "Solution validates" << std::endl;
      double avgtime = nstream_time/iterations;
      double nbytes = 4.0 * length * sizeof(double);
      std::cout << "Rate (MB/s): " << 1.e-6*nbytes/avgtime
                << " Avg time (s): " << avgtime << std::endl;
  }

  return 0;
}


