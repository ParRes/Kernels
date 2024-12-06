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

#ifndef PRK_SIMD_H
#define PRK_SIMD_H

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

#endif /* PRK_SIMD_H */
