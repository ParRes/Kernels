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

#ifndef PRK_PSTL_H
#define PRK_PSTL_H

#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1800)
#define USE_INTEL_PSTL
#endif

#if defined(USE_PSTL)
# if defined(__GNUC__) && (__GNUC__ >= 9)
#  include <execution>
#  include <algorithm>
#  include <numeric>
//#  include <memory>
namespace exec = __pstl::execution;
# elif defined(USE_INTEL_PSTL)
#  include <pstl/execution>
#  include <pstl/algorithm>
#  include <pstl/numeric>
//#  include <pstl/memory>
namespace exec = std::execution;
# elif defined(__GNUC__) && defined(__GNUC_MINOR__) && \
       ( (__GNUC__ >= 8) || (__GNUC__ == 7) && (__GNUC_MINOR__ >= 2) )
#  include <parallel/algorithm>
#  include <parallel/numeric>
# endif
#endif

#endif /* PRK_PSTL_H */
