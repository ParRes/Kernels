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

#ifndef PRK_RANGES_H
#define PRK_RANGES_H

#if defined(USE_RANGES)
# if defined(USE_BOOST_IRANGE)
#  include "boost/range/irange.hpp"
# elif defined(USE_RANGES_TS)
#  include "range/v3/view/iota.hpp"
#  include "range/v3/view/slice.hpp"
#  include "range/v3/view/stride.hpp"
# else
#  error You have not provided a version of ranges to use.
# endif
#endif

namespace prk {

    template <class S, class E>
    auto range(S start, E end) {
#if defined(USE_BOOST_IRANGE)
        return boost::irange(static_cast<decltype(end)>(start), end);
#elif defined(USE_RANGES_TS)
        return ranges::view::iota(static_cast<decltype(end)>(start), end);
#endif
    }

    template <class S, class E, class B>
    auto range(S start, E end, B blocking) {
#if defined(USE_BOOST_IRANGE)
        return boost::irange(static_cast<decltype(end)>(start), end, static_cast<decltype(end)>(blocking) );
#elif defined(USE_RANGES_TS)
        // NOTE:
        // iota(s) | slice(s,e) | stride(b)  is faster than
        // iota(s,e) | stride(b) for some reason.
        return ranges::view::iota(static_cast<decltype(end)>(start)) |
               ranges::view::slice(static_cast<decltype(end)>(start), end) |
               ranges::view::stride(static_cast<decltype(end)>(blocking));
#endif
    }

} // namespace prk

#endif /* PRK_RANGES_H */
