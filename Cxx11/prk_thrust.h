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

#ifndef PRK_THRUST_H
#define PRK_THRUST_H

#ifdef USE_THRUST
# ifdef __NVCC__
#  include <thrust/device_vector.h>
# elif defined(_OPENMP)
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_OMP
#  include <thrust/system/omp/execution_policy.h>
//#  include <thrust/system/omp/vector.h>
#else
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#  include <thrust/system/omp/execution_policy.h>
//#  include <thrust/system/cpp/vector.h>
# endif
# include <thrust/host_vector.h>
# include <thrust/fill.h>
# include <thrust/transform.h>
# include <thrust/transform_reduce.h>
# include <thrust/functional.h>
# include <thrust/sequence.h>
# include <thrust/for_each.h>
# include <thrust/iterator/counting_iterator.h>
# include <thrust/execution_policy.h>
#endif

#endif /* PRK_THRUST_H */
