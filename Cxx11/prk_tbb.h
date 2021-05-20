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

#ifndef PRK_TBB_H
#define PRK_TBB_H

//#include <tbb/tbb.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/flow_graph.h>
#include <tbb/parallel_for_each.h>
#if TBB_INTERFACE_VERSION <= 12000
#   include <tbb/parallel_do.h>
#endif

#if ( PRK_TBB_PARTITIONER == 1)
   tbb::static_partitioner tbb_partitioner;
   const std::string tbb_partitioner_name("static_partitioner");
#elif ( PRK_TBB_PARTITIONER == 2)
   tbb::affinity_partitioner tbb_partitioner;
   const std::string tbb_partitioner_name("affinity_partitioner");
#elif ( PRK_TBB_PARTITIONER == 3)
   tbb::simple_partitioner tbb_partitioner;
   const std::string tbb_partitioner_name("simple_partitioner");
#else
   tbb::auto_partitioner tbb_partitioner;
   const std::string tbb_partitioner_name("auto_partitioner");
#endif

#endif /* PRK_TBB_H */
