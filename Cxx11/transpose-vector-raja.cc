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

//////////////////////////////////////////////////////////////////////
///
/// NAME:    transpose
///
/// PURPOSE: This program measures the time for the transpose of a
///          column-major stored matrix into a row-major stored matrix.
///
/// USAGE:   Program input is the matrix order and the number of times to
///          repeat the operation:
///
///          transpose <matrix_size> <# iterations> <variant>
///
///          The output consists of diagnostics to make sure the
///          transpose worked and timing statistics.
///
/// HISTORY: Written by  Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, February 2016 and May 2017.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"

const int tile_size = 32;

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/RAJA Matrix transpose: B = A^T" << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  int order;
  int variant;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <matrix order> <variant>";
      }

      // number of times to do the transpose
      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      // order of a the matrix
      order = std::atoi(argv[2]);
      if (order <= 0) {
        throw "ERROR: Matrix Order must be greater than 0";
      } else if (order > std::floor(std::sqrt(INT_MAX))) {
        throw "ERROR: matrix dimension too large - overflow risk";
      }

      // RAJA implementation variant
      variant  = (argc > 3) ? std::atoi(argv[3]) : 1;
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  // Make sure to update these with the implementations below...
  std::string vname;
  switch (variant) {
      case 1: vname = "forall(seq_exec),forall(seq_exec)"; break;
      case 2: vname = "forallN(seq_exec,seq_exec,PERM_IJ)"; break;
      case 3: vname = "forallN(seq_exec,seq_exec,PERM_JI)"; break;
      case 4: vname = "forallN(simd_exec,simd_exec,PERM_IJ)"; break;
      case 5: vname = "forallN(simd_exec,simd_exec,PERM_JI)"; break;
      case 6: vname = "forallN(simd_exec,simd_exec,tiled,PERM_IJ)"; break;
      case 7: vname = "forallN(simd_exec,simd_exec,tiled,PERM_JI)"; break;
#ifdef RAJA_ENABLE_OPENMP
      case 10: vname = "forall(omp_parallel_for_exec),forall(simd_exec)"; break;
      case 11: vname = "forallN(omp_parallel_for_exec,simd_exec)"; break;
      case 12: vname = "forallN(omp_parallel_for_exec,simd_exec,PERM_IJ)"; break;
      case 13: vname = "forallN(omp_parallel_for_exec,simd_exec,PERM_JI)"; break;
      case 14: vname = "forallN(omp_parallel_for_exec,simd_exec,tiled,PERM_IJ)"; break;
      case 15: vname = "forallN(omp_parallel_for_exec,simd_exec,tiled,PERM_JI)"; break;
#endif
      default: std::cout << "Invalid RAJA variant number (" << variant << ")" << std::endl; return 0; break;
  }

  std::cout << "Number of iterations  = " << iterations << std::endl;
  std::cout << "Matrix order          = " << order << std::endl;
  std::cout << "RAJA variant          = " << vname << std::endl;
  std::cout << "Tile size             = " << tile_size << "(compile-time constant, unlike other impls)" << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

  std::vector<double> A;
  std::vector<double> B;
  A.resize(order*order);
  B.resize(order*order,0.0);
  // fill A with the sequence 0 to order^2-1 as doubles
  std::iota(A.begin(), A.end(), 0.0);

  auto trans_time = 0.0;

  for (auto iter = 0; iter<=iterations; iter++) {

    if (iter==1) trans_time = prk::wtime();

    // transpose
    // If a new variant is added, it must be added above to the vname case-switch...
    switch (variant) {
        case 1:
            RAJA::forall<RAJA::seq_exec>(RAJA::Index_type(0), RAJA::Index_type(order), [&](RAJA::Index_type i) {
                RAJA::forall<RAJA::seq_exec>(RAJA::Index_type(0), RAJA::Index_type(order), [&](RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
                });
            });
            break;
        case 2:
            RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>,
                                             RAJA::Permute<RAJA::PERM_IJ>>>
                    ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                      [&](RAJA::Index_type i, RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
            });
            break;
        case 3:
            RAJA::forallN< RAJA::NestedPolicy< RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>,
                                               RAJA::Permute<RAJA::PERM_JI> > >
                    ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                      [&](RAJA::Index_type i, RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
            });
            break;
        case 4:
            RAJA::forallN< RAJA::NestedPolicy< RAJA::ExecList<RAJA::simd_exec, RAJA::simd_exec>,
                                               RAJA::Permute<RAJA::PERM_IJ> > >
                    ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                      [&](RAJA::Index_type i, RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
            });
            break;
        case 5:
            RAJA::forallN< RAJA::NestedPolicy< RAJA::ExecList<RAJA::simd_exec, RAJA::simd_exec>,
                                               RAJA::Permute<RAJA::PERM_JI> > >
                    ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                      [&](RAJA::Index_type i, RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
            });
            break;
        case 6:
            RAJA::forallN< RAJA::NestedPolicy< RAJA::ExecList<RAJA::simd_exec, RAJA::simd_exec>,
                                               RAJA::Tile< RAJA::TileList<RAJA::tile_fixed<tile_size>, RAJA::tile_fixed<tile_size>>,
                                                           RAJA::Permute<RAJA::PERM_IJ> > > >
                    ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                      [&](RAJA::Index_type i, RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
            });
            break;
        case 7:
            RAJA::forallN< RAJA::NestedPolicy< RAJA::ExecList<RAJA::simd_exec, RAJA::simd_exec>,
                                               RAJA::Tile< RAJA::TileList<RAJA::tile_fixed<tile_size>, RAJA::tile_fixed<tile_size>>,
                                                           RAJA::Permute<RAJA::PERM_JI> > > >
                    ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                      [&](RAJA::Index_type i, RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
            });
            break;
#ifdef RAJA_ENABLE_OPENMP
        case 10:
            RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::Index_type(0), RAJA::Index_type(order), [&](RAJA::Index_type i) {
                RAJA::forall<RAJA::simd_exec>(RAJA::Index_type(0), RAJA::Index_type(order), [&](RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
                });
            });
            break;
        case 11:
            RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::simd_exec>>>
                    ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                      [&](RAJA::Index_type i, RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
            });
            break;
        case 12:
            RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::simd_exec>,RAJA::Permute<RAJA::PERM_IJ>>>
                    ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                      [&](RAJA::Index_type i, RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
            });
            break;
        case 13:
            RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::simd_exec>,RAJA::Permute<RAJA::PERM_JI>>>
                    ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                      [&](RAJA::Index_type i, RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
            });
            break;
        case 14:
            RAJA::forallN< RAJA::NestedPolicy< RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::simd_exec>,
                                               RAJA::Tile< RAJA::TileList<RAJA::tile_fixed<tile_size>, RAJA::tile_fixed<tile_size>>,
                                                           RAJA::Permute<RAJA::PERM_IJ> > > >
                    ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                      [&](RAJA::Index_type i, RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
            });
            break;
        case 15:
            RAJA::forallN< RAJA::NestedPolicy< RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::simd_exec>,
                                               RAJA::Tile< RAJA::TileList<RAJA::tile_fixed<tile_size>, RAJA::tile_fixed<tile_size>>,
                                                           RAJA::Permute<RAJA::PERM_JI> > > >
                    ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                      [&](RAJA::Index_type i, RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
            });
            break;
#endif
    }
  }
  trans_time = prk::wtime() - trans_time;

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

#ifdef RAJA_ENABLE_OPENMP
  typedef RAJA::omp_reduce reduce_policy;
  typedef RAJA::omp_parallel_for_exec loop_policy;
#else
  typedef RAJA::seq_reduce reduce_policy;
  typedef RAJA::seq_exec loop_policy;
#endif
  RAJA::ReduceSum<reduce_policy, double> abserr(0.0);
  RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<loop_policy, RAJA::seq_exec>>>
          ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
            [&](RAJA::Index_type i, RAJA::Index_type j) {
      const int ij = i*order+j;
      const int ji = j*order+i;
      const auto addit = (iterations+1.) * (iterations/2.);
      const double reference = static_cast<double>(ij)*(1.+iterations)+addit;
      abserr += std::fabs(B[ji] - reference);
  });

#ifdef VERBOSE
  std::cout << "Sum of absolute differences: " << abserr << std::endl;
#endif

  const auto epsilon = 1.0e-8;
  if (abserr < epsilon) {
    std::cout << "Solution validates" << std::endl;
    auto avgtime = trans_time/iterations;
    auto bytes = (size_t)order * (size_t)order * sizeof(double);
    std::cout << "Rate (MB/s): " << 1.0e-6 * (2L*bytes)/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  } else {
    std::cout << "ERROR: Aggregate squared error " << abserr
              << " exceeds threshold " << epsilon << std::endl;
    return 1;
  }

  return 0;
}


