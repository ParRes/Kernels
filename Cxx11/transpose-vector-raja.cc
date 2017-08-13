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

  int iterations, order;
  std::string use_for="seq", use_permute="no";
  auto use_simd=true, use_nested=true, use_tiled=false;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <matrix order> <nested={y,n} for={seq,omp,tbb} simd={y,n}>";
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
      for (int i=3; i<argc; ++i) {
          //std::cout << "argv[" << i << "] = " << argv[i] << "\n";
          auto s  = std::string(argv[i]);
          auto pf = s.find("for=");
          if (pf != std::string::npos) {
              auto sf = s.substr(4,s.size());
              //std::cout << pf << "," << sf << "\n";
              if (sf=="omp")    use_for="omp";
              if (sf=="openmp") use_for="omp";
              if (sf=="tbb")    use_for="tbb";
          }
          auto ps = s.find("simd=");
          if (ps != std::string::npos) {
              auto ss = s.substr(5,s.size());
              //std::cout << ps << "," << ss[0] << "\n";
              if (ss=="n")  use_simd=false;
              if (ss=="np") use_simd=false;
          }
          auto pn = s.find("nested=");
          if (pn != std::string::npos) {
              auto sn = s.substr(7,s.size());
              //std::cout << pn << "," << sn[0] << "\n";
              if (sn=="n")  use_nested=false;
              if (sn=="no") use_nested=false;
          }
          auto pt = s.find("tiled=");
          if (pt != std::string::npos) {
              auto st = s.substr(6,s.size());
              //std::cout << pt << "," << st[0] << "\n";
              if (st=="y")   use_tiled=true;
              if (st=="yes") use_tiled=true;
          }
          auto pp = s.find("permute=");
          if (pp != std::string::npos) {
              auto sp = s.substr(8,s.size());
              //std::cout << pp << "," << pp[0] << "\n";
              if (sp=="ij") use_permute="ij";
              if (sp=="ji") use_permute="ji";
          }
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations  = " << iterations << std::endl;
  std::cout << "Matrix order          = " << order << std::endl;
  std::cout << "Tile size             = " << tile_size << "(compile-time constant, unlike other impls)" << std::endl;
  std::string         for_name = "Sequential";
  if (use_for=="omp") for_name = "OpenMP";
  if (use_for=="tbb") for_name = "TBB";
  std::cout << "RAJA threading        = " << for_name << std::endl;
  std::cout << "RAJA forallN          = " << (use_nested ? "yes" : "no") << std::endl;
  std::cout << "RAJA use simd         = " << (use_simd ? "yes" : "no") << std::endl;
  std::cout << "RAJA use tiling       = " << (use_tiled ? "yes" : "no") << std::endl;
  std::cout << "RAJA use permute      = " << use_permute << std::endl;

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
    if (use_for=="seq") {
      if (use_nested) {
        if (use_simd) {
            RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::simd_exec>>>
                    ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                      [=,&A,&B](RAJA::Index_type i, RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
            });
        } else {
            RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>>>
                    ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                      [=,&A,&B](RAJA::Index_type i, RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
            });
        }
      } else /* !use_nested */ {
        if (use_simd) {
            RAJA::forall<RAJA::seq_exec>(RAJA::Index_type(0), RAJA::Index_type(order), [=,&A,&B](RAJA::Index_type i) {
                RAJA::forall<RAJA::simd_exec>(RAJA::Index_type(0), RAJA::Index_type(order), [=,&A,&B](RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
                });
            });
        } else {
            RAJA::forall<RAJA::seq_exec>(RAJA::Index_type(0), RAJA::Index_type(order), [=,&A,&B](RAJA::Index_type i) {
                RAJA::forall<RAJA::seq_exec>(RAJA::Index_type(0), RAJA::Index_type(order), [=,&A,&B](RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
                });
            });
        }
      }
    }
#ifdef RAJA_ENABLE_OPENMP
    else if (use_for=="omp") {
      if (use_nested) {
        if (use_simd) {
          RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::simd_exec>>>
                  ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                    [=,&A,&B](RAJA::Index_type i, RAJA::Index_type j) {
                  B[i*order+j] += A[j*order+i];
                  A[j*order+i] += 1.0;
          });
        } else {
            RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::seq_exec>>>
                    ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                      [=,&A,&B](RAJA::Index_type i, RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
            });
        }
      } else /* !use_nested */ {
        if (use_simd) {
          RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::Index_type(0), RAJA::Index_type(order), [=,&A,&B](RAJA::Index_type i) {
              RAJA::forall<RAJA::simd_exec>(RAJA::Index_type(0), RAJA::Index_type(order), [=,&A,&B](RAJA::Index_type j) {
                  B[i*order+j] += A[j*order+i];
                  A[j*order+i] += 1.0;
              });
          });
        } else {
          RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::Index_type(0), RAJA::Index_type(order), [=,&A,&B](RAJA::Index_type i) {
              RAJA::forall<RAJA::seq_exec>(RAJA::Index_type(0), RAJA::Index_type(order), [=,&A,&B](RAJA::Index_type j) {
                  B[i*order+j] += A[j*order+i];
                  A[j*order+i] += 1.0;
              });
          });
        }
      }
    }
#else
    std::cout << "You are trying to use OpenMP but RAJA does not support it!" << std::endl;
    std::abort();
#endif
#ifdef RAJA_ENABLE_TBB
    else if (use_for=="tbb") {
      if (use_nested) {
        if (use_tiled) {
          if (use_simd) {
            RAJA::forallN< RAJA::NestedPolicy< RAJA::ExecList<RAJA::tbb_for_exec, RAJA::simd_exec>,
                                               RAJA::Tile< RAJA::TileList<RAJA::tile_fixed<tile_size>,
                                                                          RAJA::tile_fixed<tile_size>>>>>
                      ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                        [=,&A,&B](RAJA::Index_type i, RAJA::Index_type j) {
                      B[i*order+j] += A[j*order+i];
                      A[j*order+i] += 1.0;
              });
          } else {
            RAJA::forallN< RAJA::NestedPolicy< RAJA::ExecList<RAJA::tbb_for_exec, RAJA::seq_exec>,
                                               RAJA::Tile< RAJA::TileList<RAJA::tile_fixed<tile_size>,
                                                                          RAJA::tile_fixed<tile_size>>>>>
                      ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                        [=,&A,&B](RAJA::Index_type i, RAJA::Index_type j) {
                      B[i*order+j] += A[j*order+i];
                      A[j*order+i] += 1.0;
              });
          }
        } else {
          if (use_simd) {
            RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::tbb_for_exec, RAJA::simd_exec>>>
                    ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                      [=,&A,&B](RAJA::Index_type i, RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
            });
          } else {
            RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::tbb_for_exec, RAJA::seq_exec>>>
                    ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                      [=,&A,&B](RAJA::Index_type i, RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
            });
          }
        }
      } else /* !use_nested */ {
        if (use_simd) {
            RAJA::forall<RAJA::tbb_for_exec>(RAJA::Index_type(0), RAJA::Index_type(order), [=,&A,&B](RAJA::Index_type i) {
                RAJA::forall<RAJA::simd_exec>(RAJA::Index_type(0), RAJA::Index_type(order), [=,&A,&B](RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
                });
            });
        } else {
            RAJA::forall<RAJA::tbb_for_exec>(RAJA::Index_type(0), RAJA::Index_type(order), [=,&A,&B](RAJA::Index_type i) {
                RAJA::forall<RAJA::seq_exec>(RAJA::Index_type(0), RAJA::Index_type(order), [=,&A,&B](RAJA::Index_type j) {
                    B[i*order+j] += A[j*order+i];
                    A[j*order+i] += 1.0;
                });
            });
        }
      }
    }
#else
    std::cout << "You are trying to use TBB but RAJA does not support it!" << std::endl;
    std::abort();
#endif
  }
  trans_time = prk::wtime() - trans_time;

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

#if defined(RAJA_ENABLE_OPENMP) && !defined(RAJA_ENABLE_TBB)
  typedef RAJA::omp_reduce reduce_policy;
  typedef RAJA::omp_parallel_for_exec loop_policy;
#else
  typedef RAJA::seq_reduce reduce_policy;
  typedef RAJA::seq_exec loop_policy;
#endif
  RAJA::ReduceSum<reduce_policy, double> abserr(0.0);
  RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<loop_policy, RAJA::seq_exec>>>
          ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
            [=,&A,&B](RAJA::Index_type i, RAJA::Index_type j) {
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


#if 0
        RAJA::forallN< RAJA::NestedPolicy< RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>,
                                           RAJA::Permute<RAJA::PERM_JI> > >
                ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                  [=,&A,&B](RAJA::Index_type i, RAJA::Index_type j) {
                B[i*order+j] += A[j*order+i];
                A[j*order+i] += 1.0;
        });
        RAJA::forallN< RAJA::NestedPolicy< RAJA::ExecList<RAJA::simd_exec, RAJA::simd_exec>,
                                           RAJA::Tile< RAJA::TileList<RAJA::tile_fixed<tile_size>, RAJA::tile_fixed<tile_size>>,
                                                       RAJA::Permute<RAJA::PERM_IJ> > > >
                ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                  [=,&A,&B](RAJA::Index_type i, RAJA::Index_type j) {
                B[i*order+j] += A[j*order+i];
                A[j*order+i] += 1.0;
        });
        RAJA::forallN< RAJA::NestedPolicy< RAJA::ExecList<RAJA::simd_exec, RAJA::simd_exec>,
                                           RAJA::Tile< RAJA::TileList<RAJA::tile_fixed<tile_size>, RAJA::tile_fixed<tile_size>>,
                                                       RAJA::Permute<RAJA::PERM_JI> > > >
                ( RAJA::RangeSegment(0, order), RAJA::RangeSegment(0, order),
                  [=,&A,&B](RAJA::Index_type i, RAJA::Index_type j) {
                B[i*order+j] += A[j*order+i];
                A[j*order+i] += 1.0;
        });
#endif
