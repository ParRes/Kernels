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
#include "prk_raja.h"

const int tile_size = 32;

typedef RAJA::Index_type indx;
typedef RAJA::RangeSegment range;
typedef RAJA::TileList<RAJA::tile_fixed<tile_size>, RAJA::tile_fixed<tile_size>> tile;
typedef RAJA::Tile<tile> tiling;
typedef RAJA::Tile<tile,RAJA::Permute<RAJA::PERM_IJ>> tiling_ij;
typedef RAJA::Tile<tile,RAJA::Permute<RAJA::PERM_JI>> tiling_ji;

typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::simd_exec>>         seq_for_simd;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>>          seq_for_seq;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>, tiling>  seq_for_seq_tiled;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::simd_exec>, tiling> seq_for_simd_tiled;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>, tiling_ij>  seq_for_seq_tiled_ij;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::simd_exec>, tiling_ij> seq_for_simd_tiled_ij;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>, tiling_ji>  seq_for_seq_tiled_ji;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::simd_exec>, tiling_ji> seq_for_simd_tiled_ji;
#ifdef RAJA_ENABLE_OPENMP
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::seq_exec>>          omp_for_seq;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::simd_exec>>         omp_for_simd;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::seq_exec>, tiling>  omp_for_seq_tiled;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::simd_exec>, tiling> omp_for_simd_tiled;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::seq_exec>, tiling_ij>  omp_for_seq_tiled_ij;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::simd_exec>, tiling_ij> omp_for_simd_tiled_ij;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::seq_exec>, tiling_ji>  omp_for_seq_tiled_ji;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::simd_exec>, tiling_ji> omp_for_simd_tiled_ji;
#endif
#ifdef RAJA_ENABLE_TBB
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::tbb_for_exec, RAJA::seq_exec>>             tbb_for_seq;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::tbb_for_exec, RAJA::simd_exec>>            tbb_for_simd;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::tbb_for_exec, RAJA::seq_exec>, tiling>     tbb_for_seq_tiled;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::tbb_for_exec, RAJA::simd_exec>, tiling>    tbb_for_simd_tiled;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::tbb_for_exec, RAJA::seq_exec>, tiling_ij>  tbb_for_seq_tiled_ij;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::tbb_for_exec, RAJA::simd_exec>, tiling_ij> tbb_for_simd_tiled_ij;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::tbb_for_exec, RAJA::seq_exec>, tiling_ji>  tbb_for_seq_tiled_ji;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::tbb_for_exec, RAJA::simd_exec>, tiling_ji> tbb_for_simd_tiled_ji;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::tbb_for_dynamic, RAJA::seq_exec>>          tbb_for_dynamic_seq;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::tbb_for_dynamic, RAJA::simd_exec>>         tbb_for_dynamic_simd;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::tbb_for_dynamic, RAJA::seq_exec>, tiling>  tbb_for_dynamic_seq_tiled;
typedef RAJA::NestedPolicy<RAJA::ExecList<RAJA::tbb_for_dynamic, RAJA::simd_exec>, tiling> tbb_for_dynamic_simd_tiled;
#endif

template <typename exec_policy, typename LoopBody>
void Lambda(int order, LoopBody body)
{
    RAJA::forallN<exec_policy>( range(0, order), range(0, order), body);
}

template <typename exec_policy>
void Initialize(int order, std::vector<double> & A, std::vector<double> & B)
{
    Lambda<exec_policy>(order, [=,&A,&B](int i, int j) {
        A[i*order+j] = static_cast<double>(i*order+j);
        B[i*order+j] = 0.0;
    });
}

template <typename exec_policy>
void Transpose(int order, std::vector<double> & A, std::vector<double> & B)
{
    Lambda<exec_policy>(order, [=,&A,&B](int i, int j) {
        B[i*order+j] += A[j*order+i];
        A[j*order+i] += 1.0;
    });
}

template <typename outer_policy, typename inner_policy>
void Initialize(int order, std::vector<double> & A, std::vector<double> & B)
{
    RAJA::forall<outer_policy>(0, order, [=,&A,&B](int i) {
      RAJA::forall<inner_policy>(0, order, [=,&A,&B](int j) {
        A[i*order+j] = static_cast<double>(i*order+j);
        B[i*order+j] = 0.0;
      });
    });
}

template <typename outer_policy, typename inner_policy>
void Transpose(int order, std::vector<double> & A, std::vector<double> & B)
{
    RAJA::forall<outer_policy>(0, order, [=,&A,&B](int i) {
      RAJA::forall<inner_policy>(0, order, [=,&A,&B](int j) {
        B[i*order+j] += A[j*order+i];
        A[j*order+i] += 1.0;
      });
    });
}

template <typename loop_policy, typename reduce_policy>
double Error(int iterations, int order, std::vector<double> & B)
{
      RAJA::ReduceSum<reduce_policy, double> abserr(0.0);
      typedef RAJA::NestedPolicy<RAJA::ExecList<loop_policy, RAJA::seq_exec>> exec_policy;
      RAJA::forallN<exec_policy>( range(0, order), range(0, order), [=,&B](int i, int j) {
          const auto dij = static_cast<double>(i*order+j);
          const auto addit = (iterations+1.) * (0.5*iterations);
          const auto reference = dij*(1.+iterations)+addit;
          abserr += std::fabs(B[j*order+i] - reference);
      });
      return abserr;
}

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11/RAJA Matrix transpose: B = A^T" << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  if (argc < 3) {
      std::cerr << "Usage: <# iterations> <matrix order> ";
      std::cerr << "<for={seq,omp,tbb,tbbdyn} nested={y,n} tiled={y,n} permute={no,ij,ji} simd={y,n}>\n";
      std::cerr << "Caveat: tiled/permute only supported for nested=y.\n";
      std::cerr << "Feature: RAJA args (foo=bar) can be given in any order.\n";
      return argc;
  }

  int iterations, order;
  std::string use_for="seq", use_permute="no";
  auto use_simd=true, use_nested=true, use_tiled=false;
  try {

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
              if (sf=="omp" || sf=="openmp") {
#ifdef RAJA_ENABLE_OPENMP
                  use_for="omp";
#else
                  std::cerr << "You are trying to use OpenMP but RAJA does not support it!" << std::endl;
#endif
              }
              if (sf=="tbb") {
#ifdef RAJA_ENABLE_TBB
                  use_for="tbb";
#else
                  std::cerr << "You are trying to use TBB but RAJA does not support it!" << std::endl;
#endif
              }
              if (sf=="tbbdyn") {
#ifdef RAJA_ENABLE_TBB
                  use_for="tbbdyn";
#else
                  std::cerr << "You are trying to use TBB but RAJA does not support it!" << std::endl;
#endif
              }
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

  std::string            for_name = "Sequential";
  if (use_for=="omp")    for_name = "OpenMP";
  if (use_for=="tbb")    for_name = "TBB (static)";
  if (use_for=="tbbdyn") for_name = "TBB (dynamic)";

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Matrix order         = " << order << std::endl;
  std::cout << "Tile size            = " << tile_size << "(compile-time constant, unlike other impls)" << std::endl;
  std::cout << "RAJA threading       = " << for_name << std::endl;
  std::cout << "RAJA forallN         = " << (use_nested ? "yes" : "no") << std::endl;
  std::cout << "RAJA use tiling      = " << (use_tiled ? "yes" : "no") << std::endl;
  std::cout << "RAJA use permute     = " << use_permute << std::endl;
  std::cout << "RAJA use simd        = " << (use_simd ? "yes" : "no") << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  std::vector<double> A(order*order);
  std::vector<double> B(order*order);

  if (use_for=="seq") {
    if (use_nested) {
      if (use_tiled) {
        if (use_permute=="no") {
          if (use_simd) {
            Initialize<seq_for_simd_tiled>(order, A, B);
          } else {
            Initialize<seq_for_seq_tiled>(order, A, B);
          }
        }
        else if (use_permute=="ij") {
          if (use_simd) {
            Initialize<seq_for_simd_tiled_ij>(order, A, B);
          } else {
            Initialize<seq_for_seq_tiled_ij>(order, A, B);
          }
        }
        else if (use_permute=="ji") {
          if (use_simd) {
            Initialize<seq_for_simd_tiled_ji>(order, A, B);
          } else {
            Initialize<seq_for_seq_tiled_ji>(order, A, B);
          }
        }
      } else {
        if (use_simd) {
          Initialize<seq_for_simd>(order, A, B);
        } else {
          Initialize<seq_for_seq>(order, A, B);
        }
      }
    } else /* !use_nested */ {
      if (use_simd) {
        Initialize<RAJA::seq_exec,RAJA::simd_exec>(order, A, B);
      } else {
        Initialize<RAJA::seq_exec,RAJA::seq_exec>(order, A, B);
      }
    }
  }
#ifdef RAJA_ENABLE_OPENMP
  else if (use_for=="omp") {
    if (use_nested) {
      if (use_tiled) {
        if (use_permute=="no") {
          if (use_simd) {
            Initialize<omp_for_simd_tiled>(order, A, B);
          } else {
            Initialize<omp_for_seq_tiled>(order, A, B);
          }
        }
        else if (use_permute=="ij") {
          if (use_simd) {
            Initialize<omp_for_simd_tiled_ij>(order, A, B);
          } else {
            Initialize<omp_for_seq_tiled_ij>(order, A, B);
          }
        }
        else if (use_permute=="ji") {
          if (use_simd) {
            Initialize<omp_for_simd_tiled_ji>(order, A, B);
          } else {
            Initialize<omp_for_seq_tiled_ji>(order, A, B);
          }
        }
      } else {
        if (use_simd) {
          Initialize<omp_for_simd>(order, A, B);
        } else {
          Initialize<omp_for_seq>(order, A, B);
        }
      }
    } else /* !use_nested */ {
      if (use_simd) {
        Initialize<RAJA::omp_parallel_for_exec,RAJA::simd_exec>(order, A, B);
      } else {
        Initialize<RAJA::omp_parallel_for_exec,RAJA::seq_exec>(order, A, B);
      }
    }
  }
#endif
#ifdef RAJA_ENABLE_TBB
  else if (use_for=="tbb") {
    if (use_nested) {
      if (use_tiled) {
        if (use_permute=="no") {
          if (use_simd) {
            Initialize<tbb_for_simd_tiled>(order, A, B);
          } else {
            Initialize<tbb_for_seq_tiled>(order, A, B);
          }
        }
        else if (use_permute=="ij") {
          if (use_simd) {
            Initialize<tbb_for_simd_tiled_ij>(order, A, B);
          } else {
            Initialize<tbb_for_seq_tiled_ij>(order, A, B);
          }
        }
        else if (use_permute=="ji") {
          if (use_simd) {
            Initialize<tbb_for_simd_tiled_ji>(order, A, B);
          } else {
            Initialize<tbb_for_seq_tiled_ji>(order, A, B);
          }
        }
      } else {
        if (use_simd) {
          Initialize<tbb_for_simd>(order, A, B);
        } else {
          Initialize<tbb_for_seq>(order, A, B);
        }
      }
    } else /* !use_nested */ {
      if (use_simd) {
        Initialize<RAJA::tbb_for_exec,RAJA::simd_exec>(order, A, B);
      } else {
        Initialize<RAJA::tbb_for_exec,RAJA::seq_exec>(order, A, B);
      }
    }
  }
  else if (use_for=="tbbdyn") {
    if (use_nested) {
      if (use_tiled) {
        if (use_simd) {
          Initialize<tbb_for_dynamic_simd_tiled>(order, A, B);
        } else {
          Initialize<tbb_for_dynamic_seq_tiled>(order, A, B);
        }
      } else {
        if (use_simd) {
          Initialize<tbb_for_dynamic_simd>(order, A, B);
        } else {
          Initialize<tbb_for_dynamic_seq>(order, A, B);
        }
      }
    } else /* !use_nested */ {
      if (use_simd) {
        Initialize<RAJA::tbb_for_dynamic,RAJA::simd_exec>(order, A, B);
      } else {
        Initialize<RAJA::tbb_for_dynamic,RAJA::seq_exec>(order, A, B);
      }
    }
  }
#endif

  double trans_time(0);

  for (int iter = 0; iter<=iterations; iter++) {

    if (iter==1) trans_time = prk::wtime();

    // transpose
    if (use_for=="seq") {
      if (use_nested) {
        if (use_tiled) {
          if (use_permute=="no") {
            if (use_simd) {
              Transpose<seq_for_simd_tiled>(order, A, B);
            } else {
              Transpose<seq_for_seq_tiled>(order, A, B);
            }
          }
          else if (use_permute=="ij") {
            if (use_simd) {
              Transpose<seq_for_simd_tiled_ij>(order, A, B);
            } else {
              Transpose<seq_for_seq_tiled_ij>(order, A, B);
            }
          }
          else if (use_permute=="ji") {
            if (use_simd) {
              Transpose<seq_for_simd_tiled_ji>(order, A, B);
            } else {
              Transpose<seq_for_seq_tiled_ji>(order, A, B);
            }
          }
        } else {
          if (use_simd) {
            Transpose<seq_for_simd>(order, A, B);
          } else {
            Transpose<seq_for_seq>(order, A, B);
          }
        }
      } else /* !use_nested */ {
        if (use_simd) {
          Transpose<RAJA::seq_exec,RAJA::simd_exec>(order, A, B);
        } else {
          Transpose<RAJA::seq_exec,RAJA::seq_exec>(order, A, B);
        }
      }
    }
#ifdef RAJA_ENABLE_OPENMP
    else if (use_for=="omp") {
      if (use_nested) {
        if (use_tiled) {
          if (use_permute=="no") {
            if (use_simd) {
              Transpose<omp_for_simd_tiled>(order, A, B);
            } else {
              Transpose<omp_for_seq_tiled>(order, A, B);
            }
          }
          else if (use_permute=="ij") {
            if (use_simd) {
              Transpose<omp_for_simd_tiled_ij>(order, A, B);
            } else {
              Transpose<omp_for_seq_tiled_ij>(order, A, B);
            }
          }
          else if (use_permute=="ji") {
            if (use_simd) {
              Transpose<omp_for_simd_tiled_ji>(order, A, B);
            } else {
              Transpose<omp_for_seq_tiled_ji>(order, A, B);
            }
          }
        } else {
          if (use_simd) {
            Transpose<omp_for_simd>(order, A, B);
          } else {
            Transpose<omp_for_seq>(order, A, B);
          }
        }
      } else /* !use_nested */ {
        if (use_simd) {
          Transpose<RAJA::omp_parallel_for_exec,RAJA::simd_exec>(order, A, B);
        } else {
          Transpose<RAJA::omp_parallel_for_exec,RAJA::seq_exec>(order, A, B);
        }
      }
    }
#endif
#ifdef RAJA_ENABLE_TBB
    else if (use_for=="tbb") {
      if (use_nested) {
        if (use_tiled) {
          if (use_permute=="no") {
            if (use_simd) {
              Transpose<tbb_for_simd_tiled>(order, A, B);
            } else {
              Transpose<tbb_for_seq_tiled>(order, A, B);
            }
          }
          else if (use_permute=="ij") {
            if (use_simd) {
              Transpose<tbb_for_simd_tiled_ij>(order, A, B);
            } else {
              Transpose<tbb_for_seq_tiled_ij>(order, A, B);
            }
          }
          else if (use_permute=="ji") {
            if (use_simd) {
              Transpose<tbb_for_simd_tiled_ji>(order, A, B);
            } else {
              Transpose<tbb_for_seq_tiled_ji>(order, A, B);
            }
          }
        } else {
          if (use_simd) {
            Transpose<tbb_for_simd>(order, A, B);
          } else {
            Transpose<tbb_for_seq>(order, A, B);
          }
        }
      } else /* !use_nested */ {
        if (use_simd) {
          Transpose<RAJA::tbb_for_exec,RAJA::simd_exec>(order, A, B);
        } else {
          Transpose<RAJA::tbb_for_exec,RAJA::seq_exec>(order, A, B);
        }
      }
    }
    else if (use_for=="tbbdyn") {
      if (use_nested) {
        if (use_tiled) {
          if (use_simd) {
            Transpose<tbb_for_dynamic_simd_tiled>(order, A, B);
          } else {
            Transpose<tbb_for_dynamic_seq_tiled>(order, A, B);
          }
        } else {
          if (use_simd) {
            Transpose<tbb_for_dynamic_simd>(order, A, B);
          } else {
            Transpose<tbb_for_dynamic_seq>(order, A, B);
          }
        }
      } else /* !use_nested */ {
        if (use_simd) {
          Transpose<RAJA::tbb_for_dynamic,RAJA::simd_exec>(order, A, B);
        } else {
          Transpose<RAJA::tbb_for_dynamic,RAJA::seq_exec>(order, A, B);
        }
      }
    }
#endif
  }
  trans_time = prk::wtime() - trans_time;

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  double abserr = 1.0;

  if (use_for=="seq") {
      abserr = Error<RAJA::seq_exec,RAJA::seq_reduce>(iterations,order,B);
  }
#if defined(RAJA_ENABLE_OPENMP)
  else if (use_for=="omp") {
      abserr = Error<RAJA::omp_parallel_for_exec,RAJA::omp_reduce>(iterations,order,B);
  }
#endif
#if defined(RAJA_ENABLE_TBB)
  else if (use_for=="tbb") {
      abserr = Error<RAJA::tbb_for_exec,RAJA::tbb_reduce>(iterations,order,B);
  }
  else if (use_for=="tbbdyn") {
      abserr = Error<RAJA::tbb_for_dynamic,RAJA::tbb_reduce>(iterations,order,B);
  }
#endif

#ifdef VERBOSE
  std::cout << "Sum of absolute differences: " << abserr << std::endl;
#endif

  double epsilon(1.0e-8);
  if (abserr < epsilon) {
    std::cout << "Solution validates" << std::endl;
    auto avgtime = trans_time/iterations;
    auto bytes = (size_t)order * (size_t)order * sizeof(double);
    std::cout << "Rate (MB/s): " << 1.0e-6 * (2.*bytes)/avgtime
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
                ( range(0, order), range(0, order),
                  [=,&A,&B](indx i, indx j) {
                B[i*order+j] += A[j*order+i];
                A[j*order+i] += 1.0;
        });
        RAJA::forallN< RAJA::NestedPolicy< RAJA::ExecList<RAJA::simd_exec, RAJA::simd_exec>,
                                           RAJA::Tile< RAJA::TileList<RAJA::tile_fixed<tile_size>, RAJA::tile_fixed<tile_size>>,
                                                       RAJA::Permute<RAJA::PERM_IJ> > > >
                ( range(0, order), range(0, order),
                  [=,&A,&B](indx i, indx j) {
                B[i*order+j] += A[j*order+i];
                A[j*order+i] += 1.0;
        });
        RAJA::forallN< RAJA::NestedPolicy< RAJA::ExecList<RAJA::simd_exec, RAJA::simd_exec>,
                                           RAJA::Tile< RAJA::TileList<RAJA::tile_fixed<tile_size>, RAJA::tile_fixed<tile_size>>,
                                                       RAJA::Permute<RAJA::PERM_JI> > > >
                ( range(0, order), range(0, order),
                  [=,&A,&B](indx i, indx j) {
                B[i*order+j] += A[j*order+i];
                A[j*order+i] += 1.0;
        });
#endif
