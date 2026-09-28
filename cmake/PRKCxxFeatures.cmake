#[=======================================================================[.rst:
PRKCxxFeatures
--------------

try_compile()/FetchContent-based probes for C++ compiler capabilities and
header-only dependencies used by Cxx11, mirroring PRKFortranFeatures.cmake/
PRKCFeatures.cmake's approach for the other languages.
#]=======================================================================]

include(CheckCXXSourceCompiles)
include(FetchContent)

function(prk_check_openmp_offload_cxx)
  if(DEFINED PRK_CXX_OPENMP_OFFLOAD_FLAG)
    return()
  endif()
  if(NOT OpenMP_CXX_FOUND)
    set(PRK_CXX_OPENMP_OFFLOAD_FLAG "" CACHE STRING "Compiler flags enabling C++ OpenMP target offload")
    message(STATUS "C++ OpenMP target offload: not probed, no OpenMP found")
    return()
  endif()
  set(_src "int main() { int s = 0;\n#pragma omp target map(tofrom: s)\n  { s = 1; }\n  return s; }\n")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(_candidate "-fopenmp;-foffload=-O3")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_candidate "-fopenmp;-fopenmp-targets=x86_64-pc-linux-gnu")
  else()
    set(_candidate "")
  endif()
  if(_candidate STREQUAL "")
    set(PRK_CXX_OPENMP_OFFLOAD_FLAG "" CACHE STRING "Compiler flags enabling C++ OpenMP target offload")
    message(STATUS "C++ OpenMP target offload: not probed, no known flag for ${CMAKE_CXX_COMPILER_ID}")
    return()
  endif()
  set(CMAKE_REQUIRED_FLAGS "${_candidate}")
  check_cxx_source_compiles("${_src}" PRK_CXX_OPENMP_OFFLOAD_COMPILES)
  unset(CMAKE_REQUIRED_FLAGS)
  if(PRK_CXX_OPENMP_OFFLOAD_COMPILES)
    set(PRK_CXX_OPENMP_OFFLOAD_FLAG "${_candidate}" CACHE STRING "Compiler flags enabling C++ OpenMP target offload")
    message(STATUS "C++ OpenMP target offload (${_candidate}): compiles and links")
  else()
    set(PRK_CXX_OPENMP_OFFLOAD_FLAG "" CACHE STRING "Compiler flags enabling C++ OpenMP target offload")
    message(STATUS "C++ OpenMP target offload: compiler rejected ${_candidate}")
  endif()
endfunction()

function(prk_check_range_v3_cxx)
  if(DEFINED PRK_RANGE_V3_FOUND)
    return()
  endif()
  find_path(PRK_RANGE_V3_INCLUDE_DIR range/v3/all.hpp)
  if(PRK_RANGE_V3_INCLUDE_DIR)
    set(PRK_RANGE_V3_FOUND TRUE CACHE BOOL "range-v3 headers available")
    message(STATUS "range-v3: found at ${PRK_RANGE_V3_INCLUDE_DIR}")
    return()
  endif()
  # Header-only; fetch it rather than skip stl/ranges entirely, unlike the
  # heavier autobuilt dependencies (GA/PETSc/Kokkos/RAJA) which need an
  # actual build step.
  message(STATUS "range-v3: not found on system, fetching headers (GitHub: ericniebler/range-v3)")
  FetchContent_Declare(range_v3
    GIT_REPOSITORY https://github.com/ericniebler/range-v3.git
    GIT_TAG        0.12.0
    GIT_SHALLOW    TRUE)
  FetchContent_GetProperties(range_v3)
  if(NOT range_v3_POPULATED)
    FetchContent_Populate(range_v3)
  endif()
  if(EXISTS "${range_v3_SOURCE_DIR}/include/range/v3/all.hpp")
    set(PRK_RANGE_V3_INCLUDE_DIR "${range_v3_SOURCE_DIR}/include" CACHE PATH "range-v3 include directory" FORCE)
    set(PRK_RANGE_V3_FOUND TRUE CACHE BOOL "range-v3 headers available")
    message(STATUS "range-v3: fetched to ${PRK_RANGE_V3_INCLUDE_DIR}")
  else()
    set(PRK_RANGE_V3_FOUND FALSE CACHE BOOL "range-v3 headers available")
    message(STATUS "range-v3: fetch failed")
  endif()
endfunction()

# range-v3 (ericniebler/range-v3) was a pre-C++20 hack for getting Ranges
# before the standard had them; prefer the real thing (std::ranges) when it
# actually works, matching prk_ranges.h's USE_GCC_RANGES/USE_BOOST_IRANGE/
# USE_RANGES_TS macro selection. prk::range()'s two call patterns need
# different levels of ranges support, so they're probed and selected
# independently:
#   "basic"  -- prk::range(start,end): std::views::iota alone, C++20 is
#               enough. Priority: STL (C++20) -> Boost.Irange -> range-v3.
#   "full"   -- prk::range(start,end,blocking)/prk::range2(): also needs
#               views::stride and views::cartesian_product, which are C++23
#               additions; Boost.Hana's cartesian_product doesn't compose
#               with boost::irange without real glue code (not worth
#               writing), so there's no usable Boost path here. Priority:
#               STL (C++23) -> range-v3.
# Each tier sets _DEFS (compile definition(s) for prk_ranges.h), _INCLUDES
# (extra include dirs, only needed for range-v3), and _CXX_STANDARD (the
# CXX_STANDARD target property to request, empty if the project default is
# sufficient).
function(prk_check_ranges_cxx)
  if(DEFINED PRK_RANGES_BASIC_DEFS)
    return()
  endif()

  # Pass the standard as a raw flag (CMAKE_REQUIRED_FLAGS) rather than via
  # the CMAKE_CXX_STANDARD variable: that goes through CMake's own
  # per-compiler dialect-flag table, which doesn't know a C++23 flag for
  # every compiler ID here (NVHPC) and hard-errors the try_compile itself
  # rather than just failing the probe. Clear CMAKE_CXX_STANDARD/_REQUIRED
  # locally too (function-scoped, doesn't leak to the caller) -- otherwise
  # try_compile still adds its own -std=gnu++17 (from the project's global
  # CMAKE_CXX_STANDARD 17) *after* our CMAKE_REQUIRED_FLAGS, and the later
  # -std= flag wins, silently discarding the one we asked for.
  set(CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD_REQUIRED)

  # --- basic: prk::range(start,end) -----------------------------------
  set(_basic_found FALSE)
  if(PRK_CXX20_SUPPORTED)
    set(CMAKE_REQUIRED_FLAGS "-std=c++20")
    check_cxx_source_compiles(
      "#include <ranges>\nint main() { auto r = std::views::iota(0,10); for (auto i : r) { (void)i; } return 0; }\n"
      PRK_RANGES_BASIC_STL_COMPILES)
    unset(CMAKE_REQUIRED_FLAGS)
    if(PRK_RANGES_BASIC_STL_COMPILES)
      set(PRK_RANGES_BASIC_DEFS USE_GCC_RANGES CACHE STRING "prk_ranges.h backend (basic)" FORCE)
      set(PRK_RANGES_BASIC_INCLUDES "" CACHE STRING "extra include dirs for the basic ranges backend" FORCE)
      set(PRK_RANGES_BASIC_CXX_STANDARD 20 CACHE STRING "CXX_STANDARD needed for the basic ranges backend" FORCE)
      set(_basic_found TRUE)
      message(STATUS "C++ ranges (basic): std::ranges (C++20)")
    endif()
  endif()
  if(NOT _basic_found)
    find_path(PRK_BOOST_IRANGE_INCLUDE_DIR boost/range/irange.hpp)
    if(PRK_BOOST_IRANGE_INCLUDE_DIR)
      set(CMAKE_REQUIRED_INCLUDES "${PRK_BOOST_IRANGE_INCLUDE_DIR}")
      check_cxx_source_compiles(
        "#include \"boost/range/irange.hpp\"\nint main() { auto r = boost::irange(0,10); for (auto i : r) { (void)i; } return 0; }\n"
        PRK_RANGES_BASIC_BOOST_COMPILES)
      unset(CMAKE_REQUIRED_INCLUDES)
      if(PRK_RANGES_BASIC_BOOST_COMPILES)
        set(PRK_RANGES_BASIC_DEFS USE_BOOST_IRANGE CACHE STRING "prk_ranges.h backend (basic)" FORCE)
        set(PRK_RANGES_BASIC_INCLUDES "" CACHE STRING "extra include dirs for the basic ranges backend" FORCE)
        set(PRK_RANGES_BASIC_CXX_STANDARD "" CACHE STRING "CXX_STANDARD needed for the basic ranges backend" FORCE)
        set(_basic_found TRUE)
        message(STATUS "C++ ranges (basic): Boost.Range irange")
      endif()
    endif()
  endif()
  if(NOT _basic_found)
    prk_check_range_v3_cxx()
    set(PRK_RANGES_BASIC_DEFS USE_RANGES_TS CACHE STRING "prk_ranges.h backend (basic)" FORCE)
    set(PRK_RANGES_BASIC_INCLUDES "${PRK_RANGE_V3_INCLUDE_DIR}" CACHE STRING "extra include dirs for the basic ranges backend" FORCE)
    set(PRK_RANGES_BASIC_CXX_STANDARD "" CACHE STRING "CXX_STANDARD needed for the basic ranges backend" FORCE)
    message(STATUS "C++ ranges (basic): range-v3 (fallback)")
  endif()

  # --- full: prk::range(start,end,blocking) / prk::range2() -----------
  set(_full_found FALSE)
  if(PRK_CXX23_SUPPORTED)
    set(CMAKE_REQUIRED_FLAGS "-std=c++23")
    check_cxx_source_compiles(
      "#include <ranges>\nint main() { auto a = std::views::iota(0,10) | std::views::stride(2); auto b = std::views::iota(0,3); auto cp = std::views::cartesian_product(b,b); for (auto t : cp) { (void)t; } return 0; }\n"
      PRK_RANGES_FULL_STL_COMPILES)
    unset(CMAKE_REQUIRED_FLAGS)
    if(PRK_RANGES_FULL_STL_COMPILES)
      set(PRK_RANGES_FULL_DEFS USE_GCC_RANGES CACHE STRING "prk_ranges.h backend (full)" FORCE)
      set(PRK_RANGES_FULL_INCLUDES "" CACHE STRING "extra include dirs for the full ranges backend" FORCE)
      set(PRK_RANGES_FULL_CXX_STANDARD 23 CACHE STRING "CXX_STANDARD needed for the full ranges backend" FORCE)
      set(_full_found TRUE)
      message(STATUS "C++ ranges (full, stride+cartesian_product): std::ranges (C++23)")
    endif()
  endif()
  if(NOT _full_found)
    prk_check_range_v3_cxx()
    set(PRK_RANGES_FULL_DEFS USE_RANGES_TS CACHE STRING "prk_ranges.h backend (full)" FORCE)
    set(PRK_RANGES_FULL_INCLUDES "${PRK_RANGE_V3_INCLUDE_DIR}" CACHE STRING "extra include dirs for the full ranges backend" FORCE)
    set(PRK_RANGES_FULL_CXX_STANDARD "" CACHE STRING "CXX_STANDARD needed for the full ranges backend" FORCE)
    message(STATUS "C++ ranges (full, stride+cartesian_product): range-v3 (fallback)")
  endif()
endfunction()

function(prk_check_pstl_cxx)
  if(DEFINED PRK_PSTL_WORKS)
    return()
  endif()
  if(NOT TBB_FOUND)
    set(PRK_PSTL_WORKS FALSE CACHE BOOL "libstdc++/libc++ parallel <execution> algorithms link with TBB")
    return()
  endif()
  # libstdc++'s <execution> parallel policies dispatch to TBB; this only
  # actually works if the TBB found is compatible with what the standard
  # library expects, so probe by linking, not just compiling.
  set(_src "#include <execution>\n#include <vector>\n#include <algorithm>\nint main() { std::vector<int> v(10,1); std::sort(std::execution::par, v.begin(), v.end()); return v[0]; }\n")
  set(CMAKE_REQUIRED_LIBRARIES TBB::tbb)
  check_cxx_source_compiles("${_src}" PRK_PSTL_LINK_COMPILES)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(PRK_PSTL_LINK_COMPILES)
    set(PRK_PSTL_WORKS TRUE CACHE BOOL "libstdc++/libc++ parallel <execution> algorithms link with TBB")
    message(STATUS "C++ parallel <execution> + TBB: compiles and links")
  else()
    set(PRK_PSTL_WORKS FALSE CACHE BOOL "libstdc++/libc++ parallel <execution> algorithms link with TBB")
    message(STATUS "C++ parallel <execution> + TBB: link failed")
  endif()
endfunction()

function(prk_check_openacc_cxx)
  if(DEFINED PRK_OPENACC_FLAGS)
    return()
  endif()
  # Only GNU and NVHPC have a real, working OpenACC C++ implementation
  # among the compilers available here (Intel oneAPI never shipped one).
  # Match the flags the Makefile's make.defs.gcc/make.defs.nvhpc use, but
  # verify with a real compile+link probe rather than trusting compiler ID
  # alone -- the same principle as the OpenMP-offload/CUDASTF probes above:
  # a compiler can accept -fopenacc/-acc and still reject or silently
  # miscompile a given pragma.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(_candidate "-fopenacc")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    set(_candidate "-acc;-target=gpu;-Mlarge_arrays")
  else()
    set(_candidate "")
  endif()
  if(_candidate STREQUAL "")
    set(PRK_OPENACC_FLAGS "" CACHE STRING "Compiler flags enabling C++ OpenACC")
    message(STATUS "C++ OpenACC: not probed, no known flags for ${CMAKE_CXX_COMPILER_ID}")
    return()
  endif()
  set(_src "int main() { int s = 0;\n#pragma acc parallel copy(s)\n  { s = 1; }\n  return s; }\n")
  set(CMAKE_REQUIRED_FLAGS "${_candidate}")
  check_cxx_source_compiles("${_src}" PRK_CXX_OPENACC_COMPILES)
  unset(CMAKE_REQUIRED_FLAGS)
  if(PRK_CXX_OPENACC_COMPILES)
    set(PRK_OPENACC_FLAGS "${_candidate}" CACHE STRING "Compiler flags enabling C++ OpenACC")
    message(STATUS "C++ OpenACC (${_candidate}): compiles and links")
  else()
    set(PRK_OPENACC_FLAGS "" CACHE STRING "Compiler flags enabling C++ OpenACC")
    message(STATUS "C++ OpenACC: compiler rejected ${_candidate}")
  endif()
endfunction()

function(prk_check_stdpar_cxx)
  if(DEFINED PRK_STDPAR_WORKS)
    return()
  endif()
  # NVHPC's nvc++ has its own GPU-offloading implementation of parallel
  # <execution> algorithms behind -stdpar=gpu, independent of TBB/PSTL_WORKS
  # (matches make.defs.nvhpc/make.defs.cuda's STDPARFLAG). Everywhere else,
  # "stdpar" just means the standard library's TBB-backed parallel
  # <execution>, which PRK_PSTL_WORKS already verified with a real link
  # probe -- no extra flags needed there, just the same TBB::tbb link.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    set(_candidate "-stdpar=gpu;-gpu=managed;-Minfo=accel;-cudalib=cublas,cutensor")
    set(CMAKE_REQUIRED_FLAGS "${_candidate}")
    set(_src "#include <execution>\n#include <vector>\n#include <algorithm>\nint main() { std::vector<int> v(10,1); std::sort(std::execution::par_unseq, v.begin(), v.end()); return v[0]; }\n")
    check_cxx_source_compiles("${_src}" PRK_CXX_STDPAR_NVHPC_COMPILES)
    unset(CMAKE_REQUIRED_FLAGS)
    if(PRK_CXX_STDPAR_NVHPC_COMPILES)
      set(PRK_STDPAR_FLAGS "${_candidate}" CACHE STRING "Compiler flags enabling GPU-offloaded stdpar")
      set(PRK_STDPAR_WORKS TRUE CACHE BOOL "stdpar (parallel <execution>) is usable")
      message(STATUS "C++ stdpar (NVHPC GPU, ${_candidate}): compiles and links")
    else()
      set(PRK_STDPAR_FLAGS "" CACHE STRING "Compiler flags enabling GPU-offloaded stdpar")
      set(PRK_STDPAR_WORKS FALSE CACHE BOOL "stdpar (parallel <execution>) is usable")
      message(STATUS "C++ stdpar: NVHPC rejected ${_candidate}")
    endif()
  elseif(PRK_PSTL_WORKS)
    set(PRK_STDPAR_FLAGS "" CACHE STRING "Compiler flags enabling GPU-offloaded stdpar")
    set(PRK_STDPAR_WORKS TRUE CACHE BOOL "stdpar (parallel <execution>) is usable")
    message(STATUS "C++ stdpar: using TBB-backed parallel <execution> (same as PSTL)")
  else()
    set(PRK_STDPAR_FLAGS "" CACHE STRING "Compiler flags enabling GPU-offloaded stdpar")
    set(PRK_STDPAR_WORKS FALSE CACHE BOOL "stdpar (parallel <execution>) is usable")
    message(STATUS "C++ stdpar: not available (no NVHPC GPU offload, no working TBB-backed PSTL)")
  endif()
endfunction()

function(prk_check_cudastf_cuda cudax_include libcudacxx_include)
  if(DEFINED PRK_CUDASTF_WORKS)
    return()
  endif()
  # deps/stf/cccl (a local checkout of NVIDIA's CCCL/CUDASTF) predates
  # CUDA 13's cudaGraphAddDependencies() signature change; compile-check
  # against the real header rather than trusting "the path exists", the
  # same principle as the OpenACC/Kokkos-CUDA link probes elsewhere.
  set(_src "#include <cuda/experimental/stf.cuh>\nusing namespace cuda::experimental::stf;\nint main() { context ctx; ctx.finalize(); return 0; }\n")
  file(WRITE "${CMAKE_BINARY_DIR}/prk_cudastf_probe.cu" "${_src}")
  try_compile(PRK_CUDASTF_WORKS
    "${CMAKE_BINARY_DIR}/prk_cudastf_probe"
    "${CMAKE_BINARY_DIR}/prk_cudastf_probe.cu"
    CXX_STANDARD 17
    COMPILE_DEFINITIONS "--extended-lambda;--expt-relaxed-constexpr"
    LINK_LIBRARIES CUDA::cuda_driver
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${cudax_include};${libcudacxx_include}")
  set(PRK_CUDASTF_WORKS ${PRK_CUDASTF_WORKS} CACHE BOOL "CUDASTF (deps/stf/cccl) actually compiles against this CUDA Toolkit")
  if(PRK_CUDASTF_WORKS)
    message(STATUS "CUDASTF: compiles against CUDA ${CMAKE_CUDA_COMPILER_VERSION}")
  else()
    message(STATUS "CUDASTF: deps/stf/cccl checkout doesn't compile against CUDA ${CMAKE_CUDA_COMPILER_VERSION} (likely too old for this CUDA Toolkit's API)")
  endif()
endfunction()

function(prk_check_thrust_host_cxx)
  if(DEFINED PRK_THRUST_HOST_WORKS)
    return()
  endif()
  # nstream-host-thrust.cc/transpose-host-thrust.cc use nvcc's "extended
  # lambda" syntax (__host__ __device__ between [] and ()) even though the
  # Makefile's own %-thrust: %-thrust.cc rule compiles them with the plain
  # host compiler ($(CXX), not nvcc) -- that syntax is a CUDA-specific
  # language extension a standard host compiler's parser rejects outright
  # (this isn't a missing-macro issue __host__/__device__ do get defined as
  # empty by Thrust's headers for non-nvcc builds; it's the placement
  # between [] and () itself that's nvcc-only syntax). Probe for it rather
  # than assume any C++ compiler accepts it.
  set(_src "int main() { auto f = [=] __host__ __device__ (int x) { return x; }; return f(0); }\n")
  check_cxx_source_compiles("${_src}" PRK_THRUST_HOST_COMPILES)
  set(PRK_THRUST_HOST_WORKS ${PRK_THRUST_HOST_COMPILES} CACHE BOOL "Host C++ compiler accepts nvcc's extended-lambda syntax")
  if(PRK_THRUST_HOST_WORKS)
    message(STATUS "Host Thrust (nvcc extended-lambda syntax): ${CMAKE_CXX_COMPILER_ID} accepts it")
  else()
    message(STATUS "Host Thrust: ${CMAKE_CXX_COMPILER_ID} does not understand nvcc's extended-lambda syntax, skipping nstream/transpose-host-thrust")
  endif()
endfunction()
