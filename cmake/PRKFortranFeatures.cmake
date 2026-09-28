#[=======================================================================[.rst:
PRKFortranFeatures
-------------------

try_compile()-based probes for Fortran capabilities that aren't libraries
CMake can find_package()/find_library() for -- they're compiler flags that
either work or don't. Mirrors the COARRAYFLAG/OFFLOADFLAG logic spread
across common/make.defs.*.
#]=======================================================================]

include(CheckFortranSourceCompiles)
include(CheckFortranSourceRuns)

function(prk_check_coarray_single)
  if(DEFINED PRK_COARRAY_SINGLE_FLAG)
    return()
  endif()
  set(_src "program main\n  integer :: x[*]\n  x = 1\n  sync all\nend program main\n")
  if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    set(_candidate "-fcoarray=single")
  elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    set(_candidate "-coarray=single")
  else()
    set(_candidate "")
  endif()
  set(CMAKE_REQUIRED_FLAGS "${_candidate}")
  check_fortran_source_compiles("${_src}" PRK_COARRAY_SINGLE_COMPILES SRC_EXT F90)
  unset(CMAKE_REQUIRED_FLAGS)
  if(PRK_COARRAY_SINGLE_COMPILES)
    set(PRK_COARRAY_SINGLE_FLAG "${_candidate}" CACHE STRING "Compiler flag enabling single-image Fortran coarrays")
    message(STATUS "Fortran coarrays (single-image, ${_candidate}): supported")
  else()
    set(PRK_COARRAY_SINGLE_FLAG "" CACHE STRING "Compiler flag enabling single-image Fortran coarrays")
    message(STATUS "Fortran coarrays (single-image): not supported by ${CMAKE_Fortran_COMPILER_ID}")
  endif()
endfunction()

function(prk_check_coarray_lib)
  if(DEFINED PRK_COARRAY_LIB_FLAG OR NOT OpenCoarrays_FOUND)
    return()
  endif()
  set(_src "program main\n  integer :: x[*]\n  x = 1\n  sync all\nend program main\n")
  set(CMAKE_REQUIRED_FLAGS "-fcoarray=lib")
  set(CMAKE_REQUIRED_LIBRARIES "${OpenCoarrays_LIBRARY}")
  check_fortran_source_compiles("${_src}" PRK_COARRAY_LIB_COMPILES SRC_EXT F90)
  unset(CMAKE_REQUIRED_FLAGS)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(PRK_COARRAY_LIB_COMPILES)
    set(PRK_COARRAY_LIB_FLAG "-fcoarray=lib" CACHE STRING "Compiler flag enabling OpenCoarrays-backed distributed coarrays")
    message(STATUS "Fortran coarrays (distributed, -fcoarray=lib + OpenCoarrays): supported")
  else()
    set(PRK_COARRAY_LIB_FLAG "" CACHE STRING "Compiler flag enabling OpenCoarrays-backed distributed coarrays")
    message(STATUS "Fortran coarrays (distributed, -fcoarray=lib): OpenCoarrays found but probe failed")
  endif()
endfunction()

function(prk_check_openacc_links)
  if(DEFINED PRK_OPENACC_WORKS OR NOT OpenACC_Fortran_FOUND)
    return()
  endif()
  # find_package(OpenACC) only checks the compiler accepts the flag, not that
  # the accelerator codegen/link actually succeeds (e.g. gcc's nvptx-none
  # offload toolchain can be installed-but-broken, failing at link time with
  # "mkoffload ... returned 1 exit status" even though compilation is fine).
  set(_src "program main\n  integer :: i, s(10)\n  !$acc parallel loop\n  do i=1,10\n    s(i) = i\n  end do\n  !$acc end parallel loop\nend program main\n")
  set(CMAKE_REQUIRED_FLAGS "${OpenACC_Fortran_FLAGS}")
  check_fortran_source_compiles("${_src}" PRK_OPENACC_LINK_COMPILES SRC_EXT F90)
  unset(CMAKE_REQUIRED_FLAGS)
  if(PRK_OPENACC_LINK_COMPILES)
    set(PRK_OPENACC_WORKS TRUE CACHE BOOL "OpenACC Fortran actually links/codegens, not just compiles")
    message(STATUS "Fortran OpenACC (${OpenACC_Fortran_FLAGS}): compiles and links")
  else()
    set(PRK_OPENACC_WORKS FALSE CACHE BOOL "OpenACC Fortran actually links/codegens, not just compiles")
    message(STATUS "Fortran OpenACC: found but accelerator codegen/link failed (broken offload toolchain?), skipping")
  endif()
endfunction()

function(prk_check_openmp_offload)
  if(DEFINED PRK_OPENMP_OFFLOAD_FLAG)
    return()
  endif()
  # Actually run the probe, not just compile it: on this machine gfortran-14
  # accepts -fopenmp/-foffload=-O3 and compiles real *-openmp-target*.F90
  # sources cleanly, but executing them segfaults or silently produces wrong
  # answers (a broken offload runtime, not a compile-time issue at all) --
  # the same "detected but broken" class of problem as the C++/OpenACC
  # offload toolchain found elsewhere this session, just surfacing only at
  # runtime here. stop 1 if the target region didn't actually run.
  set(_src "program main\n  integer :: s\n  s = 0\n  !$omp target map(tofrom: s)\n  s = 1\n  !$omp end target\n  if (s /= 1) stop 1\nend program main\n")
  if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    # Must be a CMake list (semicolon-separated), not a plain string with an
    # embedded space: CMAKE_REQUIRED_FLAGS below tolerates either, but
    # target_compile_options()/target_link_options() (used when this value
    # is later passed as FLAGS to prk_fortran_executable()) do not -- a
    # plain-string value becomes a single, malformed shell argument
    # ("-fopenmp -foffload=-O3" as one token) that gfortran rejects.
    # -DGPU_SCHEDULE="" matches make.defs.gcc's own OFFLOADFLAG: the
    # *-openmp-target*.F90 sources use GPU_SCHEDULE as a raw macro token
    # appended directly after `collapse(2)` inside `!$omp` directives (via
    # Fortran's -cpp preprocessing), expected to expand to nothing on GNU
    # (other toolchains define it as e.g. schedule(static,1)) -- without
    # this define those directives fail to parse at all, a genuine
    # separate requirement from just getting -fopenmp/-foffload accepted.
    set(_candidate "-fopenmp;-foffload=-O3;-DGPU_SCHEDULE=")
  elseif(CMAKE_Fortran_COMPILER_ID MATCHES "IntelLLVM")
    # Matches make.defs.oneapi's OFFLOADFLAG=-fopenmp-targets=spir64
    # (SPIR-V codegen for Intel GPUs/CPU-as-device fallback) plus the same
    # GPU_SCHEDULE macro requirement as the GNU branch above.
    set(_candidate "-fopenmp;-fopenmp-targets=spir64;-DGPU_SCHEDULE=")
  else()
    set(_candidate "")
  endif()
  if(NOT OpenMP_Fortran_FOUND OR _candidate STREQUAL "")
    set(PRK_OPENMP_OFFLOAD_FLAG "" CACHE STRING "Compiler flags enabling OpenMP target offload")
    message(STATUS "Fortran OpenMP target offload: not probed (no OpenMP, or no known flag for ${CMAKE_Fortran_COMPILER_ID})")
    return()
  endif()
  # CMAKE_REQUIRED_FLAGS wants a plain space-separated flag string, not a
  # ;-separated CMake list (unlike CMAKE_REQUIRED_LIBRARIES/_INCLUDES) --
  # passing the list form here silently dropped every flag after the
  # first one (confirmed via CMakeConfigureLog.yaml: only "-fopenmp" made
  # it into the actual probe compile command, not -fopenmp-targets=spir64
  # or -DGPU_SCHEDULE=), which is a separate bug from the
  # target_compile_options() list-vs-string issue documented above (that
  # one genuinely does need a list).
  string(REPLACE ";" " " _candidate_flags_str "${_candidate}")
  set(CMAKE_REQUIRED_FLAGS "${_candidate_flags_str}")
  check_fortran_source_runs("${_src}" PRK_OPENMP_OFFLOAD_RUNS SRC_EXT F90)
  unset(CMAKE_REQUIRED_FLAGS)
  if(PRK_OPENMP_OFFLOAD_RUNS)
    set(PRK_OPENMP_OFFLOAD_FLAG "${_candidate}" CACHE STRING "Compiler flags enabling OpenMP target offload")
    message(STATUS "Fortran OpenMP target offload (${_candidate}): compiles and runs")
  else()
    set(PRK_OPENMP_OFFLOAD_FLAG "" CACHE STRING "Compiler flags enabling OpenMP target offload")
    message(STATUS "Fortran OpenMP target offload: compiled but failed to run correctly (broken offload runtime)")
  endif()
endfunction()
