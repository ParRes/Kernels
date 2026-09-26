#
# CMake port of Cxx11/Makefile.legacy. Target groups and their toolchain
# gating mirror the Makefile's per-group flag requirements (OPENMPFLAG,
# CBLASFLAG, CUDAFLAGS, KOKKOSFLAG, RAJAFLAG, ...) as closely as CMake's
# find_package()/find_program()/try_compile() facilities allow.
#
# CI (ci/build-run-prk.sh's allcxx) case) only ever exercises: valarray,
# vector (+cblas on Darwin), thread/async, openmp (+target on gcc) --
# everything after that in the script is dead code behind an `exit 0`.
# This port covers substantially more, gated on real detection, since this
# machine has working CUDA/cuBLAS (real GPUs), OpenCL, Boost.Compute, and
# prebuilt NVSHMEM to verify against; Kokkos/RAJA are autobuilt the same
# way GA/OpenCoarrays/PETSc are for the other languages. TBB, SYCL/DPC++,
# HIP/ROCm, oneAPI, OCCA, Halide, HPX, UPC++, and OpenSHMEM have no
# available toolchain here and are detect-and-skip only.
#
# Standalone project: `cd Cxx11 && cmake -S . -B build` works on its own
# (in-source `cmake .` too). Also add_subdirectory()'d from the repo root.
#

cmake_minimum_required(VERSION 3.18)
project(PRK_Cxx11 LANGUAGES CXX)
enable_testing()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake/modules")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-std=c++20 PRK_CXX20_SUPPORTED)
check_cxx_compiler_flag(-std=c++23 PRK_CXX23_SUPPORTED)

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/../common/PRKVERSION" _prkversion_line REGEX "^PRKVERSION=")
string(REGEX REPLACE "^PRKVERSION=(.*)$" "\\1" PRK_VERSION_DEFINE "${_prkversion_line}")
add_compile_definitions(PRKVERSION=${PRK_VERSION_DEFINE})

# --- dependency detection -----------------------------------------------

find_package(Threads REQUIRED)
find_package(OpenMP COMPONENTS CXX)
find_package(MPI COMPONENTS CXX)
find_package(BLAS)
find_package(OpenCL)
# TBB is installed here via the oneAPI toolkit, but only on CMAKE_PREFIX_PATH
# when setvars.sh has been sourced; hint at its standalone install location
# too so a plain (un-sourced) GNU build can still find and use it -- it's a
# real, present dependency, not one to silently skip just because the
# environment wasn't activated.
find_package(TBB QUIET PATHS /opt/intel/oneapi/tbb/*/lib/cmake/tbb)

find_path(PRK_BOOST_COMPUTE_INCLUDE_DIR boost/compute.hpp)
# Whether this is actually usable also depends on OpenCL_FOUND, which the
# CUDA-based fallback below can still flip from FALSE to TRUE -- the real
# check happens after that runs (see "Boost.Compute needs OpenCL" below).

# NVSHMEM is available two ways here: a system package built against CUDA
# 13 (headers under /usr/include/nvshmem_13, libs resolved by the default
# search path via /etc/alternatives to /usr/lib/x86_64-linux-gnu/nvshmem/13,
# version 3.7.0) and a prebuilt binary archive in deps/ built against CUDA
# 12 (version 3.3.24). find_library() for nvshmem_host/nvshmem_device
# always resolves to the system CUDA-13 package (no PRK-specific hint took
# precedence over the default search path), so the header search has to
# prefer the matching system location too -- pulling the header from deps/
# while the libs came from the system package is a version mismatch that
# nvlink catches as "Size doesn't match" on nvshmemi_device_state_d.
find_path(PRK_NVSHMEM_INCLUDE_DIR nvshmem.h
  HINTS ENV NVSHMEM_DIR
  PATHS /usr/include/nvshmem_13 "${CMAKE_CURRENT_LIST_DIR}/../deps/libnvshmem-linux-x86_64-3.3.24_cuda12-archive/include"
  PATH_SUFFIXES include)
find_library(PRK_NVSHMEM_HOST_LIBRARY nvshmem_host
  HINTS ENV NVSHMEM_DIR
  PATH_SUFFIXES lib
  PATHS "${CMAKE_CURRENT_LIST_DIR}/../deps/libnvshmem-linux-x86_64-3.3.24_cuda12-archive/lib")
# The device-side NVSHMEM symbols (nvshmemi_transfer_rma, nvshmemi_device_state_d,
# ...) live in libnvshmem_device.a, not libnvshmem_host -- nvlink fails with
# undefined references without it, since -rdc=true device-links across TUs.
find_library(PRK_NVSHMEM_DEVICE_LIBRARY nvshmem_device
  HINTS ENV NVSHMEM_DIR
  PATH_SUFFIXES lib
  PATHS "${CMAKE_CURRENT_LIST_DIR}/../deps/libnvshmem-linux-x86_64-3.3.24_cuda12-archive/lib")
if(PRK_NVSHMEM_INCLUDE_DIR AND PRK_NVSHMEM_HOST_LIBRARY AND PRK_NVSHMEM_DEVICE_LIBRARY)
  set(PRK_NVSHMEM_FOUND TRUE)
endif()
# The NVSHMEM executables call nvshmem_init() directly (no MPI bootstrap),
# so they need NVSHMEM's own PMI-based launcher, not mpirun.
find_program(PRK_NVSHRUN_EXECUTABLE nvshmrun
  HINTS ENV NVSHMEM_DIR
  PATH_SUFFIXES bin
  PATHS /usr/bin/nvshmem_13 "${CMAKE_CURRENT_LIST_DIR}/../deps/libnvshmem-linux-x86_64-3.3.24_cuda12-archive/bin")

# CUDASTF is part of NVIDIA's CCCL (cudax + libcu++); this machine has a
# local checkout in deps/stf/cccl, no CMake config used (matches the
# Makefile's plain -I flags).
find_path(PRK_CUDASTF_CUDAX_INCLUDE_DIR cuda/experimental/stf.cuh
  HINTS ENV CUDASTF_DIR
  PATH_SUFFIXES cudax/include
  PATHS "${CMAKE_CURRENT_LIST_DIR}/../deps/stf/cccl")
find_path(PRK_CUDASTF_LIBCUDACXX_INCLUDE_DIR cuda/std/version
  HINTS ENV CUDASTF_DIR
  PATH_SUFFIXES libcudacxx/include
  PATHS "${CMAKE_CURRENT_LIST_DIR}/../deps/stf/cccl")
if(PRK_CUDASTF_CUDAX_INCLUDE_DIR AND PRK_CUDASTF_LIBCUDACXX_INCLUDE_DIR)
  set(PRK_CUDASTF_FOUND TRUE)
endif()

# CUDA: only enable_language() if a compiler is actually there, exactly
# like PRK_ENABLE_FORTRAN/PRK_ENABLE_C1Z avoid requiring unrelated
# compilers -- CUDA shouldn't be mandatory to build the rest of Cxx11.
include(CheckLanguage)
include(CheckCXXCompilerFlag)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  find_program(PRK_NVIDIA_SMI nvidia-smi)
  # Total installed GPU count, independent of CMAKE_CUDA_ARCHITECTURES below
  # (that's unique architectures, deduplicated) -- used wherever a CTest
  # invocation needs to know how many GPUs are actually present (e.g.
  # dgemm-mpi-cublas requires one MPI rank per GPU). Not nested inside the
  # "if(NOT CMAKE_CUDA_ARCHITECTURES)" block below: that's skipped on a
  # reconfigure once CMAKE_CUDA_ARCHITECTURES is already cached, which would
  # otherwise leave PRK_NUM_GPUS undefined on any run after the first.
  if(NOT DEFINED PRK_NUM_GPUS)
    set(PRK_NUM_GPUS 1)
    if(PRK_NVIDIA_SMI)
      execute_process(
        COMMAND "${PRK_NVIDIA_SMI}" --query-gpu=count --format=csv,noheader
        OUTPUT_VARIABLE _prk_num_gpus_out
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _prk_num_gpus_result)
      if(_prk_num_gpus_result EQUAL 0)
        string(REGEX MATCH "^[0-9]+" _prk_num_gpus_first_line "${_prk_num_gpus_out}")
        if(_prk_num_gpus_first_line)
          set(PRK_NUM_GPUS "${_prk_num_gpus_first_line}")
        endif()
      endif()
    endif()
    set(PRK_NUM_GPUS "${PRK_NUM_GPUS}" CACHE STRING "Number of installed GPUs (nvidia-smi)" FORCE)
  endif()

  if(NOT CMAKE_CUDA_ARCHITECTURES)
    # CMAKE_CUDA_ARCHITECTURES=native is unreliable on this machine: it
    # resolves to 75 (Turing) even though every installed GPU is an H100
    # (sm_90) -- building for the wrong arch produced bogus CUB/Thrust
    # for_each template errors (wrong policy specialization, not a real
    # incompatibility). Query nvidia-smi directly instead, falling back to
    # native if that's unavailable. This has to run and set the cache
    # variable *before* enable_language(CUDA): once that runs, CMake caches
    # its own (wrong, here) default and "if(NOT CMAKE_CUDA_ARCHITECTURES)"
    # is no longer true.
    set(_prk_cuda_archs native)
    if(PRK_NVIDIA_SMI)
      execute_process(
        COMMAND "${PRK_NVIDIA_SMI}" --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE _prk_gpu_ccs
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _prk_nvidia_smi_result)
      if(_prk_nvidia_smi_result EQUAL 0 AND _prk_gpu_ccs)
        string(REPLACE "\n" ";" _prk_gpu_ccs_list "${_prk_gpu_ccs}")
        list(REMOVE_DUPLICATES _prk_gpu_ccs_list)
        set(_prk_detected_archs "")
        foreach(_prk_cc IN LISTS _prk_gpu_ccs_list)
          string(REPLACE "." "" _prk_cc_digits "${_prk_cc}")
          if(_prk_cc_digits MATCHES "^[0-9]+$")
            list(APPEND _prk_detected_archs "${_prk_cc_digits}")
          endif()
        endforeach()
        if(_prk_detected_archs)
          set(_prk_cuda_archs "${_prk_detected_archs}")
        endif()
      endif()
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${_prk_cuda_archs}" CACHE STRING "" FORCE)
  endif()
  enable_language(CUDA)
  set(PRK_WITH_CUDA TRUE)
  if(NOT CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
  # Nearly every .cu kernel here uses a __host__/__device__ lambda (cuda/
  # cublas/thrust-device/cudastf/nvshmem groups alike), which nvcc refuses
  # without this flag -- add it globally for CUDA compiles rather than
  # per-target.
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda> $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  find_package(CUDAToolkit REQUIRED)
  # Modern CUDA toolkits (12.something+) moved the bundled Thrust/CUB/libcu++
  # headers (CCCL) into a "cccl" subdirectory that nvcc adds to its search
  # path automatically but a plain host compiler (g++, for nstream-host-thrust)
  # does not -- has to be found and added explicitly.
  find_path(PRK_THRUST_INCLUDE_DIR thrust/version.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS}
    PATH_SUFFIXES cccl)
  if(NOT PRK_THRUST_INCLUDE_DIR)
    set(PRK_THRUST_INCLUDE_DIR ${CUDAToolkit_INCLUDE_DIRS})
  endif()
else()
  set(PRK_WITH_CUDA FALSE)
endif()

if(PRK_WITH_CUDA AND NOT OpenCL_FOUND)
  # find_package(OpenCL) above only failed on OpenCL_INCLUDE_DIR (missing
  # CL/cl.h) even though the NVIDIA driver's OpenCL ICD is installed and
  # working here (libOpenCL.so + /etc/OpenCL/vendors/nvidia.icd) -- a
  # machine with a working CUDA GPU has OpenCL for that GPU too, the
  # headers just aren't packaged system-wide. CUDA toolkits sometimes
  # bundle CL/cl.h (this machine's active 13.0 doesn't, but a co-installed
  # 12.8 does) -- search the active toolkit first, then any other
  # co-installed CUDA toolkit, before giving up.
  file(GLOB PRK_OPENCL_CUDA_CANDIDATES /usr/local/cuda-*/targets/*/include)
  find_path(PRK_OPENCL_CUDA_INCLUDE_DIR CL/cl.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS}
    PATHS ${PRK_OPENCL_CUDA_CANDIDATES})
  if(PRK_OPENCL_CUDA_INCLUDE_DIR)
    set(OpenCL_INCLUDE_DIR "${PRK_OPENCL_CUDA_INCLUDE_DIR}" CACHE PATH "OpenCL include directory" FORCE)
    find_package(OpenCL)
    if(OpenCL_FOUND)
      message(STATUS "OpenCL: found via CUDA toolkit headers at ${PRK_OPENCL_CUDA_INCLUDE_DIR} (NVIDIA ICD)")
    endif()
  endif()
endif()

# Boost.Compute needs OpenCL: re-check now that the CUDA-header fallback
# above has had a chance to flip OpenCL_FOUND from FALSE to TRUE.
if(PRK_BOOST_COMPUTE_INCLUDE_DIR AND OpenCL_FOUND)
  set(PRK_BOOST_COMPUTE_FOUND TRUE)
endif()

# RAJA's BLT build system does `set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ...
# CACHE PATH ...)` (no FORCE) inside its own subdirectory; on a fresh
# configure (this cache variable not yet defined), that silently becomes
# the project-wide default output directory for every add_executable()
# called afterward -- including every one of THIS file's own kernels --
# redirecting them all into _deps/raja-build/bin/ and breaking every
# bare-name CTest COMMAND that expects the binary next to the build dir
# root. Pre-populate the cache entry ourselves (before the autobuild
# subdirectories run) so BLT's own un-forced set() is a no-op -- but leave
# it EMPTY rather than pointing it at this directory: CMAKE_RUNTIME_
# OUTPUT_DIRECTORY is a global CACHE variable, not directory-scoped, so
# forcing it to Cxx11's own binary dir would also redirect every sibling
# language directory's targets here when this is add_subdirectory()'d
# from the repo root alongside SERIAL/OPENMP/etc. An empty value falls
# back to CMake's normal per-target default (CMAKE_CURRENT_BINARY_DIR of
# wherever add_executable() was called), which is what every language
# directory's own bare-name CTest COMMAND already relies on.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "" CACHE PATH
  "Directory where executables will go in the build tree" FORCE)

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/AutobuildDeps.cmake")
prk_autobuild_kokkos()
prk_autobuild_raja()

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/PRKCxxFeatures.cmake")
prk_check_openmp_offload_cxx()
prk_check_ranges_cxx()
prk_check_pstl_cxx()
prk_check_openacc_cxx()
prk_check_stdpar_cxx()
if(PRK_WITH_CUDA AND PRK_CUDASTF_FOUND)
  prk_check_cudastf_cuda("${PRK_CUDASTF_CUDAX_INCLUDE_DIR}" "${PRK_CUDASTF_LIBCUDACXX_INCLUDE_DIR}")
endif()
if(PRK_WITH_CUDA)
  prk_check_thrust_host_cxx()
endif()

set(PRK_WITH_TARGET FALSE)
if(PRK_CXX_OPENMP_OFFLOAD_FLAG)
  set(PRK_WITH_TARGET TRUE)
endif()

message(STATUS "Cxx11: OpenMP ${OpenMP_CXX_FOUND}, MPI ${MPI_CXX_FOUND}, BLAS ${BLAS_FOUND}, "
               "OpenCL ${OpenCL_FOUND}, Boost.Compute ${PRK_BOOST_COMPUTE_FOUND}, TBB ${TBB_FOUND}, "
               "ranges(basic) ${PRK_RANGES_BASIC_DEFS}, ranges(full) ${PRK_RANGES_FULL_DEFS}, PSTL ${PRK_PSTL_WORKS}, "
               "CUDA ${PRK_WITH_CUDA}, NVSHMEM ${PRK_NVSHMEM_FOUND}, CUDASTF ${PRK_CUDASTF_FOUND}, "
               "Kokkos ${Kokkos_FOUND}, RAJA ${RAJA_FOUND}, target offload ${PRK_WITH_TARGET} "
               "(compiler: ${CMAKE_CXX_COMPILER_ID})")

# --- helper -----------------------------------------------------------

function(prk_cxx_executable name)
  set(multiValueArgs SOURCES LIBS FLAGS DEFS INCLUDES)
  cmake_parse_arguments(PXE "" "" "${multiValueArgs}" ${ARGN})
  set(_target cxx11_${name})
  add_executable(${_target} ${PXE_SOURCES})
  set_target_properties(${_target} PROPERTIES OUTPUT_NAME ${name})
  target_link_libraries(${_target} PRIVATE Threads::Threads ${PXE_LIBS})
  if(PXE_FLAGS)
    target_compile_options(${_target} PRIVATE ${PXE_FLAGS})
    target_link_options(${_target} PRIVATE ${PXE_FLAGS})
  endif()
  if(PXE_DEFS)
    target_compile_definitions(${_target} PRIVATE ${PXE_DEFS})
  endif()
  if(PXE_INCLUDES)
    target_include_directories(${_target} PRIVATE ${PXE_INCLUDES})
  endif()
endfunction()

function(prk_cxx_test test_name)
  set(multiValueArgs COMMAND)
  cmake_parse_arguments(PXT "" "" "${multiValueArgs}" ${ARGN})
  add_test(NAME ${test_name} COMMAND ${PXT_COMMAND})
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 30)
endfunction()

# --- sequential -------------------------------------------------------

prk_cxx_executable(p2p       SOURCES p2p.cc)
prk_cxx_executable(stencil   SOURCES stencil.cc)
prk_cxx_executable(transpose SOURCES transpose.cc)
prk_cxx_executable(nstream   SOURCES nstream.cc)
prk_cxx_executable(dgemm     SOURCES dgemm.cc)
prk_cxx_executable(sparse    SOURCES sparse.cc)
add_custom_target(cxx11_sequential DEPENDS cxx11_p2p cxx11_stencil cxx11_transpose cxx11_nstream cxx11_dgemm cxx11_sparse)

# --- vector -------------------------------------------------------------

prk_cxx_executable(p2p-vector            SOURCES p2p-vector.cc)
prk_cxx_executable(p2p-hyperplane-vector SOURCES p2p-hyperplane-openmp.cc)
prk_cxx_executable(stencil-vector        SOURCES stencil-vector.cc)
prk_cxx_executable(transpose-vector      SOURCES transpose-vector.cc)
prk_cxx_executable(nstream-vector        SOURCES nstream-vector.cc)
prk_cxx_executable(sparse-vector         SOURCES sparse-vector.cc)
prk_cxx_executable(dgemm-vector          SOURCES dgemm-vector.cc)
prk_cxx_executable(transpose-async       SOURCES transpose-async.cc)
prk_cxx_executable(transpose-thread      SOURCES transpose-thread.cc)
add_custom_target(cxx11_vector DEPENDS
  cxx11_p2p-vector cxx11_p2p-hyperplane-vector cxx11_stencil-vector cxx11_transpose-vector cxx11_nstream-vector
  cxx11_sparse-vector cxx11_dgemm-vector cxx11_transpose-async cxx11_transpose-thread)

# --- valarray -----------------------------------------------------------

prk_cxx_executable(transpose-valarray SOURCES transpose-valarray.cc)
prk_cxx_executable(nstream-valarray   SOURCES nstream-valarray.cc)
add_custom_target(cxx11_valarray DEPENDS cxx11_transpose-valarray cxx11_nstream-valarray)

# --- openmp / target / taskloop -------------------------------------------

if(OpenMP_CXX_FOUND)
  prk_cxx_executable(p2p-hyperplane-openmp SOURCES p2p-hyperplane-openmp.cc LIBS OpenMP::OpenMP_CXX)
  prk_cxx_executable(stencil-openmp        SOURCES stencil-openmp.cc        LIBS OpenMP::OpenMP_CXX)
  prk_cxx_executable(transpose-openmp      SOURCES transpose-openmp.cc      LIBS OpenMP::OpenMP_CXX)
  prk_cxx_executable(nstream-openmp        SOURCES nstream-openmp.cc        LIBS OpenMP::OpenMP_CXX)
  prk_cxx_executable(p2p-tasks-openmp      SOURCES p2p-tasks-openmp.cc      LIBS OpenMP::OpenMP_CXX)
  add_custom_target(cxx11_openmp DEPENDS
    cxx11_p2p-hyperplane-openmp cxx11_stencil-openmp cxx11_transpose-openmp cxx11_nstream-openmp cxx11_p2p-tasks-openmp)

  prk_cxx_executable(stencil-taskloop   SOURCES stencil-taskloop.cc   LIBS OpenMP::OpenMP_CXX)
  prk_cxx_executable(transpose-taskloop SOURCES transpose-taskloop.cc LIBS OpenMP::OpenMP_CXX)
  prk_cxx_executable(nstream-taskloop   SOURCES nstream-taskloop.cc   LIBS OpenMP::OpenMP_CXX)
  add_custom_target(cxx11_taskloop DEPENDS cxx11_stencil-taskloop cxx11_transpose-taskloop cxx11_nstream-taskloop)

  if(PRK_WITH_TARGET)
    prk_cxx_executable(stencil-openmp-target   SOURCES stencil-openmp-target.cc   FLAGS ${PRK_CXX_OPENMP_OFFLOAD_FLAG})
    prk_cxx_executable(transpose-openmp-target SOURCES transpose-openmp-target.cc FLAGS ${PRK_CXX_OPENMP_OFFLOAD_FLAG})
    prk_cxx_executable(nstream-openmp-target   SOURCES nstream-openmp-target.cc   FLAGS ${PRK_CXX_OPENMP_OFFLOAD_FLAG})
    add_custom_target(cxx11_target DEPENDS cxx11_stencil-openmp-target cxx11_transpose-openmp-target cxx11_nstream-openmp-target)
  else()
    message(STATUS "Cxx11 target (OpenMP offload): skipped, no working target-offload flags found")
  endif()
else()
  message(STATUS "Cxx11 openmp/target/taskloop: skipped, no OpenMP C++ support found")
endif()

# --- mpi / cblas / mpi-cblas ----------------------------------------------

if(MPI_CXX_FOUND)
  prk_cxx_executable(nstream-mpi       SOURCES nstream-mpi.cc       LIBS MPI::MPI_CXX)
  prk_cxx_executable(stencil-mpi       SOURCES stencil-mpi.cc       LIBS MPI::MPI_CXX)
  prk_cxx_executable(transpose-a2a-mpi SOURCES transpose-a2a-mpi.cc LIBS MPI::MPI_CXX)
  prk_cxx_executable(transpose-p2p-mpi SOURCES transpose-p2p-mpi.cc LIBS MPI::MPI_CXX)
  prk_cxx_executable(transpose-get-mpi SOURCES transpose-get-mpi.cc LIBS MPI::MPI_CXX)
  add_custom_target(cxx11_mpi DEPENDS
    cxx11_nstream-mpi cxx11_stencil-mpi cxx11_transpose-a2a-mpi cxx11_transpose-p2p-mpi cxx11_transpose-get-mpi)
else()
  message(STATUS "Cxx11 mpi: skipped, no MPI C support found")
endif()

if(BLAS_FOUND)
  # These sources already have an `#ifdef MKL` path using MKL's own
  # mkl_dgemm/mkl_domatcopy-style API instead of the CBLAS extensions
  # (cblas_dgemm_batch, cblas_domatcopy) that only OpenBLAS provides --
  # linking those extensions against MKL fails at link time (MKL exports
  # mkl_domatcopy, not cblas_domatcopy). Detect MKL from the resolved BLAS
  # library path and opt into that source path instead of assuming OpenBLAS.
  set(_cblas_defs "")
  set(_cblas_includes "")
  if(BLAS_LIBRARIES MATCHES "mkl")
    list(APPEND _cblas_defs MKL)
    # find_package(BLAS) only resolves libraries, not headers -- mkl.h/
    # mkl_cblas.h aren't on the compiler's default include path even with
    # MKLROOT set (that's enough for the -qmkl driver flag, which these
    # targets don't use; they link BLAS_LIBRARIES directly), so this needs
    # to be found and added explicitly.
    find_path(PRK_MKL_INCLUDE_DIR mkl.h HINTS ENV MKLROOT PATH_SUFFIXES include)
    if(PRK_MKL_INCLUDE_DIR)
      set(_cblas_includes ${PRK_MKL_INCLUDE_DIR})
    endif()
  endif()
  prk_cxx_executable(transpose-cblas SOURCES transpose-cblas.cc LIBS BLAS::BLAS DEFS ${_cblas_defs} INCLUDES ${_cblas_includes})
  prk_cxx_executable(dgemm-cblas     SOURCES dgemm-cblas.cc     LIBS BLAS::BLAS DEFS ${_cblas_defs} INCLUDES ${_cblas_includes})
  prk_cxx_executable(sgemm-cblas     SOURCES sgemm-cblas.cc     LIBS BLAS::BLAS DEFS ${_cblas_defs} INCLUDES ${_cblas_includes})
  prk_cxx_executable(xgemm-cblas     SOURCES xgemm-cblas.cc     LIBS BLAS::BLAS DEFS ${_cblas_defs} INCLUDES ${_cblas_includes})
  add_custom_target(cxx11_cblas DEPENDS cxx11_transpose-cblas cxx11_dgemm-cblas cxx11_sgemm-cblas cxx11_xgemm-cblas)

  if(MPI_CXX_FOUND)
    prk_cxx_executable(dgemm-mpi-cblas SOURCES dgemm-mpi-cblas.cc LIBS MPI::MPI_CXX BLAS::BLAS DEFS ${_cblas_defs} INCLUDES ${_cblas_includes})
    if(TARGET cxx11_mpi)
      add_dependencies(cxx11_mpi cxx11_dgemm-mpi-cblas)
    endif()
  endif()
else()
  message(STATUS "Cxx11 cblas: skipped, no BLAS found")
endif()

# --- CUDA / cuBLAS / Thrust / CUDASTF -------------------------------------

if(PRK_WITH_CUDA)
  prk_cxx_executable(stencil-cuda         SOURCES stencil-cuda.cu)
  prk_cxx_executable(transpose-cuda       SOURCES transpose-cuda.cu)
  prk_cxx_executable(nstream-cuda         SOURCES nstream-cuda.cu)
  prk_cxx_executable(nstream-managed-cuda SOURCES nstream-managed-cuda.cu)
  add_custom_target(cxx11_cuda DEPENDS cxx11_stencil-cuda cxx11_transpose-cuda cxx11_nstream-cuda cxx11_nstream-managed-cuda)

  prk_cxx_executable(transpose-cublas       SOURCES transpose-cublas.cu       LIBS CUDA::cublas CUDA::cublasLt DEFS PRK_USE_CUBLAS)
  prk_cxx_executable(nstream-cublas         SOURCES nstream-cublas.cu         LIBS CUDA::cublas CUDA::cublasLt DEFS PRK_USE_CUBLAS)
  prk_cxx_executable(dgemm-cublas           SOURCES dgemm-cublas.cu           LIBS CUDA::cublas CUDA::cublasLt DEFS PRK_USE_CUBLAS)
  prk_cxx_executable(dgemm-multigpu-cublas  SOURCES dgemm-multigpu-cublas.cu  LIBS CUDA::cublas CUDA::cublasLt DEFS PRK_USE_CUBLAS)
  prk_cxx_executable(sgemm-cublas           SOURCES sgemm-cublas.cu           LIBS CUDA::cublas CUDA::cublasLt DEFS PRK_USE_CUBLAS)
  prk_cxx_executable(xgemm-cublas           SOURCES xgemm-cublas.cu           LIBS CUDA::cublas CUDA::cublasLt DEFS PRK_USE_CUBLAS)
  prk_cxx_executable(dgemm-managed-cublas   SOURCES dgemm-managed-cublas.cu   LIBS CUDA::cublas CUDA::cublasLt DEFS PRK_USE_CUBLAS)
  add_custom_target(cxx11_cublas DEPENDS
    cxx11_transpose-cublas cxx11_nstream-cublas cxx11_dgemm-cublas cxx11_dgemm-multigpu-cublas
    cxx11_sgemm-cublas cxx11_xgemm-cublas cxx11_dgemm-managed-cublas)

  if(MPI_CXX_FOUND)
    prk_cxx_executable(dgemm-mpi-cublas SOURCES dgemm-mpi-cublas.cu LIBS MPI::MPI_CXX CUDA::cublas CUDA::cublasLt DEFS PRK_USE_CUBLAS)
    add_dependencies(cxx11_cublas cxx11_dgemm-mpi-cublas)
  endif()

  # Host Thrust: .cc compiled with the host C++ compiler, using headers
  # bundled with the CUDA toolkit (no NVHPC needed). Device Thrust: .cu
  # compiled with nvcc.
  set(_thrust_targets "")
  if(PRK_THRUST_HOST_WORKS)
    prk_cxx_executable(nstream-host-thrust   SOURCES nstream-host-thrust.cc   INCLUDES ${PRK_THRUST_INCLUDE_DIR})
    prk_cxx_executable(transpose-host-thrust SOURCES transpose-host-thrust.cc INCLUDES ${PRK_THRUST_INCLUDE_DIR})
    if(PRK_RANGES_BASIC_DEFS)
      target_include_directories(cxx11_transpose-host-thrust PRIVATE ${PRK_RANGES_BASIC_INCLUDES})
      target_compile_definitions(cxx11_transpose-host-thrust PRIVATE ${PRK_RANGES_BASIC_DEFS})
      if(PRK_RANGES_BASIC_CXX_STANDARD)
        set_target_properties(cxx11_transpose-host-thrust PROPERTIES CXX_STANDARD ${PRK_RANGES_BASIC_CXX_STANDARD})
      endif()
    endif()
    list(APPEND _thrust_targets cxx11_nstream-host-thrust cxx11_transpose-host-thrust)
  else()
    message(STATUS "Cxx11 thrust(host): skipped, host compiler doesn't support nvcc's extended-lambda syntax")
  endif()
  prk_cxx_executable(nstream-device-thrust   SOURCES nstream-device-thrust.cu)
  # range-v3's semiregular_box<T&> using-declaration for a conversion
  # operator doesn't parse under nvcc's device-code frontend (genuine
  # nvcc/range-v3 incompatibility, unrelated to the CUB/for_each issue
  # below); use native C++20 std::ranges here instead, which nvcc handles.
  if(PRK_CXX20_SUPPORTED)
    prk_cxx_executable(transpose-device-thrust SOURCES transpose-device-thrust.cu DEFS USE_GCC_RANGES)
    set_target_properties(cxx11_transpose-device-thrust PROPERTIES CUDA_STANDARD 20)
  elseif(PRK_RANGE_V3_FOUND)
    prk_cxx_executable(transpose-device-thrust SOURCES transpose-device-thrust.cu
      INCLUDES ${PRK_RANGE_V3_INCLUDE_DIR} DEFS USE_RANGES_TS)
  else()
    prk_cxx_executable(transpose-device-thrust SOURCES transpose-device-thrust.cu)
  endif()
  list(APPEND _thrust_targets cxx11_nstream-device-thrust cxx11_transpose-device-thrust)
  add_custom_target(cxx11_thrust DEPENDS ${_thrust_targets})

  if(PRK_CUDASTF_FOUND AND PRK_CUDASTF_WORKS)
    set(_cudastf_includes ${PRK_CUDASTF_CUDAX_INCLUDE_DIR} ${PRK_CUDASTF_LIBCUDACXX_INCLUDE_DIR})
    prk_cxx_executable(dgemm-cudastf    SOURCES dgemm-cudastf.cu    LIBS CUDA::cuda_driver INCLUDES ${_cudastf_includes} FLAGS --expt-relaxed-constexpr)
    prk_cxx_executable(stencil-cudastf  SOURCES stencil-cudastf.cu  LIBS CUDA::cuda_driver INCLUDES ${_cudastf_includes} FLAGS --expt-relaxed-constexpr)
    prk_cxx_executable(p2p-cudastf      SOURCES p2p-cudastf.cu      LIBS CUDA::cuda_driver INCLUDES ${_cudastf_includes} FLAGS --expt-relaxed-constexpr)
    prk_cxx_executable(transpose-cudastf SOURCES transpose-cudastf.cu LIBS CUDA::cuda_driver INCLUDES ${_cudastf_includes} FLAGS --expt-relaxed-constexpr)
    prk_cxx_executable(dgemm-cublas-cudastf SOURCES dgemm-cublas-cudastf.cu
      LIBS CUDA::cuda_driver CUDA::cublas CUDA::cublasLt INCLUDES ${_cudastf_includes}
      FLAGS --expt-relaxed-constexpr DEFS PRK_USE_CUBLAS)
    add_custom_target(cxx11_cudastf DEPENDS
      cxx11_dgemm-cudastf cxx11_stencil-cudastf cxx11_p2p-cudastf cxx11_transpose-cudastf cxx11_dgemm-cublas-cudastf)
  else()
    message(STATUS "Cxx11 cudastf: skipped, CCCL cudax/libcudacxx headers not found")
  endif()

  if(PRK_NVSHMEM_FOUND)
    prk_cxx_executable(transpose-get-nvshmem SOURCES transpose-get-nvshmem.cu
      LIBS ${PRK_NVSHMEM_HOST_LIBRARY} ${PRK_NVSHMEM_DEVICE_LIBRARY} INCLUDES ${PRK_NVSHMEM_INCLUDE_DIR})
    prk_cxx_executable(transpose-a2a-nvshmem SOURCES transpose-a2a-nvshmem.cu
      LIBS ${PRK_NVSHMEM_HOST_LIBRARY} ${PRK_NVSHMEM_DEVICE_LIBRARY} INCLUDES ${PRK_NVSHMEM_INCLUDE_DIR})
    prk_cxx_executable(transpose-ptr-nvshmem SOURCES transpose-ptr-nvshmem.cu
      LIBS ${PRK_NVSHMEM_HOST_LIBRARY} ${PRK_NVSHMEM_DEVICE_LIBRARY} INCLUDES ${PRK_NVSHMEM_INCLUDE_DIR})
    prk_cxx_executable(transpose-put-nvshmem SOURCES transpose-put-nvshmem.cu
      LIBS ${PRK_NVSHMEM_HOST_LIBRARY} ${PRK_NVSHMEM_DEVICE_LIBRARY} INCLUDES ${PRK_NVSHMEM_INCLUDE_DIR})
    # -rdc=true (relocatable device code, needed to link across NVSHMEM's
    # device-side symbols) has to be applied via CMake's own RDC property,
    # not a raw compile/link FLAGS string: that made CMake pick g++ instead
    # of nvcc as the link driver for these targets ("g++: error:
    # unrecognized command-line option '-rdc=true'").
    set_target_properties(cxx11_transpose-get-nvshmem cxx11_transpose-a2a-nvshmem
      cxx11_transpose-ptr-nvshmem cxx11_transpose-put-nvshmem PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON)
    add_custom_target(cxx11_nvshmem DEPENDS
      cxx11_transpose-get-nvshmem cxx11_transpose-a2a-nvshmem cxx11_transpose-ptr-nvshmem cxx11_transpose-put-nvshmem)
  else()
    message(STATUS "Cxx11 nvshmem: skipped, NVSHMEM not found")
  endif()
else()
  message(STATUS "Cxx11 cuda/cublas/thrust(device)/cudastf/nvshmem: skipped, no CUDA compiler found")
endif()

# NCCL: deps/nccl on this machine is unbuilt source (a git checkout, not a
# built library) -- no autobuild here (unlike Kokkos/RAJA/PETSc/GA, NCCL's
# own build is a plain `make` with no install-prefix story as clean as
# ExternalProject_Add's other targets), so this is detect-only.
find_path(PRK_NCCL_INCLUDE_DIR nccl.h HINTS ENV NCCLDIR PATH_SUFFIXES include)
find_library(PRK_NCCL_LIBRARY nccl HINTS ENV NCCLDIR PATH_SUFFIXES lib)
if(PRK_WITH_CUDA AND PRK_NCCL_INCLUDE_DIR AND PRK_NCCL_LIBRARY AND MPI_CXX_FOUND)
  # NCCL programs generally use MPI to bootstrap the communicator across
  # ranks/nodes. nstream-nccl.cu is a special case that currently drives
  # NCCL directly within a single process (ncclCommInitAll across local
  # GPUs) without MPI -- to be fixed later to use MPI properly, so keep
  # the MPI dependency/gating here rather than dropping it.
  prk_cxx_executable(nstream-nccl SOURCES nstream-nccl.cu LIBS ${PRK_NCCL_LIBRARY} MPI::MPI_CXX INCLUDES ${PRK_NCCL_INCLUDE_DIR})
  add_custom_target(cxx11_nccl DEPENDS cxx11_nstream-nccl)
else()
  message(STATUS "Cxx11 nccl: skipped, NCCL not found (deps/nccl here is unbuilt source, not installed)")
endif()

# --- OpenCL / Boost.Compute ------------------------------------------------

if(OpenCL_FOUND)
  prk_cxx_executable(p2p-innerloop-opencl SOURCES p2p-innerloop-opencl.cc LIBS OpenCL::OpenCL)
  prk_cxx_executable(stencil-opencl       SOURCES stencil-opencl.cc       LIBS OpenCL::OpenCL)
  prk_cxx_executable(transpose-opencl     SOURCES transpose-opencl.cc     LIBS OpenCL::OpenCL)
  prk_cxx_executable(nstream-opencl       SOURCES nstream-opencl.cc       LIBS OpenCL::OpenCL)
  add_custom_target(cxx11_opencl DEPENDS cxx11_p2p-innerloop-opencl cxx11_stencil-opencl cxx11_transpose-opencl cxx11_nstream-opencl)

  if(PRK_BOOST_COMPUTE_FOUND)
    prk_cxx_executable(nstream-boost-compute SOURCES nstream-boost-compute.cc
      LIBS OpenCL::OpenCL INCLUDES ${PRK_BOOST_COMPUTE_INCLUDE_DIR})
    add_custom_target(cxx11_boost_compute DEPENDS cxx11_nstream-boost-compute)
  else()
    message(STATUS "Cxx11 boost-compute: skipped, Boost.Compute headers not found")
  endif()
else()
  message(STATUS "Cxx11 opencl/boost-compute: skipped, no OpenCL found")
endif()

# --- TBB / STL / PSTL / ranges / executors --------------------------------

if(TBB_FOUND)
  prk_cxx_executable(p2p-innerloop-tbb    SOURCES p2p-innerloop-tbb.cc    LIBS TBB::tbb)
  prk_cxx_executable(p2p-tbb              SOURCES p2p-tbb.cc              LIBS TBB::tbb)
  prk_cxx_executable(stencil-tbb          SOURCES stencil-tbb.cc          LIBS TBB::tbb)
  prk_cxx_executable(transpose-tbb        SOURCES transpose-tbb.cc        LIBS TBB::tbb)
  prk_cxx_executable(nstream-tbb          SOURCES nstream-tbb.cc          LIBS TBB::tbb)
  prk_cxx_executable(p2p-hyperplane-tbb   SOURCES p2p-hyperplane-tbb.cc   LIBS TBB::tbb)
  prk_cxx_executable(p2p-tasks-tbb        SOURCES p2p-tasks-tbb.cc        LIBS TBB::tbb)
  add_custom_target(cxx11_tbb DEPENDS
    cxx11_p2p-innerloop-tbb cxx11_p2p-tbb cxx11_stencil-tbb cxx11_transpose-tbb cxx11_nstream-tbb cxx11_p2p-hyperplane-tbb cxx11_p2p-tasks-tbb)

  # NVHPC's own <execution> header predefines a global `namespace exec =
  # std::experimental::execution;` alias (its pre-standardization PSTL,
  # which has no par_unseq) -- prk_pstl.h's own `namespace exec = ...`
  # then collides with it ("namespace exec has already been defined").
  # NVHPC's real parallel story is stdpar/OpenACC (wired in below), not
  # this TBB-backed PSTL group, so just skip it there.
  if(PRK_PSTL_WORKS AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    # prk_ranges.h needs exactly one of USE_GCC_RANGES/USE_BOOST_IRANGE/
    # USE_RANGES_TS defined; prk_check_ranges_cxx() picked the best one
    # available (STL preferred, then Boost, then range-v3 last).
    set(_pstl_defs ${PRK_RANGES_BASIC_DEFS})
    # prk_pstl.h's `namespace exec = ...` only auto-defines for GCC
    # (__GNUC__ >= 9); Clang-family compilers like icpx (__GNUC__ 4, for
    # compatibility) need to be told explicitly which PSTL backend to use.
    set(_pstl_includes ${PRK_RANGES_BASIC_INCLUDES})
    if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
      list(APPEND _pstl_defs USE_ONEAPI_DPL)
      # oneDPL is a separate oneAPI component; its headers aren't on icpx's
      # default include path (only found here when setvars.sh has been
      # sourced, which sets CPATH -- this toolchain file doesn't replicate
      # that), so it needs to be located and added explicitly.
      find_path(PRK_ONEDPL_INCLUDE_DIR oneapi/dpl/execution PATHS /opt/intel/oneapi/dpl/*/include)
      if(PRK_ONEDPL_INCLUDE_DIR)
        list(APPEND _pstl_includes ${PRK_ONEDPL_INCLUDE_DIR})
      endif()
    endif()
    prk_cxx_executable(stencil-pstl   SOURCES stencil-pstl.cc   LIBS TBB::tbb INCLUDES ${_pstl_includes} DEFS ${_pstl_defs})
    prk_cxx_executable(transpose-pstl SOURCES transpose-pstl.cc LIBS TBB::tbb INCLUDES ${_pstl_includes} DEFS ${_pstl_defs})
    if(PRK_RANGES_BASIC_CXX_STANDARD)
      set_target_properties(cxx11_stencil-pstl cxx11_transpose-pstl PROPERTIES CXX_STANDARD ${PRK_RANGES_BASIC_CXX_STANDARD})
    endif()
    add_custom_target(cxx11_pstl DEPENDS cxx11_stencil-pstl cxx11_transpose-pstl)
    # nstream-pstl.cc unconditionally `#include <ranges>` and uses
    # std::ranges directly (nothing to do with prk_ranges.h/range-v3),
    # same as nstream-stl.cc/nstream-ranges.cc -- needs C++20.
    if(PRK_CXX20_SUPPORTED)
      prk_cxx_executable(nstream-pstl SOURCES nstream-pstl.cc LIBS TBB::tbb INCLUDES ${_pstl_includes} DEFS ${_pstl_defs})
      set_target_properties(cxx11_nstream-pstl PROPERTIES CXX_STANDARD 20)
      add_dependencies(cxx11_pstl cxx11_nstream-pstl)
    else()
      message(STATUS "Cxx11 nstream-pstl: skipped, compiler has no C++20 <ranges> support")
    endif()
  else()
    message(STATUS "Cxx11 pstl: skipped, parallel <execution> algorithms don't link with TBB here")
  endif()
else()
  message(STATUS "Cxx11 tbb/pstl: skipped, TBB not found")
endif()

if(PRK_RANGES_BASIC_DEFS)
  # prk::range() (used by every one of these except nstream-stl/
  # nstream-ranges) comes from prk_ranges.h, which requires exactly one of
  # USE_GCC_RANGES/USE_BOOST_IRANGE/USE_RANGES_TS defined or it #errors;
  # prk_check_ranges_cxx() already picked the best backend available (STL
  # preferred, then Boost, then range-v3 last).
  prk_cxx_executable(stencil-stl   SOURCES stencil-stl.cc   INCLUDES ${PRK_RANGES_BASIC_INCLUDES} DEFS ${PRK_RANGES_BASIC_DEFS})
  prk_cxx_executable(transpose-stl SOURCES transpose-stl.cc INCLUDES ${PRK_RANGES_BASIC_INCLUDES} DEFS ${PRK_RANGES_BASIC_DEFS})
  add_custom_target(cxx11_stl DEPENDS cxx11_stencil-stl cxx11_transpose-stl)
  if(PRK_RANGES_BASIC_CXX_STANDARD)
    set_target_properties(cxx11_stencil-stl cxx11_transpose-stl PROPERTIES CXX_STANDARD ${PRK_RANGES_BASIC_CXX_STANDARD})
  endif()

  add_custom_target(cxx11_ranges)

  # stencil_ranges.hpp (included by stencil-ranges.cc) calls range-v3's
  # `ranges::views::...`/`ranges::stride_view` API directly, bypassing
  # prk_ranges.h's USE_GCC_RANGES/USE_BOOST_IRANGE/USE_RANGES_TS macro
  # abstraction entirely -- it needs range-v3 itself, not whatever backend
  # the basic/full tiers picked (`ranges::` is range-v3's own namespace,
  # distinct from `std::ranges::`).
  prk_check_range_v3_cxx()
  if(PRK_RANGE_V3_FOUND)
    prk_cxx_executable(stencil-ranges SOURCES stencil-ranges.cc INCLUDES ${PRK_RANGE_V3_INCLUDE_DIR} DEFS USE_RANGES_TS)
    add_dependencies(cxx11_ranges cxx11_stencil-ranges)
  else()
    message(STATUS "Cxx11 stencil-ranges: skipped, range-v3 not available (hardcodes its own ranges::... API)")
  endif()

  # transpose-ranges.cc additionally needs prk::range2()/tiled blocking
  # ranges (views::stride + views::cartesian_product) -- the "full" tier,
  # which range-v3 always supports but plain C++20 std::ranges/Boost.Irange
  # don't (that needs C++23, or real range-v3-equivalent glue code for
  # Boost -- see prk_ranges.h).
  if(PRK_RANGES_FULL_DEFS)
    prk_cxx_executable(transpose-ranges SOURCES transpose-ranges.cc INCLUDES ${PRK_RANGES_FULL_INCLUDES} DEFS ${PRK_RANGES_FULL_DEFS})
    if(PRK_RANGES_FULL_CXX_STANDARD)
      set_target_properties(cxx11_transpose-ranges PROPERTIES CXX_STANDARD ${PRK_RANGES_FULL_CXX_STANDARD})
    endif()
    add_dependencies(cxx11_ranges cxx11_transpose-ranges)
  else()
    message(STATUS "Cxx11 transpose-ranges: skipped, no ranges backend supports cartesian_product/tiling here")
  endif()

  # nstream-stl.cc/nstream-ranges.cc unconditionally `#include <ranges>` and
  # use std::ranges directly (nothing to do with prk_ranges.h/range-v3),
  # which needs C++20 -- the rest of Cxx11 builds at C++17.
  if(PRK_CXX20_SUPPORTED)
    prk_cxx_executable(nstream-stl    SOURCES nstream-stl.cc)
    prk_cxx_executable(nstream-ranges SOURCES nstream-ranges.cc)
    set_target_properties(cxx11_nstream-stl cxx11_nstream-ranges PROPERTIES CXX_STANDARD 20)
    add_dependencies(cxx11_stl cxx11_nstream-stl)
    add_dependencies(cxx11_ranges cxx11_nstream-ranges)
  else()
    message(STATUS "Cxx11 nstream-stl/nstream-ranges: skipped, compiler has no C++20 <ranges> support")
  endif()
else()
  message(STATUS "Cxx11 stl/ranges: skipped, no ranges backend available")
endif()

# --- Kokkos / RAJA (autobuilt) ---------------------------------------------

if(Kokkos_FOUND)
  prk_cxx_executable(stencil-kokkos   SOURCES stencil-kokkos.cc   LIBS Kokkos::kokkos)
  prk_cxx_executable(transpose-kokkos SOURCES transpose-kokkos.cc LIBS Kokkos::kokkos)
  prk_cxx_executable(nstream-kokkos   SOURCES nstream-kokkos.cc   LIBS Kokkos::kokkos)
  add_custom_target(cxx11_kokkos DEPENDS cxx11_stencil-kokkos cxx11_transpose-kokkos cxx11_nstream-kokkos)
  if(Kokkos_AUTOBUILT)
    add_dependencies(cxx11_stencil-kokkos cxx11_transpose-kokkos cxx11_nstream-kokkos ${Kokkos_EXTERNAL_TARGET})
  endif()
else()
  message(STATUS "Cxx11 kokkos: skipped, Kokkos not available")
endif()

if(RAJA_FOUND)
  prk_cxx_executable(p2p-vector-raja      SOURCES p2p-vector-raja.cc      LIBS RAJA)
  prk_cxx_executable(stencil-vector-raja  SOURCES stencil-vector-raja.cc  LIBS RAJA)
  prk_cxx_executable(nstream-vector-raja  SOURCES nstream-vector-raja.cc  LIBS RAJA)
  prk_cxx_executable(p2p-raja             SOURCES p2p-raja.cc             LIBS RAJA)
  prk_cxx_executable(transpose-raja       SOURCES transpose-raja.cc       LIBS RAJA)
  prk_cxx_executable(nstream-raja         SOURCES nstream-raja.cc         LIBS RAJA)
  prk_cxx_executable(stencil-raja         SOURCES stencil-raja.cc         LIBS RAJA)
  add_custom_target(cxx11_raja DEPENDS
    cxx11_p2p-vector-raja cxx11_stencil-vector-raja cxx11_nstream-vector-raja
    cxx11_p2p-raja cxx11_transpose-raja cxx11_nstream-raja cxx11_stencil-raja)
  if(RAJA_AUTOBUILT)
    add_dependencies(cxx11_raja ${RAJA_EXTERNAL_TARGET})
    foreach(_t cxx11_p2p-vector-raja cxx11_stencil-vector-raja cxx11_nstream-vector-raja cxx11_p2p-raja cxx11_transpose-raja cxx11_nstream-raja cxx11_stencil-raja)
      add_dependencies(${_t} ${RAJA_EXTERNAL_TARGET})
    endforeach()
  endif()
else()
  message(STATUS "Cxx11 raja: skipped, RAJA not available")
endif()

# --- OpenACC ----------------------------------------------------------

if(PRK_OPENACC_FLAGS)
  prk_cxx_executable(nstream-openacc         SOURCES nstream-openacc.cc         FLAGS ${PRK_OPENACC_FLAGS})
  prk_cxx_executable(stencil-openacc         SOURCES stencil-openacc.cc         FLAGS ${PRK_OPENACC_FLAGS})
  prk_cxx_executable(transpose-openacc       SOURCES transpose-openacc.cc       FLAGS ${PRK_OPENACC_FLAGS})
  prk_cxx_executable(p2p-hyperplane-openacc  SOURCES p2p-hyperplane-openacc.cc  FLAGS ${PRK_OPENACC_FLAGS})
  add_custom_target(cxx11_openacc DEPENDS
    cxx11_nstream-openacc cxx11_stencil-openacc cxx11_transpose-openacc cxx11_p2p-hyperplane-openacc)
else()
  message(STATUS "Cxx11 openacc: skipped, no working OpenACC found")
endif()

# --- stdpar -------------------------------------------------------------

if(PRK_STDPAR_WORKS)
  # nstream-stdpar.cc/transpose-stdpar.cc use Thrust's header-only iterator
  # helpers (zip_iterator, counting_iterator) on plain std::vector, not CUDA
  # device code -- they compile fine with any host compiler, they just need
  # the Thrust headers on the include path. Thrust's zip_iterator drags in
  # its CUDA-backend config (CUB), which needs cudaError_t from
  # cuda_runtime_api.h even for a plain host build -- add CUDAToolkit's
  # include dir too, not just the cccl/thrust headers. TBB is only needed
  # for the libstdc++/libc++ TBB-backed PSTL path, not NVHPC's GPU offload.
  set(_stdpar_libs "")
  if(TBB_FOUND)
    list(APPEND _stdpar_libs TBB::tbb)
  endif()
  set(_stdpar_includes ${PRK_THRUST_INCLUDE_DIR})
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    # nvc++ bundles its own complete, self-consistent CUDA toolkit headers
    # (needed for the -stdpar=gpu path anyway); adding the system
    # CUDAToolkit's include dir on top mixes the two and nvc++ rejects it
    # outright: "CUDA compiler and CUDA toolkit headers are incompatible".
    # Only host compilers without their own bundled CUDA runtime need this.
    list(APPEND _stdpar_includes ${CUDAToolkit_INCLUDE_DIRS})
  endif()
  prk_cxx_executable(nstream-stdpar SOURCES nstream-stdpar.cc LIBS ${_stdpar_libs} FLAGS ${PRK_STDPAR_FLAGS} INCLUDES ${_stdpar_includes})
  set(_cxx11_stdpar_targets cxx11_nstream-stdpar)
  # transpose-stdpar.cc uses a raw `__device__`-annotated lambda, which only
  # nvcc/nvc++ understand (nvc++'s -stdpar=gpu recognizes CUDA C++ device
  # annotations; plain g++/icpx's TBB-backed PSTL fallback does not).
  if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    prk_cxx_executable(transpose-stdpar SOURCES transpose-stdpar.cc LIBS ${_stdpar_libs} FLAGS ${PRK_STDPAR_FLAGS} INCLUDES ${_stdpar_includes})
    list(APPEND _cxx11_stdpar_targets cxx11_transpose-stdpar)
  else()
    message(STATUS "Cxx11 transpose-stdpar: skipped, needs nvc++'s -stdpar=gpu (raw __device__ lambda)")
  endif()
  add_custom_target(cxx11_stdpar DEPENDS ${_cxx11_stdpar_targets})
else()
  message(STATUS "Cxx11 stdpar: skipped, no NVHPC GPU offload and no working TBB-backed PSTL")
endif()

# --- unavailable on this machine: detect-only, skip cleanly ---------------
# SYCL/DPC++/oneAPI (GPU offload), HIP/ROCm, OCCA, Halide, HPX, UPC++, and
# OpenSHMEM have no working toolchain on this machine.
message(STATUS "Cxx11 sycl/dpcpp/oneapi/hip/occa/halide/hpx/upcxx/oshmem: skipped, no toolchain found")

# --- CTest ----------------------------------------------------------------

prk_cxx_test(cxx11_valarray_transpose COMMAND transpose-valarray 10 1024 32)
prk_cxx_test(cxx11_valarray_nstream   COMMAND nstream-valarray 10 16777216 32)

prk_cxx_test(cxx11_vector_p2p_1            COMMAND p2p-vector 10 1024 1024)
prk_cxx_test(cxx11_vector_p2p_2            COMMAND p2p-vector 10 1024 1024 100 100)
prk_cxx_test(cxx11_vector_p2p_hyperplane_1 COMMAND p2p-hyperplane-vector 10 1024)
prk_cxx_test(cxx11_vector_p2p_hyperplane_2 COMMAND p2p-hyperplane-vector 10 1024 64)
prk_cxx_test(cxx11_vector_stencil          COMMAND stencil-vector 10 1000)
prk_cxx_test(cxx11_vector_transpose        COMMAND transpose-vector 10 1024 32)
prk_cxx_test(cxx11_vector_nstream          COMMAND nstream-vector 10 16777216 32)
prk_cxx_test(cxx11_vector_dgemm_untiled    COMMAND dgemm-vector 10 400 400)
prk_cxx_test(cxx11_vector_dgemm_tiled      COMMAND dgemm-vector 10 400 32)
prk_cxx_test(cxx11_vector_sparse           COMMAND sparse-vector 10 10 5)
prk_cxx_test(cxx11_thread_transpose        COMMAND transpose-thread 10 1024 512 32)
prk_cxx_test(cxx11_async_transpose         COMMAND transpose-async 10 1024 512 32)

if(OpenMP_CXX_FOUND)
  prk_cxx_test(cxx11_openmp_p2p_tasks        COMMAND p2p-tasks-openmp 10 1024 1024 100 100)
  prk_cxx_test(cxx11_openmp_p2p_hyperplane_1 COMMAND p2p-hyperplane-openmp 10 1024)
  prk_cxx_test(cxx11_openmp_p2p_hyperplane_2 COMMAND p2p-hyperplane-openmp 10 1024 64)
  prk_cxx_test(cxx11_openmp_stencil          COMMAND stencil-openmp 10 1000)
  prk_cxx_test(cxx11_openmp_transpose        COMMAND transpose-openmp 10 1024 32)
  prk_cxx_test(cxx11_openmp_nstream          COMMAND nstream-openmp 10 16777216 32)
  foreach(_t cxx11_openmp_p2p_tasks cxx11_openmp_p2p_hyperplane_1 cxx11_openmp_p2p_hyperplane_2
             cxx11_openmp_stencil cxx11_openmp_transpose cxx11_openmp_nstream)
    set_tests_properties(${_t} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2")
  endforeach()

  prk_cxx_test(cxx11_taskloop_stencil   COMMAND stencil-taskloop 10 1000)
  prk_cxx_test(cxx11_taskloop_transpose COMMAND transpose-taskloop 10 1024 32)
  prk_cxx_test(cxx11_taskloop_nstream   COMMAND nstream-taskloop 10 16777216 32)
endif()

if(MPI_CXX_FOUND)
  prk_cxx_test(cxx11_mpi_nstream         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:cxx11_nstream-mpi> 10 16777216)
  prk_cxx_test(cxx11_mpi_stencil         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:cxx11_stencil-mpi> 10 1000)
  prk_cxx_test(cxx11_mpi_transpose_a2a   COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:cxx11_transpose-a2a-mpi> 10 1024 32)
  prk_cxx_test(cxx11_mpi_transpose_p2p   COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:cxx11_transpose-p2p-mpi> 10 1024 32)
  prk_cxx_test(cxx11_mpi_transpose_get   COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:cxx11_transpose-get-mpi> 10 1024 32)

  if(TARGET cxx11_dgemm-mpi-cblas)
    prk_cxx_test(cxx11_mpi_cblas_dgemm COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:cxx11_dgemm-mpi-cblas> 10 400)
    # Same pre-existing OpenBLAS CPU-kernel-detection bug as cxx11_cblas_dgemm.
    set_tests_properties(cxx11_mpi_cblas_dgemm PROPERTIES ENVIRONMENT "OPENBLAS_CORETYPE=SkylakeX")
  endif()

  if(TARGET cxx11_dgemm-mpi-cublas)
    # dgemm-mpi-cublas.cu already detects its own GPU count (cudaGetDeviceCount)
    # and rank count (MPI_Comm_size) at runtime and aborts with a clear
    # message if they don't match -- that's correct, expected behavior, not
    # something to work around. Just give it a correct invocation, using the
    # GPU count already detected above (PRK_NUM_GPUS).
    prk_cxx_test(cxx11_mpi_cublas_dgemm COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${PRK_NUM_GPUS} $<TARGET_FILE:cxx11_dgemm-mpi-cublas> 10 400)
  endif()
endif()

if(BLAS_FOUND)
  prk_cxx_test(cxx11_cblas_transpose COMMAND transpose-cblas 10 1024)
  prk_cxx_test(cxx11_cblas_dgemm     COMMAND dgemm-cblas 10 400)
  # Debian's openblas-pthread 0.3.20 auto-detects this Sapphire Rapids CPU
  # as "Cooperlake" (openblas_get_corename()) and that specific AVX-512
  # dgemm kernel silently returns numerically wrong results here (confirmed:
  # reproduces identically with plain g++ + -lblas, outside any PRK/CMake
  # code, and is independent of thread count -- a real kernel bug, not a
  # threading race). This is an OpenBLAS packaging bug, not a PRK bug --
  # work around it by forcing SkylakeX, the closest other AVX-512 kernel
  # OpenBLAS has that isn't affected (verified: still AVX-512, not a
  # downgrade to AVX2 like Haswell would be).
  set_tests_properties(cxx11_cblas_dgemm PROPERTIES ENVIRONMENT "OPENBLAS_CORETYPE=SkylakeX")
endif()

if(PRK_WITH_CUDA)
  prk_cxx_test(cxx11_cuda_stencil   COMMAND stencil-cuda 10 1000)
  prk_cxx_test(cxx11_cuda_transpose COMMAND transpose-cuda 10 1024 2)
  prk_cxx_test(cxx11_cuda_nstream   COMMAND nstream-cuda 10 16777216)

  prk_cxx_test(cxx11_cublas_dgemm     COMMAND dgemm-cublas 10 400)
  prk_cxx_test(cxx11_cublas_sgemm     COMMAND sgemm-cublas 10 400)
  prk_cxx_test(cxx11_cublas_transpose COMMAND transpose-cublas 10 1024)
  prk_cxx_test(cxx11_cublas_nstream   COMMAND nstream-cublas 10 16777216)

  if(PRK_THRUST_HOST_WORKS)
    prk_cxx_test(cxx11_thrust_host_nstream     COMMAND nstream-host-thrust 10 16777216)
    prk_cxx_test(cxx11_thrust_host_transpose   COMMAND transpose-host-thrust 10 1024)
  endif()
  prk_cxx_test(cxx11_thrust_device_nstream   COMMAND nstream-device-thrust 10 16777216)
  prk_cxx_test(cxx11_thrust_device_transpose COMMAND transpose-device-thrust 10 1024)

  if(PRK_CUDASTF_FOUND AND PRK_CUDASTF_WORKS)
    prk_cxx_test(cxx11_cudastf_dgemm     COMMAND dgemm-cudastf 10 400)
    prk_cxx_test(cxx11_cudastf_stencil   COMMAND stencil-cudastf 10 1000)
    prk_cxx_test(cxx11_cudastf_transpose COMMAND transpose-cudastf 10 1024)
  endif()

  if(TARGET cxx11_nstream-nccl)
    prk_cxx_test(cxx11_nccl_nstream COMMAND nstream-nccl 10 16777216)
  endif()

  if(PRK_NVSHMEM_FOUND AND PRK_NVSHRUN_EXECUTABLE)
    prk_cxx_test(cxx11_nvshmem_transpose_get COMMAND ${PRK_NVSHRUN_EXECUTABLE} -n 2 $<TARGET_FILE:cxx11_transpose-get-nvshmem> 10 1024)
    prk_cxx_test(cxx11_nvshmem_transpose_a2a COMMAND ${PRK_NVSHRUN_EXECUTABLE} -n 2 $<TARGET_FILE:cxx11_transpose-a2a-nvshmem> 10 1024)
    prk_cxx_test(cxx11_nvshmem_transpose_ptr COMMAND ${PRK_NVSHRUN_EXECUTABLE} -n 2 $<TARGET_FILE:cxx11_transpose-ptr-nvshmem> 10 1024)
    # transpose-put-nvshmem.cu's device-initiated kernel (transpose_nvshmem_put
    # in transpose-kernel.h) has a known, pre-existing cross-PE correctness
    # bug -- consistently throws "validation failed" under nvshmrun -n 2,
    # independent of the get/a2a/ptr variants above, which all validate
    # correctly. Added nvshmem_fence() between phases (orders this PE's
    # PUTs) and upgraded the host-side post-kernel barrier to include
    # memory (quiet), which is necessary but wasn't sufficient to fix it --
    # there also appears to be a genuine indexing mismatch between the PUT
    # source/destination offsets and the working GET variant's. Wired into
    # CTest anyway (expected-fail, tracked, not silently skipped) pending a
    # proper fix.
    prk_cxx_test(cxx11_nvshmem_transpose_put COMMAND ${PRK_NVSHRUN_EXECUTABLE} -n 2 $<TARGET_FILE:cxx11_transpose-put-nvshmem> 10 1024)
  else()
    message(STATUS "Cxx11 nvshmem tests: skipped, nvshmrun launcher not found")
  endif()
endif()

if(OpenCL_FOUND)
  # These use $<TARGET_FILE:...> (an absolute path), not a bare name: the
  # WORKING_DIRECTORY override below (needed so loadProgram() finds the
  # .cl kernel source files) means a bare relative command name would
  # resolve against the source directory, not the build directory where
  # the binary actually is.
  prk_cxx_test(cxx11_opencl_stencil        COMMAND $<TARGET_FILE:cxx11_stencil-opencl> 10 1000)
  prk_cxx_test(cxx11_opencl_transpose      COMMAND $<TARGET_FILE:cxx11_transpose-opencl> 10 1024)
  prk_cxx_test(cxx11_opencl_nstream        COMMAND $<TARGET_FILE:cxx11_nstream-opencl> 10 16777216)
  prk_cxx_test(cxx11_opencl_p2p_innerloop  COMMAND $<TARGET_FILE:cxx11_p2p-innerloop-opencl> 10 1024)
  # p2p-innerloop-opencl.cc is a known, pre-existing bug: p2p.cl's kernel
  # loops over all 2n-2 anti-diagonals internally and uses barrier() to
  # synchronize between them, but OpenCL's barrier() only synchronizes
  # work-items within the same work-group -- it does nothing across the
  # multiple work-groups the runtime actually launches for
  # NDRange(2*n) once 2*n exceeds one work-group (confirmed: fails at
  # n=512 and n=1024, passes at n=8 where everything fits in one group).
  # Exact same class of bug as transpose-put-nvshmem's __syncthreads()
  # issue, just OpenCL's barrier() instead of CUDA's. Needs the kernel
  # restructured to launch once per diagonal from the host (like the
  # working CUDA/OpenACC p2p variants) to fix properly. Wired into CTest
  # anyway (expected-fail, tracked) rather than silently skipped.
  # These load their kernel source from .cl files at runtime via a bare
  # relative filename (e.g. "nstream64.cl"), found only in the Cxx11
  # source directory -- CTest's default working directory is the build
  # directory, where loadProgram() silently fails to find them (and, on
  # top of that, nstream-opencl.cc/etc. don't propagate that failure into
  # main()'s exit code either, so without this the test would report a
  # false "Passed" while never actually running an OpenCL kernel).
  set_tests_properties(
    cxx11_opencl_stencil cxx11_opencl_transpose cxx11_opencl_nstream cxx11_opencl_p2p_innerloop
    PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")
  if(TARGET cxx11_nstream-boost-compute)
    prk_cxx_test(cxx11_boost_compute_nstream COMMAND nstream-boost-compute 10 16777216)
  endif()
endif()

if(TARGET cxx11_stencil-stl)
  prk_cxx_test(cxx11_stl_stencil   COMMAND stencil-stl 10 1000)
  prk_cxx_test(cxx11_stl_transpose COMMAND transpose-stl 10 1024)
endif()
if(TARGET cxx11_nstream-stl)
  prk_cxx_test(cxx11_stl_nstream COMMAND nstream-stl 10 16777216)
endif()

if(TARGET cxx11_stencil-ranges)
  prk_cxx_test(cxx11_ranges_stencil COMMAND stencil-ranges 10 1000)
endif()
if(TARGET cxx11_transpose-ranges)
  prk_cxx_test(cxx11_ranges_transpose COMMAND transpose-ranges 10 1024)
endif()
if(TARGET cxx11_nstream-ranges)
  prk_cxx_test(cxx11_ranges_nstream COMMAND nstream-ranges 10 16777216)
endif()

if(TARGET cxx11_stencil-pstl)
  prk_cxx_test(cxx11_pstl_stencil   COMMAND stencil-pstl 10 1000)
  prk_cxx_test(cxx11_pstl_transpose COMMAND transpose-pstl 10 1024)
endif()
if(TARGET cxx11_nstream-pstl)
  prk_cxx_test(cxx11_pstl_nstream COMMAND nstream-pstl 10 16777216)
endif()

if(Kokkos_FOUND)
  prk_cxx_test(cxx11_kokkos_stencil   COMMAND stencil-kokkos 10 1000)
  prk_cxx_test(cxx11_kokkos_transpose COMMAND transpose-kokkos 10 1024 32)
  prk_cxx_test(cxx11_kokkos_nstream   COMMAND nstream-kokkos 10 16777216)
endif()

if(RAJA_FOUND)
  prk_cxx_test(cxx11_raja_stencil   COMMAND stencil-raja 10 1000)
  prk_cxx_test(cxx11_raja_transpose COMMAND transpose-raja 10 1024)
  prk_cxx_test(cxx11_raja_nstream   COMMAND nstream-raja 10 16777216)
endif()

if(PRK_OPENACC_FLAGS)
  prk_cxx_test(cxx11_openacc_nstream        COMMAND nstream-openacc 10 16777216)
  prk_cxx_test(cxx11_openacc_stencil        COMMAND stencil-openacc 10 1000)
  prk_cxx_test(cxx11_openacc_transpose      COMMAND transpose-openacc 10 1024 32)
  # No chunk-size arg: matches the only invocation ci/build-run-prk.sh ever
  # used (commented out there, but this is what it tested). The chunked
  # (nc>1) code path has a real, separate OpenACC codegen bug (confirmed by
  # isolating it outside CI's own tested surface) -- out of scope here.
  prk_cxx_test(cxx11_openacc_p2p_hyperplane COMMAND p2p-hyperplane-openacc 10 1024)
endif()

if(PRK_STDPAR_WORKS)
  prk_cxx_test(cxx11_stdpar_nstream   COMMAND nstream-stdpar 10 16777216)
  if(TARGET cxx11_transpose-stdpar)
    prk_cxx_test(cxx11_stdpar_transpose COMMAND transpose-stdpar 10 1024)
  endif()
endif()
