#
# CMake port of FORTRAN/Makefile.legacy. Target groups and their
# compiler/toolchain gating mirror the Makefile's EXTRA=... conditionals
# (see FORTRAN/Makefile.legacy and its per-target pattern rules) as closely
# as CMake's find_package()/compiler-ID/try_compile() facilities allow.
#
# Standalone project: `cd FORTRAN && cmake -S . -B build` works on its own
# (in-source `cmake .` works too -- the legacy hand-written Makefile this
# would otherwise collide with has been moved to Makefile.legacy). It's also
# add_subdirectory()'d from the repo root for a combined build; CMake fully
# supports a nested project() in that case.
#

cmake_minimum_required(VERSION 3.18)
project(PRK_Fortran LANGUAGES Fortran)
enable_testing()

# Shared find modules / helper scripts live in ../cmake, one level up from
# this file, regardless of whether this is the top-level project or was
# add_subdirectory()'d from the repo root.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake/modules")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# common/make.defs.gcc: FCFLAGS += -DRADIUS=$(RADIUS) $(STARFLAG), RADIUS defaults to 2.
set(RADIUS 2 CACHE STRING "Stencil radius (see FORTRAN/Makefile RADIUS)")
add_compile_definitions(RADIUS=${RADIUS} STAR)

set(CMAKE_Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mod")

# --- dependency detection -----------------------------------------------

find_package(OpenMP COMPONENTS Fortran)
find_package(MPI COMPONENTS Fortran)
find_package(BLAS)
find_package(LAPACK)
find_package(OpenACC COMPONENTS Fortran)
find_package(OpenCoarrays)
find_package(GlobalArrays)

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/AutobuildDeps.cmake")
prk_autobuild_opencoarrays()
prk_autobuild_global_arrays()

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/PRKFortranFeatures.cmake")
prk_check_coarray_single()
prk_check_coarray_lib()
prk_check_openmp_offload()
prk_check_openacc_links()

set(PRK_WITH_COARRAY FALSE)
if(PRK_COARRAY_LIB_FLAG OR PRK_COARRAY_SINGLE_FLAG)
  set(PRK_WITH_COARRAY TRUE)
endif()
set(PRK_WITH_OFFLOAD FALSE)
if(PRK_OPENMP_OFFLOAD_FLAG)
  set(PRK_WITH_OFFLOAD TRUE)
endif()
set(PRK_WITH_TASKLOOP FALSE)
if(OpenMP_Fortran_FOUND AND CMAKE_Fortran_COMPILER_ID MATCHES "^(GNU|Intel|IntelLLVM)$")
  set(PRK_WITH_TASKLOOP TRUE)
endif()
set(PRK_WITH_STDPAR FALSE)
set(PRK_WITH_CUFORTRAN FALSE)
if(CMAKE_Fortran_COMPILER_ID MATCHES "^(NVHPC|PGI)$")
  set(PRK_WITH_STDPAR TRUE)
  set(PRK_WITH_CUFORTRAN TRUE)
endif()

message(STATUS "FORTRAN: OpenMP ${OpenMP_Fortran_FOUND}, MPI ${MPI_Fortran_FOUND}, BLAS ${BLAS_FOUND}, "
               "OpenACC ${PRK_OPENACC_WORKS}, coarrays ${PRK_WITH_COARRAY}, OMP offload ${PRK_WITH_OFFLOAD}, "
               "taskloop ${PRK_WITH_TASKLOOP}, Global Arrays ${GlobalArrays_FOUND}, "
               "stdpar/CUDA Fortran ${PRK_WITH_STDPAR} (compiler: ${CMAKE_Fortran_COMPILER_ID})")

# --- support modules (mirrors "prk.mod prk_mod.o:" / "prk_mpi.mod prk_mpi_mod.o:") ---

add_library(prk_fortran_mod OBJECT prk_mod.F90)
set_target_properties(prk_fortran_mod PROPERTIES Fortran_MODULE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}")
target_include_directories(prk_fortran_mod INTERFACE "${CMAKE_Fortran_MODULE_DIRECTORY}")

if(MPI_Fortran_FOUND)
  add_library(prk_fortran_mpi_mod OBJECT prk_mpi.F90)
  set_target_properties(prk_fortran_mpi_mod PROPERTIES Fortran_MODULE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}")
  target_include_directories(prk_fortran_mpi_mod INTERFACE "${CMAKE_Fortran_MODULE_DIRECTORY}")
  target_link_libraries(prk_fortran_mpi_mod PUBLIC prk_fortran_mod MPI::MPI_Fortran)
endif()

# --- helper -----------------------------------------------------------

function(prk_fortran_executable name)
  set(multiValueArgs SOURCES EXTRA_SOURCES LIBS FLAGS)
  cmake_parse_arguments(PFE "" "" "${multiValueArgs}" ${ARGN})
  set(_target fortran_${name})
  add_executable(${_target} ${PFE_SOURCES} ${PFE_EXTRA_SOURCES})
  set_target_properties(${_target} PROPERTIES OUTPUT_NAME ${name})
  target_link_libraries(${_target} PRIVATE prk_fortran_mod ${PFE_LIBS})
  if(PFE_FLAGS)
    target_compile_options(${_target} PRIVATE ${PFE_FLAGS})
    target_link_options(${_target} PRIVATE ${PFE_FLAGS})
  endif()
endfunction()

# --- serial (Makefile: "serial: p2p stencil transpose nstream dgemm p2p-innerloop pic pic_soa") ---

prk_fortran_executable(p2p            SOURCES p2p.F90)
prk_fortran_executable(p2p-innerloop  SOURCES p2p-innerloop.F90)
prk_fortran_executable(stencil        SOURCES stencil.F90 EXTRA_SOURCES stencil_serial.F90)
prk_fortran_executable(transpose      SOURCES transpose.F90)
prk_fortran_executable(nstream        SOURCES nstream.F90)
prk_fortran_executable(dgemm          SOURCES dgemm.F90)
prk_fortran_executable(pic            SOURCES pic.F90)
prk_fortran_executable(pic_soa        SOURCES pic_soa.F90)
add_custom_target(fortran_serial DEPENDS fortran_p2p fortran_p2p-innerloop fortran_stencil fortran_transpose fortran_nstream fortran_dgemm fortran_pic fortran_pic_soa)

# --- pretty (Makefile: "pretty: stencil-pretty transpose-pretty nstream-pretty dgemm-pretty") ---

prk_fortran_executable(stencil-pretty   SOURCES stencil-pretty.F90)
prk_fortran_executable(transpose-pretty SOURCES transpose-pretty.F90)
prk_fortran_executable(nstream-pretty   SOURCES nstream-pretty.F90)
if(BLAS_FOUND)
  prk_fortran_executable(dgemm-pretty SOURCES dgemm-pretty.F90 LIBS BLAS::BLAS)
else()
  prk_fortran_executable(dgemm-pretty SOURCES dgemm-pretty.F90)
endif()
add_custom_target(fortran_pretty DEPENDS fortran_stencil-pretty fortran_transpose-pretty fortran_nstream-pretty fortran_dgemm-pretty)

# --- openmp (host) ------------------------------------------------------

if(OpenMP_Fortran_FOUND)
  prk_fortran_executable(p2p-tasks-openmp          SOURCES p2p-tasks-openmp.F90          LIBS OpenMP::OpenMP_Fortran)
  prk_fortran_executable(p2p-innerloop-openmp      SOURCES p2p-innerloop-openmp.F90      LIBS OpenMP::OpenMP_Fortran)
  prk_fortran_executable(stencil-openmp            SOURCES stencil-openmp.F90            LIBS OpenMP::OpenMP_Fortran)
  prk_fortran_executable(transpose-openmp          SOURCES transpose-openmp.F90          LIBS OpenMP::OpenMP_Fortran)
  prk_fortran_executable(transpose-tasks-openmp    SOURCES transpose-tasks-openmp.F90    LIBS OpenMP::OpenMP_Fortran)
  prk_fortran_executable(nstream-openmp            SOURCES nstream-openmp.F90            LIBS OpenMP::OpenMP_Fortran)
  prk_fortran_executable(dgemm-openmp              SOURCES dgemm-openmp.F90              LIBS OpenMP::OpenMP_Fortran)
  prk_fortran_executable(pic-openmp                SOURCES pic-openmp.F90                LIBS OpenMP::OpenMP_Fortran)
  prk_fortran_executable(pic_soa-openmp            SOURCES pic_soa-openmp.F90            LIBS OpenMP::OpenMP_Fortran)
  add_custom_target(fortran_openmp DEPENDS
    fortran_p2p-tasks-openmp fortran_p2p-innerloop-openmp fortran_stencil-openmp fortran_transpose-openmp
    fortran_nstream-openmp fortran_dgemm-openmp fortran_pic-openmp fortran_pic_soa-openmp)
  add_custom_target(fortran_tasks DEPENDS fortran_p2p-tasks-openmp fortran_transpose-tasks-openmp)

  if(PRK_WITH_TASKLOOP)
    prk_fortran_executable(stencil-taskloop-openmp   SOURCES stencil-taskloop-openmp.F90   LIBS OpenMP::OpenMP_Fortran)
    prk_fortran_executable(transpose-taskloop-openmp SOURCES transpose-taskloop-openmp.F90 LIBS OpenMP::OpenMP_Fortran)
    prk_fortran_executable(nstream-taskloop-openmp   SOURCES nstream-taskloop-openmp.F90   LIBS OpenMP::OpenMP_Fortran)
    prk_fortran_executable(dgemm-taskloop-openmp     SOURCES dgemm-taskloop-openmp.F90     LIBS OpenMP::OpenMP_Fortran)
    add_custom_target(fortran_taskloop DEPENDS
      fortran_stencil-taskloop-openmp fortran_transpose-taskloop-openmp fortran_nstream-taskloop-openmp fortran_dgemm-taskloop-openmp)
  endif()
else()
  message(STATUS "FORTRAN openmp/tasks/taskloop: skipped, no OpenMP Fortran support found")
endif()

# --- OpenMP target offload ----------------------------------------------

if(PRK_WITH_OFFLOAD)
  prk_fortran_executable(stencil-openmp-target          SOURCES stencil-openmp-target.F90          FLAGS ${PRK_OPENMP_OFFLOAD_FLAG})
  prk_fortran_executable(transpose-openmp-target         SOURCES transpose-openmp-target.F90         FLAGS ${PRK_OPENMP_OFFLOAD_FLAG})
  prk_fortran_executable(nstream-openmp-target           SOURCES nstream-openmp-target.F90           FLAGS ${PRK_OPENMP_OFFLOAD_FLAG})
  prk_fortran_executable(dgemm-openmp-target             SOURCES dgemm-openmp-target.F90             FLAGS ${PRK_OPENMP_OFFLOAD_FLAG})
  prk_fortran_executable(transpose-openmp-target-loop    SOURCES transpose-openmp-target-loop.F90    FLAGS ${PRK_OPENMP_OFFLOAD_FLAG})
  add_custom_target(fortran_target DEPENDS
    fortran_stencil-openmp-target fortran_transpose-openmp-target fortran_nstream-openmp-target
    fortran_dgemm-openmp-target fortran_transpose-openmp-target-loop)
else()
  message(STATUS "FORTRAN target (OpenMP offload): skipped, no working target-offload flags found")
endif()

# --- coarrays -------------------------------------------------------------

if(PRK_WITH_COARRAY)
  if(PRK_COARRAY_LIB_FLAG)
    set(_prk_coarray_flag "${PRK_COARRAY_LIB_FLAG}")
    set(_prk_coarray_libs OpenCoarrays::OpenCoarrays)
    if(NOT TARGET OpenCoarrays::OpenCoarrays)
      add_library(OpenCoarrays::OpenCoarrays UNKNOWN IMPORTED)
      set_target_properties(OpenCoarrays::OpenCoarrays PROPERTIES IMPORTED_LOCATION "${OpenCoarrays_LIBRARY}")
    endif()
  else()
    set(_prk_coarray_flag "${PRK_COARRAY_SINGLE_FLAG}")
    set(_prk_coarray_libs "")
  endif()

  prk_fortran_executable(nstream-coarray   SOURCES nstream-coarray.F90   FLAGS ${_prk_coarray_flag} LIBS ${_prk_coarray_libs})
  prk_fortran_executable(p2p-coarray       SOURCES p2p-coarray.F90       FLAGS ${_prk_coarray_flag} LIBS ${_prk_coarray_libs})
  prk_fortran_executable(stencil-coarray   SOURCES stencil-coarray.F90   FLAGS ${_prk_coarray_flag} LIBS ${_prk_coarray_libs})
  prk_fortran_executable(transpose-coarray SOURCES transpose-coarray.F90 FLAGS ${_prk_coarray_flag} LIBS ${_prk_coarray_libs})
  add_custom_target(fortran_coarray DEPENDS fortran_nstream-coarray fortran_p2p-coarray fortran_stencil-coarray fortran_transpose-coarray)

  if(OpenCoarrays_AUTOBUILT)
    foreach(_t fortran_nstream-coarray fortran_p2p-coarray fortran_stencil-coarray fortran_transpose-coarray)
      add_dependencies(${_t} ${OpenCoarrays_EXTERNAL_TARGET})
    endforeach()
  endif()
else()
  message(STATUS "FORTRAN coarray: skipped, no coarray support found (single-image or OpenCoarrays)")
endif()

# --- MPI ------------------------------------------------------------------

if(MPI_Fortran_FOUND)
  prk_fortran_executable(nstream-mpi       SOURCES nstream-mpi.F90       LIBS prk_fortran_mpi_mod)
  prk_fortran_executable(transpose-a2a-mpi SOURCES transpose-a2a-mpi.F90 LIBS prk_fortran_mpi_mod)
  prk_fortran_executable(transpose-p2p-mpi SOURCES transpose-p2p-mpi.F90 LIBS prk_fortran_mpi_mod)
  prk_fortran_executable(transpose-get-mpi SOURCES transpose-get-mpi.F90 LIBS prk_fortran_mpi_mod)
  add_custom_target(fortran_mpi DEPENDS fortran_nstream-mpi fortran_transpose-a2a-mpi fortran_transpose-p2p-mpi fortran_transpose-get-mpi)

  if(PRK_OPENACC_WORKS)
    prk_fortran_executable(transpose-acc-mpi SOURCES transpose-acc-mpi.F90 LIBS prk_fortran_mpi_mod OpenACC::OpenACC_Fortran)
    add_dependencies(fortran_mpi fortran_transpose-acc-mpi)
  endif()

  if(OpenMP_Fortran_FOUND)
    prk_fortran_executable(nstream-mpi-openmp SOURCES nstream-mpi.F90 LIBS prk_fortran_mpi_mod OpenMP::OpenMP_Fortran)
    add_custom_target(fortran_mpi_openmp DEPENDS fortran_nstream-mpi-openmp)
  endif()
else()
  message(STATUS "FORTRAN mpi/mpi-openmp: skipped, no MPI Fortran support found")
endif()

# --- OpenACC ----------------------------------------------------------

if(PRK_OPENACC_WORKS)
  prk_fortran_executable(p2p-openacc           SOURCES p2p-openacc.F90           LIBS OpenACC::OpenACC_Fortran)
  prk_fortran_executable(p2p-innerloop-openacc SOURCES p2p-innerloop-openacc.F90 LIBS OpenACC::OpenACC_Fortran)
  prk_fortran_executable(stencil-openacc       SOURCES stencil-openacc.F90       LIBS OpenACC::OpenACC_Fortran)
  prk_fortran_executable(transpose-openacc     SOURCES transpose-openacc.F90     LIBS OpenACC::OpenACC_Fortran)
  prk_fortran_executable(nstream-openacc       SOURCES nstream-openacc.F90       LIBS OpenACC::OpenACC_Fortran)
  add_custom_target(fortran_openacc DEPENDS
    fortran_p2p-openacc fortran_p2p-innerloop-openacc fortran_stencil-openacc fortran_transpose-openacc fortran_nstream-openacc)
else()
  message(STATUS "FORTRAN openacc: skipped, no OpenACC Fortran support found")
endif()

# --- Global Arrays ------------------------------------------------------

if(GlobalArrays_FOUND AND MPI_Fortran_FOUND AND BLAS_FOUND AND LAPACK_FOUND)
  if(NOT TARGET GlobalArrays::GlobalArrays)
    add_library(GlobalArrays::GlobalArrays UNKNOWN IMPORTED)
    set_target_properties(GlobalArrays::GlobalArrays PROPERTIES
      IMPORTED_LOCATION "${GlobalArrays_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${GlobalArrays_INCLUDE_DIR}")
    if(GlobalArrays_ARMCI_LIBRARY)
      set_target_properties(GlobalArrays::GlobalArrays PROPERTIES
        INTERFACE_LINK_LIBRARIES "${GlobalArrays_ARMCI_LIBRARY};MPI::MPI_Fortran")
    endif()
  endif()
  # GA is built with 64-bit integers, but -fdefault-integer-8 (a blanket
  # per-compilation-unit flag) is NOT used to match it: it does not reliably
  # widen every declaration GA's Fortran interface depends on, and separately
  # changes unrelated compiler-generated code. Instead every GA-facing
  # variable/argument in nstream-ga.F90/transpose-ga.F90/dgemm-ga.F90 is
  # explicitly kind=INT64 (or logical(kind=8) for GA's LOGICAL functions),
  # exactly matching GA's Integer=8-byte ABI with no compiler flag needed.
  # libga.a is static, so its own transitive BLAS/LAPACK symbol needs
  # (sgemm_/dgemm_/... from matmul.o, dsygv_/dsyev_/dgetrf_/dgetrs_ from
  # ga_diag_seq.o/ga_solve_seq.o) become our link requirements too.
  set(_prk_ga_libs prk_fortran_mpi_mod GlobalArrays::GlobalArrays LAPACK::LAPACK BLAS::BLAS)

  prk_fortran_executable(nstream-ga   SOURCES nstream-ga.F90   LIBS ${_prk_ga_libs})
  prk_fortran_executable(transpose-ga SOURCES transpose-ga.F90 LIBS ${_prk_ga_libs})
  prk_fortran_executable(dgemm-ga     SOURCES dgemm-ga.F90     LIBS ${_prk_ga_libs})
  add_custom_target(fortran_ga DEPENDS fortran_nstream-ga fortran_transpose-ga fortran_dgemm-ga)

  foreach(_t fortran_nstream-ga fortran_transpose-ga fortran_dgemm-ga)
    if(GlobalArrays_AUTOBUILT)
      add_dependencies(${_t} ${GlobalArrays_EXTERNAL_TARGET})
    endif()
  endforeach()
else()
  message(STATUS "FORTRAN ga: skipped, Global Arrays and/or MPI Fortran not available")
endif()

# --- BLAS -----------------------------------------------------------------

if(BLAS_FOUND)
  set(_prk_blas_flags "")
  if(CMAKE_Fortran_COMPILER_ID MATCHES "^(Intel|IntelLLVM)$")
    set(_prk_blas_flags "-heap-arrays")
  endif()
  prk_fortran_executable(dgemm-blas SOURCES dgemm-blas.F90 LIBS BLAS::BLAS FLAGS ${_prk_blas_flags})
  add_custom_target(fortran_blas DEPENDS fortran_dgemm-blas)
else()
  message(STATUS "FORTRAN blas: skipped, no BLAS found")
endif()

# --- stdpar / CUDA Fortran (NVHPC, PGI only) -----------------------------

if(PRK_WITH_STDPAR)
  prk_fortran_executable(nstream-stdpar   SOURCES nstream-stdpar.F90)
  prk_fortran_executable(stencil-stdpar   SOURCES stencil-stdpar.F90)
  prk_fortran_executable(transpose-stdpar SOURCES transpose-stdpar.F90)
  prk_fortran_executable(dgemm-stdpar     SOURCES dgemm-stdpar.F90)
  add_custom_target(fortran_stdpar DEPENDS fortran_nstream-stdpar fortran_stencil-stdpar fortran_transpose-stdpar fortran_dgemm-stdpar)
endif()
if(PRK_WITH_CUFORTRAN)
  # These use CUDA Fortran's <<<...>>> kernel-launch chevron syntax, which
  # nvfortran only parses with -cuda enabled (matches make.defs.nvhpc's own
  # CUFORTFLAG=-cuda -gpu=managed -acc) -- without it, it's a hard syntax
  # error, not just a missing optimization.
  prk_fortran_executable(nstream-cufortran   SOURCES nstream-cufortran.F90   FLAGS -cuda -gpu=managed -acc)
  prk_fortran_executable(transpose-cufortran SOURCES transpose-cufortran.F90 FLAGS -cuda -gpu=managed -acc)
  add_custom_target(fortran_cufortran DEPENDS fortran_nstream-cufortran fortran_transpose-cufortran)
endif()
if(NOT PRK_WITH_STDPAR AND NOT PRK_WITH_CUFORTRAN)
  message(STATUS "FORTRAN stdpar/cufortran: skipped, compiler is not NVHPC/PGI")
endif()

# --- CTest: reproduces ci/build-run-prk.sh's allfortran) run commands ---

function(prk_fortran_test test_name)
  set(multiValueArgs COMMAND)
  cmake_parse_arguments(PFT "" "" "${multiValueArgs}" ${ARGN})
  add_test(NAME ${test_name} COMMAND ${PFT_COMMAND})
  # A hung/looping kernel (e.g. a coarray/MPI deadlock) must not be able to
  # block the whole suite indefinitely.
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 30)
endfunction()

# serial
prk_fortran_test(fortran_serial_p2p           COMMAND p2p 10 1024 1024)
prk_fortran_test(fortran_serial_p2p_innerloop COMMAND p2p-innerloop 10 1024)
prk_fortran_test(fortran_serial_stencil       COMMAND stencil 10 1000)
prk_fortran_test(fortran_serial_transpose_1   COMMAND transpose 10 1024 1)
prk_fortran_test(fortran_serial_transpose_32  COMMAND transpose 10 1024 32)
prk_fortran_test(fortran_serial_nstream       COMMAND nstream 10 16777216)
prk_fortran_test(fortran_serial_dgemm_untiled COMMAND dgemm 10 400 400)
prk_fortran_test(fortran_serial_dgemm_tiled   COMMAND dgemm 10 400 32)

# pretty
prk_fortran_test(fortran_pretty_stencil   COMMAND stencil-pretty 10 1000)
prk_fortran_test(fortran_pretty_transpose COMMAND transpose-pretty 10 1024)
prk_fortran_test(fortran_pretty_nstream   COMMAND nstream-pretty 10 16777216)
prk_fortran_test(fortran_pretty_dgemm     COMMAND dgemm-pretty 10 400)

if(OpenMP_Fortran_FOUND)
  prk_fortran_test(fortran_openmp_p2p_tasks     COMMAND p2p-tasks-openmp 10 1024 1024)
  prk_fortran_test(fortran_openmp_p2p_innerloop COMMAND p2p-innerloop-openmp 10 1024)
  prk_fortran_test(fortran_openmp_stencil       COMMAND stencil-openmp 10 1000)
  prk_fortran_test(fortran_openmp_transpose_1   COMMAND transpose-openmp 10 1024 1)
  prk_fortran_test(fortran_openmp_transpose_32  COMMAND transpose-openmp 10 1024 32)
  prk_fortran_test(fortran_openmp_nstream       COMMAND nstream-openmp 10 16777216)
  prk_fortran_test(fortran_openmp_dgemm_untiled COMMAND dgemm-openmp 10 400 400)
  prk_fortran_test(fortran_openmp_dgemm_tiled   COMMAND dgemm-openmp 10 400 32)
  set_tests_properties(
    fortran_openmp_p2p_tasks fortran_openmp_p2p_innerloop fortran_openmp_stencil
    fortran_openmp_transpose_1 fortran_openmp_transpose_32 fortran_openmp_nstream
    fortran_openmp_dgemm_untiled fortran_openmp_dgemm_tiled
    PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2")
endif()

if(PRK_WITH_OFFLOAD)
  prk_fortran_test(fortran_target_stencil      COMMAND stencil-openmp-target 10 1000)
  prk_fortran_test(fortran_target_transpose_1  COMMAND transpose-openmp-target 10 1024 1)
  prk_fortran_test(fortran_target_transpose_32 COMMAND transpose-openmp-target 10 1024 32)
  prk_fortran_test(fortran_target_nstream      COMMAND nstream-openmp-target 10 16777216)
  set_tests_properties(fortran_target_stencil fortran_target_transpose_1
    fortran_target_transpose_32 fortran_target_nstream
    PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2")
endif()

if(PRK_WITH_COARRAY AND OpenCoarrays_CAFRUN)
  prk_fortran_test(fortran_coarray_p2p         COMMAND ${OpenCoarrays_CAFRUN} -n 4 $<TARGET_FILE:fortran_p2p-coarray> 10 1024 1024)
  prk_fortran_test(fortran_coarray_stencil     COMMAND ${OpenCoarrays_CAFRUN} -n 4 $<TARGET_FILE:fortran_stencil-coarray> 10 1000)
  prk_fortran_test(fortran_coarray_transpose_1  COMMAND ${OpenCoarrays_CAFRUN} -n 4 $<TARGET_FILE:fortran_transpose-coarray> 10 1024 1)
  prk_fortran_test(fortran_coarray_transpose_32 COMMAND ${OpenCoarrays_CAFRUN} -n 4 $<TARGET_FILE:fortran_transpose-coarray> 10 1024 32)
endif()

if(MPI_Fortran_FOUND)
  prk_fortran_test(fortran_mpi_nstream         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:fortran_nstream-mpi> 10 16777216)
  prk_fortran_test(fortran_mpi_transpose_a2a   COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:fortran_transpose-a2a-mpi> 10 1024 32)
  prk_fortran_test(fortran_mpi_transpose_p2p   COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:fortran_transpose-p2p-mpi> 10 1024 32)
  prk_fortran_test(fortran_mpi_transpose_get   COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:fortran_transpose-get-mpi> 10 1024 32)
endif()

if(BLAS_FOUND)
  prk_fortran_test(fortran_blas_dgemm COMMAND dgemm-blas 10 400)
endif()

if(GlobalArrays_FOUND AND MPI_Fortran_FOUND AND BLAS_FOUND AND LAPACK_FOUND)
  prk_fortran_test(fortran_ga_nstream   COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:fortran_nstream-ga> 10 16777216)
  prk_fortran_test(fortran_ga_transpose COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:fortran_transpose-ga> 10 1024)
  prk_fortran_test(fortran_ga_dgemm     COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:fortran_dgemm-ga> 10 400)
endif()
