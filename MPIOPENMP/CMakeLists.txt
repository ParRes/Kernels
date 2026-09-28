#
# CMake port of MPIOPENMP/*/Makefile. Same structure as SERIAL/CMakeLists.txt
# (one executable per kernel subdirectory), but each kernel includes
# ../../common/MPIOPENMP.defs instead of SERIAL.defs:
#   CCOMPILER=$(MPICC)
#   COMOBJS=MPI_bail_out.o wtime.o
#   PROG_ENV=-DMPI $(OPENMPFLAG)
# so every executable links both MPI::MPI_C and OpenMP::OpenMP_C, plus
# common/MPI_bail_out.c + common/wtime.c (compiled once as OBJECT
# libraries, reused across all executables), and defines -DMPI. Each
# kernel's own -D tuning flags match its Makefile's own
# `ifndef ... default` values, same pattern as SERIAL/CMakeLists.txt.
#
# Standalone project: `cd MPIOPENMP && cmake -S . -B build` works on its
# own (in-source `cmake .` too). Also add_subdirectory()'d from the repo
# root.
#

cmake_minimum_required(VERSION 3.18)
project(PRK_MPIOPENMP LANGUAGES C)
enable_testing()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(MPI REQUIRED COMPONENTS C)
find_package(OpenMP REQUIRED COMPONENTS C)

set(PRK_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")
set(PRK_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../common")

add_library(prk_mpiopenmp_wtime OBJECT "${PRK_COMMON_DIR}/wtime.c")
target_include_directories(prk_mpiopenmp_wtime PRIVATE ${PRK_INCLUDE_DIR})
target_compile_definitions(prk_mpiopenmp_wtime PRIVATE MPI)
# wtime.c itself #includes "mpi.h" under -DMPI (to implement an MPI-
# synchronized timer), so it needs MPI's include dir too, not just the
# executables that link it.
target_link_libraries(prk_mpiopenmp_wtime PRIVATE MPI::MPI_C)

add_library(prk_mpiopenmp_bail_out OBJECT "${PRK_COMMON_DIR}/MPI_bail_out.c")
target_include_directories(prk_mpiopenmp_bail_out PRIVATE ${PRK_INCLUDE_DIR})
target_compile_definitions(prk_mpiopenmp_bail_out PRIVATE MPI)
target_link_libraries(prk_mpiopenmp_bail_out PRIVATE MPI::MPI_C)

# prk_mpiopenmp_executable(<name> DIR <subdir> SOURCES <files...> [DEFS ...])
# The CMake target is named mpiopenmp_<name>, not <name>: target names
# must be unique across the whole combined root project (every PRK
# language directory has its own p2p/stencil/etc.), so OUTPUT_NAME puts
# the actual produced binary back to the bare <name>.
function(prk_mpiopenmp_executable name)
  set(oneValueArgs DIR)
  set(multiValueArgs SOURCES DEFS)
  cmake_parse_arguments(PME "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(_srcs "")
  foreach(_src IN LISTS PME_SOURCES)
    list(APPEND _srcs "${PME_DIR}/${_src}")
  endforeach()
  set(_target mpiopenmp_${name})
  add_executable(${_target} ${_srcs}
    $<TARGET_OBJECTS:prk_mpiopenmp_wtime> $<TARGET_OBJECTS:prk_mpiopenmp_bail_out>)
  set_target_properties(${_target} PROPERTIES OUTPUT_NAME ${name})
  target_include_directories(${_target} PRIVATE ${PRK_INCLUDE_DIR} "${CMAKE_CURRENT_LIST_DIR}/${PME_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/${PME_DIR}")
  target_compile_definitions(${_target} PRIVATE MPI ${PME_DEFS})
  target_link_libraries(${_target} PRIVATE Threads::Threads MPI::MPI_C OpenMP::OpenMP_C m)
endfunction()

function(prk_mpiopenmp_test test_name)
  set(multiValueArgs COMMAND)
  cmake_parse_arguments(PMT "" "" "${multiValueArgs}" ${ARGN})
  add_test(NAME ${test_name} COMMAND ${PMT_COMMAND})
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 30)
endfunction()

# --- Synch_p2p --------------------------------------------------------
# SYNCHRONOUS=0, RESTRICT_KEYWORD=0, MAXTHREADS=512, VERBOSE=0
prk_mpiopenmp_executable(p2p DIR Synch_p2p SOURCES p2p.c
  DEFS SYNCHRONOUS=0 RESTRICT_KEYWORD=0 MAXTHREADS=512 VERBOSE=0)

# --- Stencil ------------------------------------------------------------
# DOUBLE=1, STAR=1, RESTRICT_KEYWORD=0, RADIUS=2, LOOPGEN=0, VERBOSE=0.
# stencil.c #includes the generated loop_body_star.incl/loop_body_compact.incl
# unconditionally, same as SERIAL/Stencil -- generate via common/Stencil/loop_gen.
set(_stencil_gen_dir "${CMAKE_CURRENT_BINARY_DIR}/Stencil")
file(MAKE_DIRECTORY ${_stencil_gen_dir})
add_custom_command(
  OUTPUT "${_stencil_gen_dir}/loop_body_star.incl"
  COMMAND "${PRK_COMMON_DIR}/Stencil/loop_gen" 2 1
  WORKING_DIRECTORY ${_stencil_gen_dir}
  COMMENT "Generating Stencil/loop_body_star.incl (RADIUS=2)")
add_custom_command(
  OUTPUT "${_stencil_gen_dir}/loop_body_compact.incl"
  COMMAND "${PRK_COMMON_DIR}/Stencil/loop_gen" 2 0
  WORKING_DIRECTORY ${_stencil_gen_dir}
  COMMENT "Generating Stencil/loop_body_compact.incl (RADIUS=2)")
prk_mpiopenmp_executable(stencil DIR Stencil SOURCES stencil.c
  DEFS DOUBLE=1 STAR=1 RESTRICT_KEYWORD=0 RADIUS=2 LOOPGEN=0 VERBOSE=0)
target_sources(mpiopenmp_stencil PRIVATE "${_stencil_gen_dir}/loop_body_star.incl" "${_stencil_gen_dir}/loop_body_compact.incl")
set_source_files_properties("${_stencil_gen_dir}/loop_body_star.incl" "${_stencil_gen_dir}/loop_body_compact.incl" PROPERTIES HEADER_FILE_ONLY TRUE)

# --- Transpose ------------------------------------------------------------
# COLLAPSE=0, RESTRICT_KEYWORD=0, SYNCHRONOUS=0, VERBOSE=0
prk_mpiopenmp_executable(transpose DIR Transpose SOURCES transpose.c
  DEFS COLLAPSE=0 RESTRICT_KEYWORD=0 SYNCHRONOUS=0 VERBOSE=0)

# --- Nstream ---------------------------------------------------------------
# MAXLENGTH=2000000, STATIC_ALLOCATION=0, RESTRICT_KEYWORD=0, VERBOSE=0
prk_mpiopenmp_executable(nstream DIR Nstream SOURCES nstream.c
  DEFS MAXLENGTH=2000000 STATIC_ALLOCATION=0 RESTRICT_KEYWORD=0 VERBOSE=0)

# --- CTest: reproduces ci/build-run-prk.sh's allmpi) "MPI+OpenMP" section -

# Each kernel's own argv order is <#threads> <#iterations> <...size args>
# (confirmed against MPIOPENMP/Stencil/stencil.c's own parsing) -- the
# leading "2" below is that thread-count CLI arg, matching OMP_NUM_THREADS.
if(MPIEXEC_EXECUTABLE)
  prk_mpiopenmp_test(mpiopenmp_p2p       COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:mpiopenmp_p2p> 2 10 1024 1024)
  prk_mpiopenmp_test(mpiopenmp_stencil   COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:mpiopenmp_stencil> 2 10 1000)
  prk_mpiopenmp_test(mpiopenmp_transpose COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:mpiopenmp_transpose> 2 10 1024 32)
  prk_mpiopenmp_test(mpiopenmp_nstream   COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:mpiopenmp_nstream> 2 10 16777216 32)
  set_tests_properties(mpiopenmp_p2p mpiopenmp_stencil mpiopenmp_transpose mpiopenmp_nstream
    PROPERTIES ENVIRONMENT OMP_NUM_THREADS=2)
else()
  message(STATUS "MPIOPENMP tests: skipped, no mpiexec found")
endif()
