#
# CMake port of SHMEM/*/Makefile. Same structure as MPIRMA/CMakeLists.txt
# (see that file for the general pattern this mirrors): each kernel lives
# in its own subdirectory with its own Makefile including
# ../../common/SHMEM.defs -- CCOMPILER=$(SHMEMCC) (an oshcc-style
# compiler wrapper), COMOBJS=wtime.o SHMEM_bail_out.o, PROG_ENV=-DSHMEM.
# There's no CMake config/pkg-config across OpenSHMEM implementations, so
# cmake/modules/FindSHMEM.cmake asks oshcc itself what flags it uses
# (`oshcc -show`), same approach CMake's own FindMPI takes for mpicc.
#
# Standalone project: `cd SHMEM && cmake -S . -B build` works on its own
# (in-source `cmake .` too). Also add_subdirectory()'d from the repo root.
#

cmake_minimum_required(VERSION 3.18)
project(PRK_SHMEM LANGUAGES C)
enable_testing()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake/modules")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(SHMEM)

set(PRK_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")
set(PRK_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../common")

if(NOT SHMEM_FOUND)
  message(STATUS "SHMEM: skipped, no oshcc/oshrun (OpenSHMEM) found")
  return()
endif()

# Actually run a trivial 2-PE job, not just compile one: this matches the
# session's "detected but broken toolchain" precedent (Fortran/C++ OpenMP
# target offload, OpenACC) -- oshcc/oshrun being found doesn't mean the
# OpenSHMEM runtime actually works. On this machine, Open MPI's built-in
# OSHMEM only ships the UCX SPML, and that SPML fails memory registration
# ("too many segments (max = 32)") during shmem_init() itself, before any
# PRK kernel code runs -- a genuine broken-runtime environment issue, not
# a PRK bug, and not something any kernel-level fix could work around.
set(_shmem_probe_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/shmem_probe")
file(MAKE_DIRECTORY "${_shmem_probe_dir}")
file(WRITE "${_shmem_probe_dir}/probe.c" "#include <shmem.h>\nint main(void) { shmem_init(); shmem_barrier_all(); shmem_finalize(); return 0; }\n")
execute_process(
  COMMAND "${SHMEM_C_COMPILER}" "${_shmem_probe_dir}/probe.c" -o "${_shmem_probe_dir}/probe"
  RESULT_VARIABLE _shmem_probe_compile_result
  OUTPUT_QUIET ERROR_QUIET
)
set(PRK_SHMEM_WORKS FALSE)
if(_shmem_probe_compile_result EQUAL 0)
  execute_process(
    COMMAND "${SHMEM_RUN_EXECUTABLE}" -n 2 "${_shmem_probe_dir}/probe"
    RESULT_VARIABLE _shmem_probe_run_result
    OUTPUT_QUIET ERROR_QUIET
    TIMEOUT 30
  )
  if(_shmem_probe_run_result EQUAL 0)
    set(PRK_SHMEM_WORKS TRUE)
  endif()
endif()

if(NOT PRK_SHMEM_WORKS)
  message(STATUS "SHMEM: oshcc/oshrun found but a real 2-PE job failed to run -- broken OpenSHMEM runtime on this machine, skipping (not a PRK bug)")
  return()
endif()

# wtime.c only special-cases MPI (#include "mpi.h" under -DMPI), not
# SHMEM -- no SHMEM::SHMEM link needed here, unlike the MPI-family
# language directories' wtime OBJECT libraries.
add_library(prk_shmem_wtime OBJECT "${PRK_COMMON_DIR}/wtime.c")
target_include_directories(prk_shmem_wtime PRIVATE ${PRK_INCLUDE_DIR})

add_library(prk_shmem_bail_out OBJECT "${PRK_COMMON_DIR}/SHMEM_bail_out.c")
target_include_directories(prk_shmem_bail_out PRIVATE ${PRK_INCLUDE_DIR})
target_compile_definitions(prk_shmem_bail_out PRIVATE SHMEM)
target_link_libraries(prk_shmem_bail_out PRIVATE SHMEM::SHMEM)

# prk_shmem_executable(<name> DIR <subdir> SOURCES <files...> [DEFS ...] [LIBS ...])
# The CMake target is named shmem_<name>, not <name>: target names must be
# unique across the whole combined root project (every PRK language
# directory has its own p2p/stencil/etc.), so OUTPUT_NAME puts the actual
# produced binary back to the bare <name>.
function(prk_shmem_executable name)
  set(oneValueArgs DIR)
  set(multiValueArgs SOURCES DEFS LIBS)
  cmake_parse_arguments(PSE "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(_srcs "")
  foreach(_src IN LISTS PSE_SOURCES)
    list(APPEND _srcs "${PSE_DIR}/${_src}")
  endforeach()
  set(_target shmem_${name})
  add_executable(${_target} ${_srcs} $<TARGET_OBJECTS:prk_shmem_wtime> $<TARGET_OBJECTS:prk_shmem_bail_out>)
  set_target_properties(${_target} PROPERTIES OUTPUT_NAME ${name})
  target_include_directories(${_target} PRIVATE ${PRK_INCLUDE_DIR} "${CMAKE_CURRENT_LIST_DIR}/${PSE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/${PSE_DIR}")
  target_compile_definitions(${_target} PRIVATE SHMEM ${PSE_DEFS})
  target_link_libraries(${_target} PRIVATE Threads::Threads SHMEM::SHMEM ${PSE_LIBS})
endfunction()

function(prk_shmem_test test_name)
  set(multiValueArgs COMMAND)
  cmake_parse_arguments(PST "" "" "${multiValueArgs}" ${ARGN})
  add_test(NAME ${test_name} COMMAND ${PST_COMMAND})
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 30)
endfunction()

# --- Synch_p2p --------------------------------------------------------
# RESTRICT_KEYWORD=0, VERBOSE=0, SYNCHRONOUS=0
prk_shmem_executable(p2p DIR Synch_p2p SOURCES p2p.c
  DEFS RESTRICT_KEYWORD=0 VERBOSE=0 SYNCHRONOUS=0)

# --- Stencil ------------------------------------------------------------
# DOUBLE=1, STAR=1, SPLITFENCE=0, RESTRICT_KEYWORD=0, RADIUS=2, LOOPGEN=0,
# VERBOSE=0. stencil.c #includes "loop_body_star.incl"/
# "loop_body_compact.incl" unconditionally, same generated-.incl-file
# pattern as SERIAL/MPIRMA's Stencil targets (common/Stencil/loop_gen).
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
prk_shmem_executable(stencil DIR Stencil SOURCES stencil.c
  DEFS DOUBLE=1 STAR=1 SPLITFENCE=0 RESTRICT_KEYWORD=0 RADIUS=2 LOOPGEN=0 VERBOSE=0
  LIBS m)
target_sources(shmem_stencil PRIVATE "${_stencil_gen_dir}/loop_body_star.incl" "${_stencil_gen_dir}/loop_body_compact.incl")
set_source_files_properties("${_stencil_gen_dir}/loop_body_star.incl" "${_stencil_gen_dir}/loop_body_compact.incl" PROPERTIES HEADER_FILE_ONLY TRUE)

# --- Transpose ------------------------------------------------------------
# BARRIER_SYNCH=0, VERBOSE=0
prk_shmem_executable(transpose DIR Transpose SOURCES transpose.c
  DEFS BARRIER_SYNCH=0 VERBOSE=0)

# --- CTest: reproduces ci/build-run-prk.sh's allshmem) run commands -----

prk_shmem_test(shmem_p2p       COMMAND ${SHMEM_RUN_EXECUTABLE} -n 4 $<TARGET_FILE:shmem_p2p> 10 1024 1024)
prk_shmem_test(shmem_stencil   COMMAND ${SHMEM_RUN_EXECUTABLE} -n 4 $<TARGET_FILE:shmem_stencil> 10 1000)
prk_shmem_test(shmem_transpose COMMAND ${SHMEM_RUN_EXECUTABLE} -n 4 $<TARGET_FILE:shmem_transpose> 10 1024 32)
