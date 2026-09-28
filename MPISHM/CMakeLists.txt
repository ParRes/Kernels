#
# CMake port of MPISHM/*/Makefile. Structured like SERIAL (each kernel in
# its own subdirectory), but include ../../common/MPI.defs instead of
# SERIAL.defs: CCOMPILER=$(MPICC), COMOBJS=MPI_bail_out.o wtime.o,
# PROG_ENV=-DMPI. These kernels use MPI-3 shared-memory windows
# (MPI_Win_allocate_shared etc.) via ordinary MPI calls -- no extra CMake
# dependency beyond find_package(MPI).
#
# Standalone project: `cd MPISHM && cmake -S . -B build` works on its own
# (in-source `cmake .` too). Also add_subdirectory()'d from the repo root.
#

cmake_minimum_required(VERSION 3.18)
project(PRK_MPISHM LANGUAGES C)
enable_testing()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(MPI REQUIRED COMPONENTS C)

set(PRK_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")
set(PRK_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../common")

add_library(prk_mpishm_wtime OBJECT "${PRK_COMMON_DIR}/wtime.c")
target_include_directories(prk_mpishm_wtime PRIVATE ${PRK_INCLUDE_DIR})
target_compile_definitions(prk_mpishm_wtime PRIVATE MPI)
target_link_libraries(prk_mpishm_wtime PRIVATE MPI::MPI_C)

add_library(prk_mpishm_bail_out OBJECT "${PRK_COMMON_DIR}/MPI_bail_out.c")
target_include_directories(prk_mpishm_bail_out PRIVATE ${PRK_INCLUDE_DIR})
target_compile_definitions(prk_mpishm_bail_out PRIVATE MPI)
target_link_libraries(prk_mpishm_bail_out PRIVATE MPI::MPI_C)

# prk_mpishm_executable(<name> DIR <subdir> SOURCES <files...> [DEFS ...] [LIBS ...])
function(prk_mpishm_executable name)
  set(oneValueArgs DIR)
  set(multiValueArgs SOURCES DEFS LIBS)
  cmake_parse_arguments(PME "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(_srcs "")
  foreach(_src IN LISTS PME_SOURCES)
    list(APPEND _srcs "${PME_DIR}/${_src}")
  endforeach()
  set(_target mpishm_${name})
  add_executable(${_target} ${_srcs} $<TARGET_OBJECTS:prk_mpishm_wtime> $<TARGET_OBJECTS:prk_mpishm_bail_out>)
  set_target_properties(${_target} PROPERTIES OUTPUT_NAME ${name})
  target_include_directories(${_target} PRIVATE ${PRK_INCLUDE_DIR} "${CMAKE_CURRENT_LIST_DIR}/${PME_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/${PME_DIR}")
  target_compile_definitions(${_target} PRIVATE MPI ${PME_DEFS})
  target_link_libraries(${_target} PRIVATE Threads::Threads MPI::MPI_C ${PME_LIBS})
endfunction()

function(prk_mpishm_test test_name)
  set(multiValueArgs COMMAND)
  cmake_parse_arguments(PMT "" "" "${multiValueArgs}" ${ARGN})
  add_test(NAME ${test_name} COMMAND ${PMT_COMMAND})
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 30)
endfunction()

# --- Synch_p2p --------------------------------------------------------
# RESTRICT_KEYWORD=0, VERBOSE=0. No group_size arg (unlike Stencil/
# Transpose below) -- p2p.c's argv parsing takes just iterations + 2 dims.
prk_mpishm_executable(p2p DIR Synch_p2p SOURCES p2p.c
  DEFS RESTRICT_KEYWORD=0 VERBOSE=0)

# --- Stencil ------------------------------------------------------------
# DOUBLE=1, STAR=1, LOCAL_BARRIER_SYNCH=0, RESTRICT_KEYWORD=0, RADIUS=2,
# LOOPGEN=0, VERBOSE=0. Same generated-.incl-file pattern as SERIAL's
# Stencil/AMR targets (common/Stencil/loop_gen).
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
prk_mpishm_executable(stencil DIR Stencil SOURCES stencil.c
  DEFS DOUBLE=1 STAR=1 LOCAL_BARRIER_SYNCH=0 RESTRICT_KEYWORD=0 RADIUS=2 LOOPGEN=0 VERBOSE=0
  LIBS m)
target_sources(mpishm_stencil PRIVATE "${_stencil_gen_dir}/loop_body_star.incl" "${_stencil_gen_dir}/loop_body_compact.incl")
set_source_files_properties("${_stencil_gen_dir}/loop_body_star.incl" "${_stencil_gen_dir}/loop_body_compact.incl" PROPERTIES HEADER_FILE_ONLY TRUE)

# --- Transpose ------------------------------------------------------------
# RESTRICT_KEYWORD=0, SYNCHRONOUS=0, VERBOSE=0.
prk_mpishm_executable(transpose DIR Transpose SOURCES transpose.c
  DEFS RESTRICT_KEYWORD=0 SYNCHRONOUS=0 VERBOSE=0)

# --- CTest: reproduces ci/build-run-prk.sh's allmpi)'s "MPI+MPI" (MPISHM)
# section. PRK_MPI_PROCS=4; Stencil/Transpose additionally take a
# "ranks per shared-memory coherence domain" arg = PRK_MPI_PROCS/2 = 2,
# inserted before the iteration count (confirmed against stencil.c/
# transpose.c's own argv parsing, which requires it -- argc must be
# exactly 4/4-5 respectively). Synch_p2p/p2p takes no such arg.

prk_mpishm_test(mpishm_p2p       COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpishm_p2p> 10 1024 1024)
prk_mpishm_test(mpishm_stencil   COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpishm_stencil> 2 10 1000)
prk_mpishm_test(mpishm_transpose COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpishm_transpose> 2 10 1024 32)
