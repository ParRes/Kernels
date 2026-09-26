#
# CMake port of MPI1/*/Makefile. Same structure as SERIAL/CMakeLists.txt
# (each kernel in its own subdirectory, its own Makefile), except each
# kernel here includes ../../common/MPI.defs instead of SERIAL.defs:
#   CCOMPILER=$(MPICC)
#   COMOBJS=MPI_bail_out.o wtime.o
#   PROG_ENV=-DMPI
# This port mirrors that: one executable per kernel subdirectory, common/
# MPI_bail_out.c + common/wtime.c compiled once and linked into all of
# them via find_package(MPI COMPONENTS C) -> MPI::MPI_C, and each kernel's
# own -D tuning flags matching its Makefile's own `ifndef ... default`
# values (no #ifndef fallbacks in the .c sources -- they rely on always
# receiving these defines from the command line, same as SERIAL).
#
# Standalone project: `cd MPI1 && cmake -S . -B build` works on its own
# (in-source `cmake .` too). Also add_subdirectory()'d from the repo root.
#

cmake_minimum_required(VERSION 3.18)
project(PRK_MPI1 LANGUAGES C)
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

add_library(prk_mpi1_wtime OBJECT "${PRK_COMMON_DIR}/wtime.c")
target_include_directories(prk_mpi1_wtime PRIVATE ${PRK_INCLUDE_DIR})

add_library(prk_mpi1_bail_out OBJECT "${PRK_COMMON_DIR}/MPI_bail_out.c")
target_include_directories(prk_mpi1_bail_out PRIVATE ${PRK_INCLUDE_DIR})
target_link_libraries(prk_mpi1_bail_out PRIVATE MPI::MPI_C)

add_library(prk_mpi1_random_draw OBJECT "${PRK_COMMON_DIR}/random_draw.c")
target_include_directories(prk_mpi1_random_draw PRIVATE ${PRK_INCLUDE_DIR})

# prk_mpi1_executable(<name> DIR <subdir> SOURCES <files...> [DEFS ...] [LIBS ...])
# The CMake target is named mpi1_<name>, not <name>: target names must be
# unique across the whole combined root project (every PRK language
# directory has its own p2p/stencil/etc.), so OUTPUT_NAME puts the actual
# produced binary back to the bare <name>.
function(prk_mpi1_executable name)
  set(oneValueArgs DIR)
  set(multiValueArgs SOURCES DEFS LIBS)
  cmake_parse_arguments(PME "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(_srcs "")
  foreach(_src IN LISTS PME_SOURCES)
    list(APPEND _srcs "${PME_DIR}/${_src}")
  endforeach()
  set(_target mpi1_${name})
  add_executable(${_target} ${_srcs} $<TARGET_OBJECTS:prk_mpi1_wtime> $<TARGET_OBJECTS:prk_mpi1_bail_out>)
  set_target_properties(${_target} PROPERTIES OUTPUT_NAME ${name})
  target_include_directories(${_target} PRIVATE ${PRK_INCLUDE_DIR} "${CMAKE_CURRENT_LIST_DIR}/${PME_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/${PME_DIR}")
  target_compile_definitions(${_target} PRIVATE ${PME_DEFS})
  target_link_libraries(${_target} PRIVATE Threads::Threads MPI::MPI_C ${PME_LIBS})
endfunction()

function(prk_mpi1_test test_name)
  set(multiValueArgs COMMAND)
  cmake_parse_arguments(PMT "" "" "${multiValueArgs}" ${ARGN})
  add_test(NAME ${test_name} COMMAND ${PMT_COMMAND})
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 30)
endfunction()

# --- Synch_p2p --------------------------------------------------------
prk_mpi1_executable(p2p DIR Synch_p2p SOURCES p2p.c
  DEFS VERBOSE=0 RESTRICT_KEYWORD=0)

# --- Stencil ------------------------------------------------------------
# stencil.c #includes "loop_body_star.incl"/"loop_body_compact.incl"
# unconditionally, same as SERIAL's -- generated at build time.
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
prk_mpi1_executable(stencil DIR Stencil SOURCES stencil.c
  DEFS DOUBLE=1 STAR=1 RESTRICT_KEYWORD=0 RADIUS=2 LOOPGEN=0 VERBOSE=0
  LIBS m)
target_sources(mpi1_stencil PRIVATE "${_stencil_gen_dir}/loop_body_star.incl" "${_stencil_gen_dir}/loop_body_compact.incl")
set_source_files_properties("${_stencil_gen_dir}/loop_body_star.incl" "${_stencil_gen_dir}/loop_body_compact.incl" PROPERTIES HEADER_FILE_ONLY TRUE)

# --- Transpose ------------------------------------------------------------
# SYNCHRONOUS=0
prk_mpi1_executable(transpose DIR Transpose SOURCES transpose.c
  DEFS VERBOSE=0 RESTRICT_KEYWORD=0 SYNCHRONOUS=0)

# --- Reduce --------------------------------------------------------------
prk_mpi1_executable(reduce DIR Reduce SOURCES reduce.c
  DEFS VERBOSE=0 RESTRICT_KEYWORD=0)

# --- Random ---------------------------------------------------------------
# LOOKAHEAD=1024, LONG_IS_64BITS=0
prk_mpi1_executable(random DIR Random SOURCES random.c
  DEFS VERBOSE=0 RESTRICT_KEYWORD=0 LOOKAHEAD=1024 LONG_IS_64BITS=0)

# --- Nstream ---------------------------------------------------------------
prk_mpi1_executable(nstream DIR Nstream SOURCES nstream.c
  DEFS VERBOSE=0 RESTRICT_KEYWORD=0)

# --- Sparse -----------------------------------------------------------
# SCRAMBLE=1, TESTDENSE=0
prk_mpi1_executable(sparse DIR Sparse SOURCES sparse.c
  DEFS VERBOSE=0 RESTRICT_KEYWORD=0 SCRAMBLE=1 TESTDENSE=0)

# --- DGEMM ------------------------------------------------------------
# BOFFSET=12 (no MKL/DEFAULTBLOCK here -- MPI1's dgemm.c doesn't have them)
prk_mpi1_executable(dgemm DIR DGEMM SOURCES dgemm.c
  DEFS VERBOSE=0 BOFFSET=12
  LIBS m)

# --- Synch_global -----------------------------------------------------
prk_mpi1_executable(global DIR Synch_global SOURCES global.c
  DEFS VERBOSE=0)

# --- PIC-static -------------------------------------------------------
# Also links common/random_draw.c (COMOBJS += random_draw.o).
add_executable(mpi1_pic PIC-static/pic.c $<TARGET_OBJECTS:prk_mpi1_wtime> $<TARGET_OBJECTS:prk_mpi1_bail_out> $<TARGET_OBJECTS:prk_mpi1_random_draw>)
set_target_properties(mpi1_pic PROPERTIES OUTPUT_NAME pic)
target_include_directories(mpi1_pic PRIVATE ${PRK_INCLUDE_DIR} "${CMAKE_CURRENT_LIST_DIR}/PIC-static")
target_compile_definitions(mpi1_pic PRIVATE VERBOSE=0 RESTRICT_KEYWORD=0)
target_link_libraries(mpi1_pic PRIVATE Threads::Threads MPI::MPI_C m)

# --- AMR --------------------------------------------------------------
# AMR also compiles timestep.c (COMOBJS += timestep.o) and needs all four
# loop_gen/loop_gen_amr variants.
set(_amr_gen_dir "${CMAKE_CURRENT_BINARY_DIR}/AMR")
file(MAKE_DIRECTORY ${_amr_gen_dir})
add_custom_command(
  OUTPUT "${_amr_gen_dir}/loop_body_star.incl"
  COMMAND "${PRK_COMMON_DIR}/Stencil/loop_gen" 2 1
  WORKING_DIRECTORY ${_amr_gen_dir}
  COMMENT "Generating AMR/loop_body_star.incl (RADIUS=2)")
add_custom_command(
  OUTPUT "${_amr_gen_dir}/loop_body_compact.incl"
  COMMAND "${PRK_COMMON_DIR}/Stencil/loop_gen" 2 0
  WORKING_DIRECTORY ${_amr_gen_dir}
  COMMENT "Generating AMR/loop_body_compact.incl (RADIUS=2)")
add_custom_command(
  OUTPUT "${_amr_gen_dir}/loop_body_star_amr.incl"
  COMMAND "${PRK_COMMON_DIR}/Stencil/loop_gen_amr" 2 1
  WORKING_DIRECTORY ${_amr_gen_dir}
  COMMENT "Generating AMR/loop_body_star_amr.incl (RADIUS=2)")
add_custom_command(
  OUTPUT "${_amr_gen_dir}/loop_body_compact_amr.incl"
  COMMAND "${PRK_COMMON_DIR}/Stencil/loop_gen_amr" 2 0
  WORKING_DIRECTORY ${_amr_gen_dir}
  COMMENT "Generating AMR/loop_body_compact_amr.incl (RADIUS=2)")
prk_mpi1_executable(amr DIR AMR SOURCES amr.c timestep.c
  DEFS DOUBLE=1 STAR=1 RESTRICT_KEYWORD=0 RADIUS=2 LOOPGEN=0 VERBOSE=0
  LIBS m)
target_sources(mpi1_amr PRIVATE
  "${_amr_gen_dir}/loop_body_star.incl" "${_amr_gen_dir}/loop_body_compact.incl"
  "${_amr_gen_dir}/loop_body_star_amr.incl" "${_amr_gen_dir}/loop_body_compact_amr.incl")
set_source_files_properties(
  "${_amr_gen_dir}/loop_body_star.incl" "${_amr_gen_dir}/loop_body_compact.incl"
  "${_amr_gen_dir}/loop_body_star_amr.incl" "${_amr_gen_dir}/loop_body_compact_amr.incl"
  PROPERTIES HEADER_FILE_ONLY TRUE)

# --- Branch (not part of ci/build-run-prk.sh's allmpi) coverage, but a
# normal kernel with its own subdirectory like the rest) -------------------
set(_branch_gen_dir "${CMAKE_CURRENT_BINARY_DIR}/Branch")
file(MAKE_DIRECTORY ${_branch_gen_dir})
add_custom_command(
  OUTPUT "${_branch_gen_dir}/func.c"
  COMMAND "${CMAKE_CURRENT_LIST_DIR}/Branch/func_gen" 10 40
  WORKING_DIRECTORY ${_branch_gen_dir}
  COMMENT "Generating Branch/func.c (MATRIX_RANK=10, NUMBER_OF_FUNCTIONS=40)")
add_executable(mpi1_branch Branch/branch.c "${_branch_gen_dir}/func.c" $<TARGET_OBJECTS:prk_mpi1_wtime> $<TARGET_OBJECTS:prk_mpi1_bail_out>)
set_target_properties(mpi1_branch PROPERTIES OUTPUT_NAME branch)
target_include_directories(mpi1_branch PRIVATE ${PRK_INCLUDE_DIR} "${CMAKE_CURRENT_LIST_DIR}/Branch")
target_compile_definitions(mpi1_branch PRIVATE VERBOSE=0 RESTRICT_KEYWORD=0)
target_link_libraries(mpi1_branch PRIVATE Threads::Threads MPI::MPI_C)

# --- CTest: reproduces ci/build-run-prk.sh's allmpi) MPI-1 run commands --

prk_mpi1_test(mpi1_p2p       COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpi1_p2p> 10 1024 1024)
prk_mpi1_test(mpi1_stencil   COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpi1_stencil> 10 1000)
prk_mpi1_test(mpi1_transpose COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpi1_transpose> 10 1024 32)
prk_mpi1_test(mpi1_reduce    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpi1_reduce> 10 16777216)
prk_mpi1_test(mpi1_nstream   COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpi1_nstream> 10 16777216 32)
prk_mpi1_test(mpi1_sparse    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpi1_sparse> 10 10 5)
prk_mpi1_test(mpi1_dgemm     COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpi1_dgemm> 10 1024 32 1)
prk_mpi1_test(mpi1_random    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpi1_random> 32 20)
prk_mpi1_test(mpi1_global    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpi1_global> 10 16384)
prk_mpi1_test(mpi1_pic_geometric  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpi1_pic> 10 1000 1000000 1 2 GEOMETRIC 0.99)
prk_mpi1_test(mpi1_pic_sinusoidal COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpi1_pic> 10 1000 1000000 0 1 SINUSOIDAL)
prk_mpi1_test(mpi1_pic_linear     COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpi1_pic> 10 1000 1000000 1 0 LINEAR 1.0 3.0)
prk_mpi1_test(mpi1_pic_patch      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpi1_pic> 10 1000 1000000 1 0 PATCH 0 200 100 200)
prk_mpi1_test(mpi1_amr_fine_grain COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpi1_amr> 10 1000 100 2 2 1 5 FINE_GRAIN 2)
prk_mpi1_test(mpi1_amr_high_water COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpi1_amr> 10 1000 100 2 2 1 5 HIGH_WATER)
prk_mpi1_test(mpi1_amr_no_talk    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpi1_amr> 10 1000 100 2 2 1 5 NO_TALK)
prk_mpi1_test(mpi1_branch    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpi1_branch> 10 1024 vector_go)
