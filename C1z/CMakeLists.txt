#
# CMake port of C1z/Makefile. Target groups and their toolchain/compiler
# gating mirror the Makefile's per-group flag requirements (OPENMPFLAG,
# OFFLOADFLAG, OPENACCFLAG, CILKFLAG, ISPCFLAG, MEMKINDFLAGS, PETSCFLAG) as
# closely as CMake's find_package()/find_program()/try_compile() facilities
# allow.
#
# Standalone project: `cd C1z && cmake -S . -B build` works on its own
# (in-source `cmake .` too). Also add_subdirectory()'d from the repo root
# for a combined build.
#

cmake_minimum_required(VERSION 3.18)
project(PRK_C1z LANGUAGES C)
enable_testing()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake/modules")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# common/PRKVERSION: a single line `PRKVERSION="2020"` (quotes included in
# the shell variable's value, and thus in the -DPRKVERSION=$(PRKVERSION)
# CPPFLAGS Make uses -- reproduce that literally instead of hardcoding it).
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/../common/PRKVERSION" _prkversion_line REGEX "^PRKVERSION=")
string(REGEX REPLACE "^PRKVERSION=(.*)$" "\\1" PRK_VERSION_DEFINE "${_prkversion_line}")
add_compile_definitions(PRKVERSION=${PRK_VERSION_DEFINE})

# --- dependency detection -----------------------------------------------

find_package(Threads REQUIRED)
find_package(OpenMP COMPONENTS C)
find_package(MPI COMPONENTS C)
find_package(OpenACC COMPONENTS C)
find_program(PRK_ISPC_EXECUTABLE ispc)

# memkind and PETSc have no CMake config upstream; probe like make.defs.*
# does (MEMKINDDIR/PETSCDIR env hints), no autobuild -- both are niche
# enough that "not found, skipped" is the right default rather than
# fetching+building a whole PETSc install.
find_path(PRK_MEMKIND_INCLUDE_DIR memkind.h HINTS ENV MEMKINDDIR PATH_SUFFIXES include)
find_library(PRK_MEMKIND_LIBRARY memkind HINTS ENV MEMKINDDIR PATH_SUFFIXES lib)
if(PRK_MEMKIND_INCLUDE_DIR AND PRK_MEMKIND_LIBRARY)
  set(PRK_MEMKIND_FOUND TRUE)
endif()

find_path(PRK_PETSC_INCLUDE_DIR petsc.h HINTS ENV PETSCDIR PATH_SUFFIXES include)
find_library(PRK_PETSC_LIBRARY petsc HINTS ENV PETSCDIR PATH_SUFFIXES lib)
if(PRK_PETSC_INCLUDE_DIR AND PRK_PETSC_LIBRARY AND MPI_C_FOUND)
  set(PRK_PETSC_FOUND TRUE)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/AutobuildDeps.cmake")
prk_autobuild_petsc()

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/PRKCFeatures.cmake")
prk_check_openmp_offload_c()
prk_check_openacc_links_c()
prk_check_cilk_c()
prk_check_avx_c()

set(PRK_WITH_TARGET FALSE)
if(PRK_C_OPENMP_OFFLOAD_FLAG)
  set(PRK_WITH_TARGET TRUE)
endif()
set(PRK_WITH_OPENACC FALSE)
if(PRK_C_OPENACC_WORKS)
  set(PRK_WITH_OPENACC TRUE)
endif()
set(PRK_WITH_CILK FALSE)
if(PRK_C_CILK_FLAG)
  set(PRK_WITH_CILK TRUE)
endif()
set(PRK_WITH_AVX FALSE)
if(PRK_C_AVX_FLAG)
  set(PRK_WITH_AVX TRUE)
endif()

message(STATUS "C1z: OpenMP ${OpenMP_C_FOUND}, MPI ${MPI_C_FOUND}, OpenACC ${PRK_WITH_OPENACC}, "
               "target offload ${PRK_WITH_TARGET}, ispc ${PRK_ISPC_EXECUTABLE}, memkind ${PRK_MEMKIND_FOUND}, "
               "PETSc ${PRK_PETSC_FOUND}, cilk ${PRK_WITH_CILK}, avx ${PRK_WITH_AVX} (compiler: ${CMAKE_C_COMPILER_ID})")

# --- helper -----------------------------------------------------------

# The CMake target is named c1z_<name>, not <name>: target names must be
# unique across the whole combined root project (every PRK language
# directory has its own p2p/stencil/etc.), so OUTPUT_NAME puts the actual
# produced binary back to the bare <name>.
function(prk_c_executable name)
  set(multiValueArgs SOURCES LIBS FLAGS DEFS)
  cmake_parse_arguments(PCE "" "" "${multiValueArgs}" ${ARGN})
  set(_target c1z_${name})
  add_executable(${_target} ${PCE_SOURCES})
  set_target_properties(${_target} PROPERTIES OUTPUT_NAME ${name})
  # Explicit libm: some link combinations (e.g. MPI::MPI_C pulling in
  # -Wl,--as-needed indirectly) otherwise fail with "libm.so.6: error
  # adding symbols: DSO missing from command line" despite every kernel
  # using fabs()/sqrt().
  target_link_libraries(${_target} PRIVATE Threads::Threads m ${PCE_LIBS})
  if(PCE_FLAGS)
    target_compile_options(${_target} PRIVATE ${PCE_FLAGS})
    target_link_options(${_target} PRIVATE ${PCE_FLAGS})
  endif()
  if(PCE_DEFS)
    target_compile_definitions(${_target} PRIVATE ${PCE_DEFS})
  endif()
endfunction()

# --- serial (Makefile "serial:" group; several of these compile the
# *-openmp.c/-2d-openmp.c source with no OpenMP flag, exactly as the
# Makefile's own explicit p2p-hyperplane/nstream/stencil/transpose/%-2d
# rules do) ---

prk_c_executable(p2p               SOURCES p2p.c)
prk_c_executable(p2p-2d            SOURCES p2p-2d.c)
prk_c_executable(p2p-tasks-2d      SOURCES p2p-tasks-2d-openmp.c)
prk_c_executable(p2p-hyperplane    SOURCES p2p-hyperplane-openmp.c)
prk_c_executable(p2p-hyperplane-2d SOURCES p2p-hyperplane-2d-openmp.c)
prk_c_executable(nstream           SOURCES nstream-openmp.c)
prk_c_executable(stencil           SOURCES stencil-openmp.c)
prk_c_executable(stencil-2d        SOURCES stencil-2d-openmp.c)
prk_c_executable(transpose         SOURCES transpose-openmp.c)
prk_c_executable(transpose-2d      SOURCES transpose-2d-openmp.c)
add_custom_target(c1z_serial DEPENDS
  c1z_p2p c1z_p2p-2d c1z_p2p-tasks-2d c1z_p2p-hyperplane c1z_p2p-hyperplane-2d
  c1z_nstream c1z_stencil c1z_stencil-2d c1z_transpose c1z_transpose-2d)

# --- thread ---------------------------------------------------------------

prk_c_executable(transpose-thread SOURCES transpose-thread.c)
add_custom_target(c1z_thread DEPENDS c1z_transpose-thread)

# --- openmp / tasks -------------------------------------------------------

if(OpenMP_C_FOUND)
  prk_c_executable(nstream-openmp           SOURCES nstream-openmp.c           LIBS OpenMP::OpenMP_C)
  prk_c_executable(p2p-hyperplane-openmp    SOURCES p2p-hyperplane-openmp.c    LIBS OpenMP::OpenMP_C)
  prk_c_executable(p2p-hyperplane-2d-openmp SOURCES p2p-hyperplane-2d-openmp.c LIBS OpenMP::OpenMP_C)
  prk_c_executable(stencil-openmp           SOURCES stencil-openmp.c           LIBS OpenMP::OpenMP_C)
  prk_c_executable(stencil-2d-openmp        SOURCES stencil-2d-openmp.c        LIBS OpenMP::OpenMP_C)
  prk_c_executable(transpose-openmp         SOURCES transpose-openmp.c         LIBS OpenMP::OpenMP_C)
  prk_c_executable(transpose-2d-openmp      SOURCES transpose-2d-openmp.c      LIBS OpenMP::OpenMP_C)
  add_custom_target(c1z_openmp DEPENDS
    c1z_nstream-openmp c1z_p2p-hyperplane-openmp c1z_p2p-hyperplane-2d-openmp
    c1z_stencil-openmp c1z_stencil-2d-openmp c1z_transpose-openmp c1z_transpose-2d-openmp)

  prk_c_executable(p2p-tasks-openmp    SOURCES p2p-tasks-openmp.c    LIBS OpenMP::OpenMP_C)
  prk_c_executable(p2p-tasks-2d-openmp SOURCES p2p-tasks-2d-openmp.c LIBS OpenMP::OpenMP_C)
  add_custom_target(c1z_tasks DEPENDS c1z_p2p-tasks-openmp c1z_p2p-tasks-2d-openmp)

  prk_c_executable(nstream-taskloop  SOURCES nstream-taskloop.c  LIBS OpenMP::OpenMP_C)
  prk_c_executable(stencil-taskloop  SOURCES stencil-taskloop.c  LIBS OpenMP::OpenMP_C)
  prk_c_executable(transpose-taskloop SOURCES transpose-taskloop.c LIBS OpenMP::OpenMP_C)
  add_custom_target(c1z_taskloop DEPENDS c1z_nstream-taskloop c1z_stencil-taskloop c1z_transpose-taskloop)

  prk_c_executable(nstream-memkind-openmp SOURCES nstream-memkind.c LIBS OpenMP::OpenMP_C)
  if(PRK_MEMKIND_FOUND)
    target_include_directories(c1z_nstream-memkind-openmp PRIVATE ${PRK_MEMKIND_INCLUDE_DIR})
    target_link_libraries(c1z_nstream-memkind-openmp PRIVATE ${PRK_MEMKIND_LIBRARY})
  else()
    set_target_properties(c1z_nstream-memkind-openmp PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()

  prk_c_executable(nstream-mmap-openmp SOURCES nstream-mmap.c LIBS OpenMP::OpenMP_C)

  if(PRK_WITH_TARGET)
    prk_c_executable(nstream-target        SOURCES nstream-target.c        FLAGS ${PRK_C_OPENMP_OFFLOAD_FLAG})
    prk_c_executable(stencil-target        SOURCES stencil-target.c        FLAGS ${PRK_C_OPENMP_OFFLOAD_FLAG})
    prk_c_executable(transpose-target      SOURCES transpose-target.c      FLAGS ${PRK_C_OPENMP_OFFLOAD_FLAG})
    prk_c_executable(nstream-alloc-target  SOURCES nstream-alloc-target.c  FLAGS ${PRK_C_OPENMP_OFFLOAD_FLAG})
    prk_c_executable(nstream-usm-target    SOURCES nstream-usm-target.c    FLAGS ${PRK_C_OPENMP_OFFLOAD_FLAG})
    prk_c_executable(nstream-memcpy-target SOURCES nstream-memcpy-target.c FLAGS ${PRK_C_OPENMP_OFFLOAD_FLAG})
    prk_c_executable(nstream-ua-target     SOURCES nstream-ua-target.c     FLAGS ${PRK_C_OPENMP_OFFLOAD_FLAG})
    add_custom_target(c1z_target DEPENDS
      c1z_nstream-target c1z_stencil-target c1z_transpose-target c1z_nstream-alloc-target
      c1z_nstream-usm-target c1z_nstream-memcpy-target c1z_nstream-ua-target)
  else()
    message(STATUS "C1z target (OpenMP offload): skipped, no working target-offload flags found")
  endif()
else()
  message(STATUS "C1z openmp/tasks/taskloop/target: skipped, no OpenMP C support found")
endif()

# --- mpi / petsc ------------------------------------------------------

if(MPI_C_FOUND)
  prk_c_executable(nstream-mpi       SOURCES nstream-mpi.c       LIBS MPI::MPI_C)
  prk_c_executable(transpose-a2a-mpi SOURCES transpose-a2a-mpi.c LIBS MPI::MPI_C)
  prk_c_executable(transpose-p2p-mpi SOURCES transpose-p2p-mpi.c LIBS MPI::MPI_C)
  add_custom_target(c1z_mpi DEPENDS c1z_nstream-mpi c1z_transpose-a2a-mpi c1z_transpose-p2p-mpi)

  if(PRK_PETSC_FOUND)
    prk_c_executable(nstream-petsc   SOURCES nstream-petsc.c   LIBS MPI::MPI_C)
    prk_c_executable(transpose-petsc SOURCES transpose-petsc.c LIBS MPI::MPI_C)
    target_include_directories(c1z_nstream-petsc   PRIVATE ${PRK_PETSC_INCLUDE_DIR})
    target_include_directories(c1z_transpose-petsc PRIVATE ${PRK_PETSC_INCLUDE_DIR})
    target_link_libraries(c1z_nstream-petsc   PRIVATE ${PRK_PETSC_LIBRARY})
    target_link_libraries(c1z_transpose-petsc PRIVATE ${PRK_PETSC_LIBRARY})
    if(PRK_PETSC_AUTOBUILT)
      add_dependencies(c1z_nstream-petsc   ${PRK_PETSC_EXTERNAL_TARGET})
      add_dependencies(c1z_transpose-petsc ${PRK_PETSC_EXTERNAL_TARGET})
      # libpetsc.so lives outside any system loader path when autobuilt;
      # rpath it so the binaries run standalone, not just under CTest with
      # LD_LIBRARY_PATH set.
      get_filename_component(_prk_petsc_libdir "${PRK_PETSC_LIBRARY}" DIRECTORY)
      set_target_properties(c1z_nstream-petsc c1z_transpose-petsc PROPERTIES
        BUILD_RPATH "${_prk_petsc_libdir}"
        INSTALL_RPATH "${_prk_petsc_libdir}")
    endif()
    add_custom_target(c1z_petsc DEPENDS c1z_nstream-petsc c1z_transpose-petsc)
  else()
    message(STATUS "C1z petsc: skipped, PETSc not found")
  endif()
else()
  message(STATUS "C1z mpi/petsc: skipped, no MPI C support found")
endif()

# --- memkind (serial) / mmap (serial) ------------------------------------

if(PRK_MEMKIND_FOUND)
  prk_c_executable(nstream-memkind SOURCES nstream-memkind.c)
  target_include_directories(c1z_nstream-memkind PRIVATE ${PRK_MEMKIND_INCLUDE_DIR})
  target_link_libraries(c1z_nstream-memkind PRIVATE ${PRK_MEMKIND_LIBRARY})
  add_custom_target(c1z_memkind DEPENDS c1z_nstream-memkind c1z_nstream-memkind-openmp)
else()
  message(STATUS "C1z memkind: skipped, memkind not found")
endif()

prk_c_executable(nstream-mmap SOURCES nstream-mmap.c)
add_custom_target(c1z_mmap DEPENDS c1z_nstream-mmap c1z_nstream-mmap-openmp)

# --- openacc ----------------------------------------------------------

if(PRK_WITH_OPENACC)
  prk_c_executable(nstream-openacc      SOURCES nstream-openacc.c      LIBS OpenACC::OpenACC_C)
  prk_c_executable(stencil-openacc      SOURCES stencil-openacc.c      LIBS OpenACC::OpenACC_C)
  prk_c_executable(transpose-openacc    SOURCES transpose-openacc.c    LIBS OpenACC::OpenACC_C)
  prk_c_executable(transpose-2d-openacc SOURCES transpose-2d-openacc.c LIBS OpenACC::OpenACC_C)
  add_custom_target(c1z_openacc DEPENDS c1z_nstream-openacc c1z_stencil-openacc c1z_transpose-openacc c1z_transpose-2d-openacc)
else()
  message(STATUS "C1z openacc: skipped, no working OpenACC C support found")
endif()

# --- cilk (Cilk Plus, dropped from modern gcc/clang) ----------------------

if(PRK_WITH_CILK)
  prk_c_executable(stencil-cilk   SOURCES stencil-cilk.c   FLAGS ${PRK_C_CILK_FLAG})
  prk_c_executable(transpose-cilk SOURCES transpose-cilk.c FLAGS ${PRK_C_CILK_FLAG})
  add_custom_target(c1z_cilk DEPENDS c1z_stencil-cilk c1z_transpose-cilk)
else()
  message(STATUS "C1z cilk: skipped, compiler has no working Cilk Plus support")
endif()

# --- vector (SSE/AVX intrinsics; not part of Makefile's "all") -----------

if(PRK_WITH_AVX)
  prk_c_executable(p2p-sse SOURCES p2p-sse.c FLAGS -msse4.2)
  prk_c_executable(p2p-avx SOURCES p2p-avx.c FLAGS -mavx2)
  if(OpenMP_C_FOUND)
    prk_c_executable(p2p-avx-tasks-openmp SOURCES p2p-avx-tasks-openmp.c LIBS OpenMP::OpenMP_C FLAGS -mavx2)
  endif()
  add_custom_target(c1z_vector DEPENDS c1z_p2p-sse c1z_p2p-avx)
else()
  message(STATUS "C1z vector (sse/avx): skipped, no AVX2 support found")
endif()

# --- ispc (Intel SPMD Program Compiler) -----------------------------------

if(PRK_ISPC_EXECUTABLE)
  set(_ispc_obj "${CMAKE_CURRENT_BINARY_DIR}/transpose.ispc${CMAKE_C_OUTPUT_EXTENSION}")
  add_custom_command(
    OUTPUT "${_ispc_obj}"
    COMMAND "${PRK_ISPC_EXECUTABLE}" -O3 --target=host --opt=fast-math
            "${CMAKE_CURRENT_LIST_DIR}/transpose.ispc" --pic -o "${_ispc_obj}"
    DEPENDS "${CMAKE_CURRENT_LIST_DIR}/transpose.ispc"
    COMMENT "Compiling transpose.ispc with ispc")
  add_executable(c1z_transpose-ispc transpose-ispc.c "${_ispc_obj}")
  set_target_properties(c1z_transpose-ispc PROPERTIES OUTPUT_NAME transpose-ispc)
  target_link_libraries(c1z_transpose-ispc PRIVATE Threads::Threads)
  if(OpenMP_C_FOUND)
    target_link_libraries(c1z_transpose-ispc PRIVATE OpenMP::OpenMP_C)
  endif()
  add_custom_target(c1z_ispc DEPENDS c1z_transpose-ispc)
else()
  message(STATUS "C1z ispc: skipped, ispc compiler not found")
endif()

# --- CTest: reproduces ci/build-run-prk.sh's allc1z) run commands --------

function(prk_c_test test_name)
  set(multiValueArgs COMMAND)
  cmake_parse_arguments(PCT "" "" "${multiValueArgs}" ${ARGN})
  add_test(NAME ${test_name} COMMAND ${PCT_COMMAND})
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 30)
endfunction()

# serial
prk_c_test(c1z_serial_nstream            COMMAND nstream 10 16777216 32)
prk_c_test(c1z_serial_p2p_1              COMMAND p2p 10 1024 1024)
prk_c_test(c1z_serial_p2p_2              COMMAND p2p 10 1024 1024 100 100)
prk_c_test(c1z_serial_p2p_hyperplane_1   COMMAND p2p-hyperplane 10 1024)
prk_c_test(c1z_serial_p2p_hyperplane_2   COMMAND p2p-hyperplane 10 1024 32)
prk_c_test(c1z_serial_stencil            COMMAND stencil 10 1000)
prk_c_test(c1z_serial_transpose          COMMAND transpose 10 1024 32)
foreach(s star grid)
  foreach(r 1 2 3 4 5)
    prk_c_test(c1z_serial_stencil_${s}_${r} COMMAND stencil 10 200 ${s} ${r})
  endforeach()
endforeach()

# 2D VLA
prk_c_test(c1z_2d_p2p                    COMMAND p2p-2d 10 1024 1024)
prk_c_test(c1z_2d_p2p_2                  COMMAND p2p-2d 10 1024 1024 100 100)
prk_c_test(c1z_2d_p2p_tasks              COMMAND p2p-tasks-2d 10 1024 1024)
prk_c_test(c1z_2d_p2p_tasks_2            COMMAND p2p-tasks-2d 10 1024 1024 100 100)
prk_c_test(c1z_2d_p2p_hyperplane         COMMAND p2p-hyperplane-2d 10 1024)
prk_c_test(c1z_2d_p2p_hyperplane_2       COMMAND p2p-hyperplane-2d 10 1024 32)
prk_c_test(c1z_2d_stencil                COMMAND stencil-2d 10 1000)
prk_c_test(c1z_2d_transpose              COMMAND transpose-2d 10 1024 32)

# thread
prk_c_test(c1z_thread_transpose COMMAND transpose-thread 10 1024 512)

if(OpenMP_C_FOUND)
  prk_c_test(c1z_openmp_nstream          COMMAND nstream-openmp 10 16777216 32)
  prk_c_test(c1z_openmp_p2p_tasks        COMMAND p2p-tasks-openmp 10 1024 1024 100 100)
  prk_c_test(c1z_openmp_p2p_hyperplane_1 COMMAND p2p-hyperplane-openmp 10 1024)
  prk_c_test(c1z_openmp_p2p_hyperplane_2 COMMAND p2p-hyperplane-openmp 10 1024 32)
  prk_c_test(c1z_openmp_stencil          COMMAND stencil-openmp 10 1000)
  prk_c_test(c1z_openmp_transpose        COMMAND transpose-openmp 10 1024 32)
  foreach(_t c1z_openmp_nstream c1z_openmp_p2p_tasks c1z_openmp_p2p_hyperplane_1
             c1z_openmp_p2p_hyperplane_2 c1z_openmp_stencil c1z_openmp_transpose)
    set_tests_properties(${_t} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2")
  endforeach()

  prk_c_test(c1z_openmp2d_p2p_tasks        COMMAND p2p-tasks-2d-openmp 10 1024 1024)
  prk_c_test(c1z_openmp2d_p2p_tasks_2      COMMAND p2p-tasks-2d-openmp 10 1024 1024 100 100)
  prk_c_test(c1z_openmp2d_p2p_hyperplane_1 COMMAND p2p-hyperplane-2d-openmp 10 1024)
  prk_c_test(c1z_openmp2d_p2p_hyperplane_2 COMMAND p2p-hyperplane-2d-openmp 10 1024 32)
  prk_c_test(c1z_openmp2d_stencil          COMMAND stencil-2d-openmp 10 1000)
  prk_c_test(c1z_openmp2d_transpose        COMMAND transpose-2d-openmp 10 1024 32)
  foreach(_t c1z_openmp2d_p2p_tasks c1z_openmp2d_p2p_tasks_2 c1z_openmp2d_p2p_hyperplane_1
             c1z_openmp2d_p2p_hyperplane_2 c1z_openmp2d_stencil c1z_openmp2d_transpose)
    set_tests_properties(${_t} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2")
  endforeach()

  prk_c_test(c1z_taskloop_nstream   COMMAND nstream-taskloop 10 16777216 32)
  prk_c_test(c1z_taskloop_stencil   COMMAND stencil-taskloop 10 1000)
  prk_c_test(c1z_taskloop_transpose COMMAND transpose-taskloop 10 1024 32)
endif()

if(MPI_C_FOUND)
  prk_c_test(c1z_mpi_nstream       COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:c1z_nstream-mpi> 10 16777216)
  prk_c_test(c1z_mpi_transpose_a2a COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:c1z_transpose-a2a-mpi> 10 1024 32)
  prk_c_test(c1z_mpi_transpose_p2p COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:c1z_transpose-p2p-mpi> 10 1024 32)
endif()

prk_c_test(c1z_mmap_nstream COMMAND nstream-mmap 10 16777216 32)
if(OpenMP_C_FOUND)
  prk_c_test(c1z_mmap_nstream_openmp COMMAND nstream-mmap-openmp 10 16777216 32)
  set_tests_properties(c1z_mmap_nstream_openmp PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2")
endif()

if(PRK_MEMKIND_FOUND)
  prk_c_test(c1z_memkind_nstream COMMAND nstream-memkind 10 16777216 32)
endif()

if(PRK_PETSC_FOUND)
  # PETSc kernels parse PETSc-style -i/-n options, not positional argv like
  # every other C1z kernel (see PetscOptionsGetInt() in nstream-petsc.c/
  # transpose-petsc.c).
  prk_c_test(c1z_petsc_nstream   COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:c1z_nstream-petsc> -i 10 -n 16777216)
  prk_c_test(c1z_petsc_transpose COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:c1z_transpose-petsc> -i 10 -n 1024)
endif()
