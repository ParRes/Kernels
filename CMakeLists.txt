cmake_minimum_required(VERSION 3.18)

# In-source builds (`cmake .`) are allowed: the legacy hand-written Makefile
# this would otherwise collide with has been moved to Makefile.legacy.

# Thin convenience wrapper only: every language under here is its own
# independent CMake project (own cmake_minimum_required()/project()/
# enable_testing(), own dependency detection) and can be configured and
# built standalone, e.g. `cd FORTRAN && cmake -S . -B build`. This root
# project just add_subdirectory()s the enabled ones so `cmake -S . -B build`
# from the repo root builds/tests everything in one combined build tree.
# CMake fully supports a nested project() under add_subdirectory().
project(PRK NONE)

enable_testing()

option(PRK_ENABLE_FORTRAN "Build FORTRAN/ kernels" ON)
option(PRK_ENABLE_C1Z "Build C1z/ kernels" ON)
option(PRK_ENABLE_CXX11 "Build Cxx11/ kernels" ON)
option(PRK_ENABLE_SERIAL "Build SERIAL/ kernels" ON)
option(PRK_ENABLE_MPI1 "Build MPI1/ kernels" ON)
option(PRK_ENABLE_OPENMP "Build OPENMP/ kernels" ON)
option(PRK_ENABLE_MPIOPENMP "Build MPIOPENMP/ kernels" ON)
option(PRK_ENABLE_MPIRMA "Build MPIRMA/ kernels" ON)
option(PRK_ENABLE_MPISHM "Build MPISHM/ kernels" ON)
option(PRK_ENABLE_SHMEM "Build SHMEM/ kernels" ON)

if(PRK_ENABLE_FORTRAN)
  add_subdirectory(FORTRAN)
endif()

if(PRK_ENABLE_C1Z)
  add_subdirectory(C1z)
endif()

if(PRK_ENABLE_CXX11)
  add_subdirectory(Cxx11)
endif()

if(PRK_ENABLE_SERIAL)
  add_subdirectory(SERIAL)
endif()

if(PRK_ENABLE_MPI1)
  add_subdirectory(MPI1)
endif()

if(PRK_ENABLE_OPENMP)
  add_subdirectory(OPENMP)
endif()

if(PRK_ENABLE_MPIOPENMP)
  add_subdirectory(MPIOPENMP)
endif()

if(PRK_ENABLE_MPIRMA)
  add_subdirectory(MPIRMA)
endif()

if(PRK_ENABLE_MPISHM)
  add_subdirectory(MPISHM)
endif()

if(PRK_ENABLE_SHMEM)
  add_subdirectory(SHMEM)
endif()

# Future languages follow the same pattern: add an option
# PRK_ENABLE_<LANG> (default ON) and add_subdirectory(<DIR>), with a
# <DIR>/CMakeLists.txt structured like FORTRAN/CMakeLists.txt -- its own
# cmake_minimum_required()/project()/enable_testing(), CMAKE_MODULE_PATH
# pointed at ../cmake/modules via CMAKE_CURRENT_LIST_DIR (not
# CMAKE_SOURCE_DIR, which only means "repo root" when standalone), and the
# same dependency-detection / target-group / CTest pattern.
