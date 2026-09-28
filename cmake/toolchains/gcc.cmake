# Mirrors common/make.defs.gcc's CC/CXX/FC selection.
#
#   CC=gcc-14 -std=c11 -pthread
#   FC=gfortran-14 -std=f2018 -cpp -fexternal-blas -fblas-matmul-limit=0
#   CXX=g++-14 -std=c++20 -pthread -fmax-errors=1
#
# Only the compiler executables are set here, not the -std=/-pthread/etc.
# flags: each language's CMakeLists.txt already manages its own C++/Fortran
# standard selection via CMAKE_CXX_STANDARD/real feature probes, which is
# more portable than hardcoding a specific -std flag in the toolchain file
# (and avoids the exact "two competing -std= flags, last one silently
# wins" footgun this project hit earlier with try_compile probes).
#
# Usage: cmake -S Cxx11 -B build -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/gcc.cmake

set(CMAKE_C_COMPILER       gcc-14       CACHE STRING "")
set(CMAKE_CXX_COMPILER     g++-14       CACHE STRING "")
set(CMAKE_Fortran_COMPILER gfortran-14  CACHE STRING "")
