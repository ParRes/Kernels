# Mirrors common/make.defs.boost's CC/CXX/FC selection -- a mixed
# toolchain (GCC for C/Fortran, Clang for C++) used to exercise the
# Boost-based fallback code paths (USE_BOOST_IRANGE etc.). gcc-9 isn't
# installed on this machine (only gcc/gcc-14) -- provided for completeness.
#
#   CC=gcc-9 -std=c11 -pthread
#   FC=gfortran-9 -std=f2008 -cpp
#   CXX=clang++ -std=gnu++17 -pthread

set(CMAKE_C_COMPILER       gcc-9      CACHE STRING "")
set(CMAKE_CXX_COMPILER     clang++    CACHE STRING "")
set(CMAKE_Fortran_COMPILER gfortran-9 CACHE STRING "")
