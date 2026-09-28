# Mirrors common/make.defs.intel's CC/CXX/FC selection -- the classic
# Intel compilers (icc/icpc/ifort), superseded by the LLVM-based oneAPI
# compilers (icx/icpx/ifx, see oneapi.cmake). Not installed on this
# machine (icc/icpc/ifort not found) -- provided for completeness/anyone
# who still has the classic Intel Parallel Studio toolchain.
#
#   CC=icc -std=c11 -pthread
#   FC=ifort -std08 -fpp -qopt-matmul
#   CXX=icpc -std=c++17 -pthread
#
# As with make.defs.intel's own comment, this assumes Intel's environment
# script has been sourced first (compilervars.sh/setvars.sh), which sets up
# more than just PATH (MKLROOT, etc.) -- unlike oneapi.cmake, this file
# doesn't hardcode those paths since the classic toolchain isn't available
# here to verify them against.

set(CMAKE_C_COMPILER       icc   CACHE STRING "")
set(CMAKE_CXX_COMPILER     icpc  CACHE STRING "")
set(CMAKE_Fortran_COMPILER ifort CACHE STRING "")
