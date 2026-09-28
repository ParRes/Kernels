# Mirrors common/make.defs.armgcc's CC/CXX/FC selection -- plain GCC
# cross/native-compiling for Arm, as opposed to Arm's own LLVM-based
# compiler (see arm.cmake). Not verified on this (x86_64) machine, but
# unlike arm.cmake this just needs a GCC on PATH (GCC_PATH/GCC_VERSION
# empty by default in the original example), so it would work as-is on
# any machine with a plain `gcc`/`g++`/`gfortran`.
#
#   CC=${GCC_PATH}gcc${GCC_VERSION} -std=c11 -pthread
#   FC=${GCC_PATH}gfortran${GCC_VERSION} -std=f2018 -cpp -fexternal-blas -fblas-matmul-limit=0
#   CXX=${GCC_PATH}g++${GCC_VERSION} -std=gnu++20 -pthread -fmax-errors=1

set(CMAKE_C_COMPILER       gcc       CACHE STRING "")
set(CMAKE_CXX_COMPILER     g++       CACHE STRING "")
set(CMAKE_Fortran_COMPILER gfortran  CACHE STRING "")
