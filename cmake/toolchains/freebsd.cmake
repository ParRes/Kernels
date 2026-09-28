# Mirrors common/make.defs.freebsd's CC/CXX/FC selection. Not applicable
# on this (Linux) machine; provided for completeness.
#
#   CC=${LLVM_PATH}clang -std=c11 -pthread
#   FC=/usr/local/bin/flang -Mpreprocess -Mfreeform -I/usr/local/flang/include -lexecinfo
#   CXX=${LLVM_PATH}clang++ -std=c++14 -pthread
#
# LLVM_PATH in the original example is ${LLVM_ROOT}/bin/, i.e. PATH search
# by default (LLVM_ROOT unset).

set(CMAKE_C_COMPILER       clang               CACHE STRING "")
set(CMAKE_CXX_COMPILER     clang++             CACHE STRING "")
set(CMAKE_Fortran_COMPILER /usr/local/bin/flang CACHE STRING "")
