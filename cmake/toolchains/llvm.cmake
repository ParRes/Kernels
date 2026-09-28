# Mirrors common/make.defs.llvm's CC/CXX/FC selection.
#
#   CC=${LLVM_PATH}clang${CLANG_VERSION} -std=c11 -pthread
#   FC=${LLVM_PATH}flang-new
#   CXX=${LLVM_PATH}clang++${CLANG_VERSION} -std=c++2a -pthread
#
# make.defs.llvm leaves LLVM_PATH/CLANG_VERSION empty by default (PATH
# search); this machine also has a specific LLVM toolchain (with a working
# flang-new) installed under /opt/llvm/latest, used here since the system
# clang/clang++ don't come with a matching Fortran frontend.
#
# Only the compiler executables are set, not -std=/-pthread -- see gcc.cmake
# for why.

set(CMAKE_C_COMPILER       /opt/llvm/latest/bin/clang     CACHE STRING "")
set(CMAKE_CXX_COMPILER     /opt/llvm/latest/bin/clang++   CACHE STRING "")
set(CMAKE_Fortran_COMPILER /opt/llvm/latest/bin/flang-new CACHE STRING "")
