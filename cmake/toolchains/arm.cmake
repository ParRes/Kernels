# Mirrors common/make.defs.arm's CC/CXX/FC selection -- Arm's own
# LLVM-based HPC compiler for AArch64. Not installed on this (x86_64)
# machine; provided for completeness.
#
#   CC=${LLVM_PATH}clang${CLANG_VERSION} -std=c11 -pthread
#   FC=${LLVM_PATH}flang -Mpreprocess -Mfreeform -DPGI
#   CXX=${LLVM_PATH}clang++${CLANG_VERSION} -std=c++2a -pthread
#
# LLVM_PATH in the original example:
#   /opt/arm/22.0.1/arm-linux-compiler-22.0.1_Generic-AArch64_Ubuntu-20.04_aarch64-linux/llvm-bin/

set(_prk_arm_bin /opt/arm/22.0.1/arm-linux-compiler-22.0.1_Generic-AArch64_Ubuntu-20.04_aarch64-linux/llvm-bin)
set(CMAKE_C_COMPILER       ${_prk_arm_bin}/clang   CACHE STRING "")
set(CMAKE_CXX_COMPILER     ${_prk_arm_bin}/clang++ CACHE STRING "")
set(CMAKE_Fortran_COMPILER ${_prk_arm_bin}/flang   CACHE STRING "")
