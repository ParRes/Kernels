# Mirrors common/make.defs.hip's CC/CXX/FC selection -- AMD's ROCm/HIP
# toolchain (Clang-based). No AMD GPU/ROCm install on this machine;
# provided for completeness.
#
#   CC=${ROCM_PATH}/llvm/bin/clang -std=gnu11 -pthread -lm
#   FC=${ROCM_PATH}/llvm/bin/flang -DAOMP
#   CXX=${ROCM_PATH}/llvm/bin/clang++ -std=gnu++17 -pthread
#
# ROCM_PATH defaults to /opt/rocm if unset, matching ROCm's own convention.

if(NOT DEFINED ENV{ROCM_PATH})
  set(_prk_rocm_path /opt/rocm)
else()
  set(_prk_rocm_path $ENV{ROCM_PATH})
endif()
set(CMAKE_C_COMPILER       ${_prk_rocm_path}/llvm/bin/clang   CACHE STRING "")
set(CMAKE_CXX_COMPILER     ${_prk_rocm_path}/llvm/bin/clang++ CACHE STRING "")
set(CMAKE_Fortran_COMPILER ${_prk_rocm_path}/llvm/bin/flang   CACHE STRING "")
