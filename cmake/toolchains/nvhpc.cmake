# Mirrors common/make.defs.nvhpc's CC/CXX/FC selection.
#
#   CC=${NVHPC_CBIN}nvc -c11
#   FC=${NVHPC_CBIN}nvfortran -DNVHPC
#   CXX=${NVHPC_CBIN}nvc++ -std=gnu++20
#
# /opt/nvidia/hpc_sdk/Linux_x86_64/2026 is this machine's "latest" symlink.
# Only the compiler executables are set, not -std=/-DNVHPC -- see
# gcc.cmake for why (each language's CMakeLists.txt manages its own
# standard selection and NVHPC-specific gating via
# CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC" already).

set(_prk_nvhpc_bin /opt/nvidia/hpc_sdk/Linux_x86_64/2026/compilers/bin)
set(CMAKE_C_COMPILER       ${_prk_nvhpc_bin}/nvc        CACHE STRING "")
set(CMAKE_CXX_COMPILER     ${_prk_nvhpc_bin}/nvc++      CACHE STRING "")
set(CMAKE_Fortran_COMPILER ${_prk_nvhpc_bin}/nvfortran  CACHE STRING "")
