# Mirrors common/make.defs.cuda's CC/CXX/FC selection.
#
#   CC=${NVHPC_CBIN}nvc -c11 -march=zen4
#   FC=${NVHPC_CBIN}nvfortran -DNVHPC -march=zen4
#   CXX=${NVHPC_CBIN}nvc++ -std=gnu++20 -march=zen4
#
# Same NVHPC compilers as nvhpc.cmake -- make.defs.cuda's own comment says
# it covers "both NVHPC and GCC" (GCC is the commented-out alternative in
# the file). The -march=zen4 in the original example is specific to the
# AMD Zen4 machine that example was written for; deliberately not copied
# here since this toolchain file is meant to be portable across whatever
# machine actually uses it, and hardcoding another vendor's microarch
# target would be wrong on non-Zen4 hardware (this machine is Intel
# Sapphire Rapids). If you need it, add
# -DCMAKE_CXX_FLAGS_INIT=-march=<your target> at configure time instead.

set(_prk_nvhpc_bin /opt/nvidia/hpc_sdk/Linux_x86_64/2026/compilers/bin)
set(CMAKE_C_COMPILER       ${_prk_nvhpc_bin}/nvc        CACHE STRING "")
set(CMAKE_CXX_COMPILER     ${_prk_nvhpc_bin}/nvc++      CACHE STRING "")
set(CMAKE_Fortran_COMPILER ${_prk_nvhpc_bin}/nvfortran  CACHE STRING "")
