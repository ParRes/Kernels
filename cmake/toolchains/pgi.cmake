# Mirrors common/make.defs.pgi's CC/CXX/FC selection -- the legacy,
# standalone PGI compilers, superseded by NVIDIA HPC SDK (nvc/nvc++/
# nvfortran, see nvhpc.cmake) after NVIDIA's acquisition of PGI. Not
# installed on this machine (only the NVHPC successor); provided for
# completeness.
#
#   CC=pgcc -c11
#   FC=pgfortran -Mpreprocess -Mfreeform
#   CXX=pgc++ --c++17

set(CMAKE_C_COMPILER       pgcc       CACHE STRING "")
set(CMAKE_CXX_COMPILER     pgc++      CACHE STRING "")
set(CMAKE_Fortran_COMPILER pgfortran  CACHE STRING "")
