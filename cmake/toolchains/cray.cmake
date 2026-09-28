# Mirrors common/make.defs.cray's CC/CXX/FC selection -- the Cray
# Programming Environment compiler wrappers (cc/CC/ftn), which pick the
# actual backend compiler (GNU/Intel/AMD/Cray Fortran) based on the
# loaded PrgEnv-* module. Not available outside a Cray system; provided
# for completeness.
#
#   CC=cc -std=c11
#   FC=ftn -e F
#   CXX=CC -std=c++17

set(CMAKE_C_COMPILER       cc  CACHE STRING "")
set(CMAKE_CXX_COMPILER     CC  CACHE STRING "")
set(CMAKE_Fortran_COMPILER ftn CACHE STRING "")
