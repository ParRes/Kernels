# Mirrors common/make.defs.ibmbg's CC/CXX/FC selection -- IBM's XL
# compilers for Blue Gene/Q. Not available outside that system; provided
# for completeness.
#
#   CC=bgxlc_r -qlanglvl=stdc99
#   FC=bgxlf_r
#   CXX=bgxlcxx_r

set(CMAKE_C_COMPILER       bgxlc_r   CACHE STRING "")
set(CMAKE_CXX_COMPILER     bgxlcxx_r CACHE STRING "")
set(CMAKE_Fortran_COMPILER bgxlf_r   CACHE STRING "")
