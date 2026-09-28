# Mirrors common/make.defs.ibmp9nv's CC/CXX/FC selection -- IBM's XL
# compilers for Power9 (+NVIDIA GPU) systems. Not available outside that
# system; provided for completeness.
#
#   CC=xlc_r -qlanglvl=stdc99
#   FC=xlf2008_r
#   CXX=xlc++_r -qlanglvl=extended1y

set(CMAKE_C_COMPILER       xlc_r     CACHE STRING "")
set(CMAKE_CXX_COMPILER     xlc++_r   CACHE STRING "")
set(CMAKE_Fortran_COMPILER xlf2008_r CACHE STRING "")
