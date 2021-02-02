#
# This file shows the HIP toolchain options for PRKs.
#
# Base compilers and language options
#
VERSION=-10
# C99 is required in some implementations.
CC=gcc${VERSION} -std=gnu11 -pthread
#EXTRA_CLIBS=-lrt
# All of the Fortran code is written for the 2008 standard and requires preprocessing.
FC=gfortran${VERSION} -std=f2008 -cpp
# C++11 may not be required but does no harm here.
CXX=g++${VERSION} -std=gnu++17 -pthread
#
# Compiler flags
#
# -mtune=native is appropriate for most cases.
# -march=native is appropriate if you want portable binaries.
DEFAULT_OPT_FLAGS=-O3 -mtune=native -ffast-math
#DEFAULT_OPT_FLAGS+=-fsanitize=undefined
#DEFAULT_OPT_FLAGS+=-fsanitize=undefined,leak
#DEFAULT_OPT_FLAGS+=-fsanitize=address
#DEFAULT_OPT_FLAGS+=-fsanitize=thread
# See https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html for details.
#
#
# OpenMP flags
#
OPENMPFLAG=-fopenmp
OPENMPSIMDFLAG=-fopenmp-simd
OFFLOADFLAG=-foffload=amdgcn-amdhsa="-march=fiji"
OPENACCFLAG=-fopenacc $(OFFLOADFLAG)
#
# OpenCL flags
#
# Linux
OPENCLDIR=/opt/rocm/opencl/
OPENCLFLAG=-I${OPENCLDIR} -L${OPENCLDIR}/lib -lOpenCL
#
# SYCL flags
#
# hipSYCL
#
SYCLDIR=/opt/hipSYCL
SYCLCXX=${SYCLDIR}/bin/syclcc-clang
SYCLFLAG=-std=c++17 -O3
SYCLFLAG+=-DHIPSYCL
# CPU platform
SYCLFLAG+=--hipsycl-platform=rocm
SYCLFLAG+=--hipsycl-gpu-arch=gfx900
#SYCLFLAG+=-Wl,-rpath=/opt/hipSYCL/llvm/lib
#
#CELERITYDIR=${SYCLDIR}
#CELERITYINC=-I$(CELERITYDIR)/include/celerity -I$(CELERITYDIR)/include/celerity/vendor
#CELERITYLIB=-L$(CELERITYDIR)/lib -lcelerity_runtime
#
# OCCA
#
#OCCADIR=${HOME}/PRK/Cxx11/occa-git
#
# TBB
#
TBBDIR=/usr
TBBFLAG=-I${TBBDIR}/include -L${TBBDIR}/lib -ltbb
#
# Parallel STL, Boost, etc.
#
BOOSTFLAG=-I/usr/include
RANGEFLAG=-DUSE_BOOST_IRANGE ${BOOSTFLAG}
#RANGEFLAG=-DUSE_RANGES_TS -I./range-v3/include
PSTLFLAG=${OPENMPSIMDFLAG} ${TBBFLAG} ${RANGEFLAG} -I./pstl/stdlib -I./pstl/include
KOKKOSDIR=/opt/kokkos/hip
KOKKOSFLAG=-I${KOKKOSDIR}/include -L${KOKKOSDIR}/lib -lkokkos ${OPENMPFLAG} -ldl
RAJADIR=/opt/raja/hip
RAJAFLAG=-I${RAJADIR}/include -L${RAJADIR}/lib -lRAJA ${OPENMPFLAG} ${TBBFLAG}
THRUSTDIR=/opt/nvidia/thrust
THRUSTFLAG=-I${THRUSTDIR} ${RANGEFLAG}
#
# CBLAS for C++ DGEMM
#
BLASFLAG=-L/usr/lib/x86_64-linux-gnu -lblas64 -lpthread
CBLASFLAG=-L/usr/lib/x86_64-linux-gnu -lblas64 -lpthread
#
# CUDA flags
#
NVCC=nvcc
CUDAFLAGS=-g -O3 -std=c++11
CUDAFLAGS+=--gpu-architecture=sm_70
CUDAFLAGS+=-rdc=true
# heavy hammer:
CUDAFLAGS+=-D_X86INTRIN_H_INCLUDED
#
# HIP flags
#
HIPCC=hipcc
HIPFLAGS=-g -O3 -std=c++14
# hipBLAS
HIPFLAGS+=-L/opt/rocm/lib -lrocblas
HIPFLAGS+=-L/opt/rocm/lib -lhipblas
# hipThrust
HIPFLAGS+=-I/opt/rocm/include
HIPFLAGS+=-I/opt/rocm/rocthrust/include
HIPFLAGS+=-I/opt/rocm/rocprim/include
HIPFLAGS+=${RANGEFLAG}
#
# MPI-3
#
# We assume you have Intel MPI and have setup your environment with e.g.
# . /opt/intel/compilers_and_libraries/linux/mpi/intel64/bin/mpivars.sh
# in your .bashrc.
#
# mpiicc wraps icc.  mpicc and mpigcc wrap gcc.
MPIDIR=/usr
MPICC=${MPIDIR}/bin/mpicc
MPICXX=${MPIDIR}/bin/mpicxx
MPIINC=-I${MPIDIR}/include
MPILIB=-L${MPIDIR}/lib -lmpi
#MPILIB=-L/usr/local/opt/libevent/lib -L${MPIDIR}/lib -lmpi
#MPIINC=-I/usr/include/mpich-3.2-x86_64
#MPILIB=-L/usr/lib64/mpich-3.2/lib -lmpi
#
# Global Arrays
#
GADIR=../deps/ga
GAFLAG=-I${GADIR}/include
GAFLAG+=-L${GADIR}/lib -lga
GAFLAG+=-L${GADIR}/../armci-mpi/lib -larmci # ARMCI-MPI
#GAFLAG+=-L${GADIR}/lib -larmci -lcomex     # ARMCI/ComEx
GAFLAG+=${MPIINC} ${MPILIB}
GAFLAG+=-lmpifort -lmpi
GAFLAG+=-fdefault-integer-8 # GA is compiled with 64b integers on 64-bit systems
#
# PETSc
#
PETSCDIR=../deps/petsc
PETSCFLAG=-I${PETSCDIR}/include
PETSCFLAG+=-L${PETSCDIR}/lib -lpetsc
PETSCFLAG+=-Wl,-rpath=${PETSCDIR}/lib
#
# Fortran 2008 coarrays
#
# see https://github.com/ParRes/Kernels/blob/master/FORTRAN/README.md for details
# single-node
COARRAYFLAG=-fcoarray=single -lcaf_single
# multi-node
# COARRAYFLAG=-fcoarray=lib -lcaf_mpi
#
# MEMKIND (used in C1z)
#
MEMKINDDIR=/home/parallels/PRK/deps
MEMKINDFLAGS=-I${MEMKINDDIR}/include -L${MEMKINDDIR}/lib -lmemkind -Wl,-rpath=${MEMKINDDIR}/lib
