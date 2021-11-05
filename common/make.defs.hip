#
# This file shows the HIP toolchain options for PRKs.
#
ROCM_PATH=/opt/rocm
#
# Base compilers and language options
#
VERSION=
# C99 is required in some implementations.
CC=${ROCM_PATH}/llvm/bin/clang -std=gnu11 -pthread
#EXTRA_CLIBS=-lrt
# All of the Fortran code is written for the 2008 standard and requires preprocessing.
FC=${ROCM_PATH}/llvm/bin/flang -DAOMP #-std=f2008 -cpp
# C++11 may not be required but does no harm here.
CXX=${ROCM_PATH}/llvm/bin/clang++ -std=gnu++17 -pthread
#
# Compiler flags
#
DEFAULT_OPT_FLAGS=-g -O3 -ffast-math -mtune=native
#
#
# OpenMP flags
#
OPENMPFLAG=-fopenmp
OPENMPSIMDFLAG=-fopenmp-simd
#GCC#OFFLOADFLAG=-foffload=amdgcn-amdhsa="-march=fiji"
OFFLOADFLAG=-fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908
OFFLOADFLAG+=-DGPU_SCHEDULE="schedule(static,1)" # makes ~10x diff with AMD Flang 12 on MI-100
OPENACCFLAG=-fopenacc $(OFFLOADFLAG)
#
# OpenCL flags
#
# Linux
OPENCLDIR=${ROCM_PATH}/opencl/
OPENCLFLAG=-I${OPENCLDIR}/include -L${OPENCLDIR}/lib -lOpenCL
#
# SYCL flags
#
# hipSYCL
#
SYCLDIR=${HOME}/AMD/hipSYCL
SYCLCXX=${SYCLDIR}/bin/syclcc-clang
SYCLFLAG=-std=c++17 -O3
SYCLFLAG+=-DHIPSYCL
# CPU platform
SYCLFLAG+=--hipsycl-platform=rocm
SYCLFLAG+=--hipsycl-gpu-arch=gfx908
SYCLFLAG+=-Wl,-rpath=/opt/rocm/llvm/lib
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
BLASFLAG=-L${HOME}/BLIS/zen2-gcc/lib -lblis
CBLASFLAG=-I${HOME}/BLIS/zen2-gcc/include
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
HIPCC=${ROCM_PATH}/hip/bin/hipcc
HIPFLAGS=-g -O3 -std=c++14
# hipBLAS
HIPFLAGS+=-L${ROCM_PATH}/lib -lrocblas
HIPFLAGS+=-L${ROCM_PATH}/lib -lhipblas
# hipThrust
HIPFLAGS+=-I${ROCM_PATH}/include
HIPFLAGS+=-I${ROCM_PATH}/rocthrust/include
HIPFLAGS+=-I${ROCM_PATH}/rocprim/include
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
MEMKINDDIR=
MEMKINDFLAGS=-I${MEMKINDDIR}/include -L${MEMKINDDIR}/lib -lmemkind -Wl,-rpath=${MEMKINDDIR}/lib
