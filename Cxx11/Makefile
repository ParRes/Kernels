include ../common/Cxx11.defs
include ../common/PRKVERSION

CPPFLAGS  = -DPRKVERSION=$(PRKVERSION)

CXXFLAGS  = $(DEFAULT_OPT_FLAGS) $(CPPFLAGS)

# debugging
ifdef VERBOSE
    CXXFLAGS += -DVERBOSE
endif

ifeq ($(USE_PRK_TBB_PARTITIONER),static)
    PRK_TBB_PARTITIONER=1
endif
ifeq ($(USE_PRK_TBB_PARTITIONER),affinity)
    PRK_TBB_PARTITIONER=2
endif
ifeq ($(USE_PRK_TBB_PARTITIONER),simple)
    PRK_TBB_PARTITIONER=3
endif
ifndef PRK_TBB_PARTITIONER
    PRK_TBB_PARTITIONER=0
endif

# Valid choices are OpenMP, Threads, Serial, Cuda
ifdef USE_PRK_KOKKOS_BACKEND
    KOKKOS_BACKEND_FLAG = -DPRK_KOKKOS_BACKEND=$(USE_PRK_KOKKOS_BACKEND)
endif

#ASMFLAGS = -fsource-asm -fverbose-asm -fasm-blocks -fcode-asm
ASMFLAGS = -fverbose-asm

OMPFLAGS = $(OPENMPFLAG) -DUSE_OPENMP
TARGETFLAGS = $(OFFLOADFLAG)
OPENCLFLAGS = $(OPENCLFLAG)
# We do not yet handle all possible exceptions...
#OPENCLFLAGS += -D__CL_ENABLE_EXCEPTIONS
ORNLACCFLAGS = $(ORNLACCFLAG)
TBBFLAGS = $(TBBFLAG) -DUSE_TBB -DPRK_TBB_PARTITIONER=$(PRK_TBB_PARTITIONER)
CBLASFLAGS = $(CBLASFLAG) $(OPENMPFLAG)
BOOSTFLAGS = $(BOOSTFLAG) -DUSE_BOOST
RANGEFLAGS = $(RANGEFLAG) -DUSE_RANGES
STLFLAGS = $(STLFLAG) $(RANGEFLAGS)
PSTLFLAGS = $(PSTLFLAG) $(RANGEFLAGS) -DUSE_PSTL
RAJAFLAGS = $(RAJAFLAG) -DUSE_RAJA
THRUSTFLAGS = $(THRUSTFLAG) $(RANGEFLAGS) -DUSE_THRUST
KOKKOSFLAGS = $(KOKKOSFLAG) $(KOKKOS_BACKEND_FLAG) $(RANGEFLAGS) -DUSE_KOKKOS
SYCLFLAGS = $(SYCLFLAG) -DUSE_SYCL -DUSE_2D_INDEXING=0 $(RANGEFLAGS)
ORNLACCFLAGS = $(ORNLACCFLAG)

ifdef OCCADIR
  include ${OCCADIR}/scripts/makefile
endif
OCCAFLAGS = -DUSE_OCCA -I${OCCADIR}/include -Wl,-rpath -Wl,${OCCADIR}/lib -L${OCCADIR}/lib -locca

.PHONY: all clean vector valarray openmp target opencl taskloop tbb stl pstl \
	rangefor kokkos raja cuda cublas sycl boost-compute thrust

EXTRA=
ifeq ($(shell uname -s),Darwin)
  ifneq ($(findstring icpc,$(CXX)),icpc)
    EXTRA += target
  endif
else
  EXTRA += target
endif
ifneq ($(findstring pgc++,$(CXX)),pgc++)
  EXTRA += tbb pstl
endif

all: sequential vector valarray openmp taskloop stl rangefor opencl sycl $(EXTRA)

#p2p: p2p-vector p2p-doacross-openmp p2p-hyperplane-openmp p2p-tasks-openmp p2p-openmp-target \
     p2p-innerloop-vector-tbb p2p-vector-raja p2p-vector-tbb p2p-innerloop-opencl p2p-hyperplane-vector-tbb \
     p2p-hyperplane-sycl p2p-hyperplane-vector-ornlacc p2p-tasks-tbb

#stencil: stencil-valarray stencil-vector stencil-vector-async stencil-openmp stencil-openmp-target \
	 stencil-vector-taskloop stencil-vector-stl stencil-vector-pstl stencil-vector-raja \
	 stencil-vector-rangefor stencil-vector-tbb stencil-vector-thread stencil-kokkos stencil-opencl \
	 stencil-cuda

#transpose: transpose-valarray transpose-vector transpose-vector-async transpose-openmp transpose-openmp-target \
	   transpose-vector-taskloop transpose-vector-stl transpose-vector-pstl transpose-vector-raja \
	   transpose-vector-rangefor transpose-vector-tbb transpose-vector-thread transpose-kokkos transpose-opencl

#nstream: nstream-valarray nstream-vector nstream-openmp nstream-openmp-target \
	 nstream-vector-taskloop nstream-vector-stl nstream-vector-pstl nstream-vector-raja \
	 nstream-vector-rangefor nstream-vector-tbb nstream-kokkos nstream-opencl

#dgemm: dgemm-vector dgemm-cblas dgemm-cublas

sequential: p2p stencil transpose nstream dgemm sparse

vector: p2p-vector p2p-hyperplane-vector stencil-vector transpose-vector nstream-vector sparse-vector dgemm-vector \
	transpose-vector-async transpose-vector-thread

valarray: transpose-valarray nstream-valarray

openmp: p2p-hyperplane-openmp p2p-tasks-openmp stencil-openmp transpose-openmp nstream-openmp

target: stencil-openmp-target transpose-openmp-target nstream-openmp-target

taskloop: stencil-vector-taskloop transpose-vector-taskloop nstream-vector-taskloop

opencl: p2p-innerloop-opencl stencil-opencl transpose-opencl nstream-opencl

sycl: nstream-sycl p2p-hyperplane-sycl stencil-sycl transpose-sycl

sycl-usm: nstream-sycl-usm stencil-sycl-usm transpose-sycl-usm

sycl-explicit: nstream-sycl-explicit transpose-sycl-explicit

tbb: p2p-innerloop-vector-tbb p2p-vector-tbb stencil-vector-tbb transpose-vector-tbb nstream-vector-tbb \
     p2p-hyperplane-vector-tbb p2p-tasks-tbb

stl: stencil-vector-stl transpose-vector-stl nstream-vector-stl

pstl: stencil-vector-pstl transpose-vector-pstl nstream-vector-pstl

rangefor: stencil-vector-rangefor transpose-vector-rangefor nstream-vector-rangefor

kokkos: stencil-kokkos transpose-kokkos nstream-kokkos

raja: p2p-vector-raja stencil-vector-raja nstream-vector-raja \
      p2p-raja transpose-raja nstream-raja stencil-raja # transpose-vector-raja

cuda: stencil-cuda transpose-cuda nstream-cuda

thrust: nstream-host-thrust nstream-device-thrust \
        transpose-host-thrust transpose-device-thrust

cuda: transpose-cuda

cublas: transpose-cublas nstream-cublas dgemm-cublas dgemm-multigpu-cublas dgemm-mpi-cublas

cblas: transpose-cblas dgemm-cblas

occa: transpose-occa nstream-occa

ornlacc: p2p-hyperplane-vector-ornlacc

boost-compute: nstream-vector-boost-compute
# busted
#nstream-valarray-boost-compute

p2p-hyperplane-vector: p2p-hyperplane-openmp.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< -o $@

transpose-opencl: transpose-opencl.cc transpose.cl prk_util.h prk_opencl.h
	$(CXX) $(CXXFLAGS) $< $(OPENCLFLAGS) -o $@

nstream-opencl: nstream-opencl.cc nstream.cl prk_util.h prk_opencl.h
	$(CXX) $(CXXFLAGS) $< $(OPENCLFLAGS) -o $@

%-opencl: %-opencl.cc prk_util.h prk_opencl.h
	$(CXX) $(CXXFLAGS) $< $(OPENCLFLAGS) -o $@

%-sycl: %-sycl.cc prk_util.h
	$(SYCLCXX) $(CPPFLAGS) $(SYCLFLAGS) $< -o $@

%-sycl-usm: %-sycl-usm.cc prk_util.h
	$(SYCLCXX) $(CPPFLAGS) $(SYCLFLAGS) $< -o $@

%-sycl-explicit: %-sycl-explicit.cc prk_util.h
	$(SYCLCXX) $(CPPFLAGS) $(SYCLFLAGS) $< -o $@

%-target: %-target.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(OMPFLAGS) $(TARGETFLAGS) -o $@

%-openmp: %-openmp.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(OMPFLAGS) -o $@

%-taskloop: %-taskloop.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(OMPFLAGS) -o $@

%-tbb: %-tbb.cc prk_util.h
	$(info PRK help: Consider setting USE_PRK_TBB_PARTITIONER={static,affinity,simple} when invoking make)
	$(CXX) $(CXXFLAGS) $< $(TBBFLAGS) -o $@

%-stl: %-pstl.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(STLFLAGS) -o $@

%-pstl: %-pstl.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(PSTLFLAGS) -o $@

%-rangefor: %-rangefor.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(RANGEFLAGS) -o $@

%-boost-compute: %-boost-compute.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(BOOSTFLAGS) $(OPENCLFLAGS) -o $@

%-raja: %-raja.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(RAJAFLAGS) -o $@

ifeq ($(PRK_KOKKOS_BACKEND),Cuda)
%-kokkos: %-kokkos.cc prk_util.h
	${KOKKOSDIR}/bin/nvcc_wrapper $(CPPFLAGS) $(CUDAFLAGS) $< $(KOKKOSFLAG) -DUSE_KOKKOS -DPRK_KOKKOS_BACKEND=Cuda -o $@
else
%-kokkos: %-kokkos.cc prk_util.h
	$(info PRK help: Set USE_PRK_KOKKOS_BACKEND={Threads,Serial,Cuda} when invoking make to not use OpenMP)
	$(CXX) $(CXXFLAGS) $< $(KOKKOSFLAGS) -o $@
endif

# for host execution
%-thrust: %-thrust.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(THRUSTFLAGS) -o $@

# for device execution (must compiler as .cu)
%-thrust: %-thrust.cu prk_util.h
	$(NVCC) $(CUDAFLAGS) $(CPPFLAGS) $< $(THRUSTFLAGS) -o $@

%-cuda: %-cuda.cu prk_util.h prk_cuda.h
	$(NVCC) $(CUDAFLAGS) $(CPPFLAGS) $< -o $@

%-mpi-cublas: %-mpi-cublas.cu prk_util.h prk_cuda.h prk_mpi.h
	$(NVCC) $(CUDAFLAGS) $(CPPFLAGS) $< -lcublas $(MPIFLAGS) -o $@

%-cublas: %-cublas.cu prk_util.h prk_cuda.h
	$(NVCC) $(CUDAFLAGS) $(CPPFLAGS) $< -lcublas -o $@

%-cblas: %-cblas.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(CBLASFLAGS) -o $@

%-occa: %-occa.cc prk_util.h
	$(info PRK help: Set OCCA_CXX=$(firstword $(CXX)) to use that compiler for OKL files.)
	$(CXX) $(CXXFLAGS) $< $(OCCAFLAGS) -o $@

%-ornlacc: %-ornlacc.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(ORNLACCFLAGS) -o $@

%: %.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< -o $@

%-raja.s: %-raja.cc prk_util.h
	$(CXX) $(CXXFLAGS) $(ASMFLAGS) -S $< $(RAJAFLAGS) -o $@

%.s: %.cc prk_util.h
	$(CXX) $(CXXFLAGS) $(ASMFLAGS) -S $< -o $@

clean:
	-rm -f *.o
	-rm -f *.s
	-rm -f *.ll # Coriander
	-rm -f *.optrpt
	-rm -f *.dwarf
	-rm -rf *.dSYM # Mac
	-rm -f nstream transpose stencil p2p sparse dgemm
	-rm -f *-vector
	-rm -f *-valarray
	-rm -f *-openmp
	-rm -f *-target
	-rm -f *-taskloop
	-rm -f *-opencl
	-rm -f *-sycl
	-rm -f *-sycl-explicit
	-rm -f *-sycl-usm
	-rm -f *-tbb
	-rm -f *-stl
	-rm -f *-pstl
	-rm -f *-rangefor
	-rm -f *-raja
	-rm -f *-kokkos
	-rm -f *-thrust
	-rm -f *-cuda
	-rm -f *-cublas
	-rm -f *-cblas
	-rm -f *-occa
	-rm -f *-boost-compute
	-rm -f *-ornlacc
	-rm -f transpose-vector-async transpose-vector-thread

cleancl:
	-rm -f star[123456789].cl
	-rm -f grid[123456789].cl
