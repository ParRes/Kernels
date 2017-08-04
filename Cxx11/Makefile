include ../common/Cxx11.defs
include ../common/PRKVERSION

CPPFLAGS  = -DPRKVERSION=$(PRKVERSION)

CXXFLAGS  = $(DEFAULT_OPT_FLAGS) $(CPPFLAGS)

# debugging
#CXXFLAGS += -DVERBOSE

OMPFLAGS = $(OPENMPFLAG)
TARGETFLAGS = $(OFFLOADFLAG)
OPENCLFLAGS = $(OPENCLFLAG)
# We do not yet handle all possible exceptions...
#OPENCLFLAGS += -D__CL_ENABLE_EXCEPTIONS
ORNLACCFLAGS = $(ORNLACCFLAG)
CILKFLAGS = $(CILKFLAG)
TBBFLAGS = $(TBBFLAG) -DUSE_TBB
BOOSTFLAGS = $(BOOSTFLAG)
PSTLFLAGS = $(PSTLFLAG) $(BOOSTFLAGS)
RAJAFLAGS = $(RAJAFLAG)
KOKKOSFLAGS = $(KOKKOSFLAG) $(BOOSTFLAGS)

.PHONY: all clean run vector valarray openmp target opencl cilk taskloop tbb pstl rangefor kokkos raja cuda

ifeq ($(shell uname -s),Darwin)
ifneq ($(findstring icpc,$(CXX)),icpc)
TARGET = target
endif
else
TARGET = target
endif

all: vector valarray openmp opencl $(TARGET) cilk taskloop tbb pstl rangefor raja kokkos

vector: p2p-vector p2p-innerloop-vector stencil-vector transpose-vector transpose-vector-async transpose-vector-thread

valarray: transpose-valarray

openmp: p2p-innerloop-vector-openmp p2p-tasks-openmp stencil-vector-openmp transpose-vector-openmp

target: p2p-openmp-target stencil-openmp-target transpose-openmp-target

taskloop: stencil-vector-taskloop transpose-vector-taskloop

opencl: p2p-innerloop-opencl stencil-opencl transpose-opencl

cilk: stencil-vector-cilk transpose-vector-cilk

tbb: p2p-innerloop-vector-tbb p2p-vector-tbb stencil-vector-tbb transpose-vector-tbb

pstl: stencil-vector-pstl transpose-vector-pstl

rangefor: stencil-vector-rangefor transpose-vector-rangefor

kokkos: stencil-kokkos transpose-kokkos

raja: p2p-vector-raja stencil-vector-raja transpose-vector-raja

cuda: transpose-cuda

p2p-innerloop-vector: p2p-innerloop-vector-openmp.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< -o $@

transpose-opencl: transpose-opencl.cc transpose.cl prk_util.h prk_opencl.h
	$(CXX) $(CXXFLAGS) $< $(OPENCLFLAGS) -o $@

%-opencl: %-opencl.cc prk_util.h prk_opencl.h
	$(CXX) $(CXXFLAGS) $< $(OPENCLFLAGS) -o $@

%-target: %-target.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(OMPFLAGS) $(TARGETFLAGS) -o $@

%-openmp: %-openmp.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(OMPFLAGS) -o $@

%-taskloop: %-taskloop.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(OMPFLAGS) -o $@

%-cilk: %-cilk.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(CILKFLAGS) -o $@

%-tbb: %-tbb.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(TBBFLAGS) -o $@

%-pstl: %-pstl.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(PSTLFLAGS) -o $@

%-rangefor: %-rangefor.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(BOOSTFLAGS) -o $@

%-raja: %-raja.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(RAJAFLAGS) -o $@

%-kokkos: %-kokkos.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(KOKKOSFLAGS) -o $@

%-cuda: %-cuda.cu prk_util.h prk_cuda.h
	$(NVCC) $(CUDAFLAGS) $(CPPFLAGS) $< -o $@

%: %.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	-rm -f *.o
	-rm -f *.ll # Coriander
	-rm -f *.optrpt
	-rm -f *.dwarf
	-rm -rf *.dSYM # Mac
	-rm -f *-vector
	-rm -f *-valarray
	-rm -f *-openmp
	-rm -f *-target
	-rm -f *-taskloop
	-rm -f *-opencl
	-rm -f *-cilk
	-rm -f *-tbb
	-rm -f *-pstl
	-rm -f *-rangefor
	-rm -f *-raja
	-rm -f *-kokkos
	-rm -f *-cuda
	-rm -f transpose-vector-async transpose-vector-thread

cleancl:
	-rm -f star[123456789].cl
	-rm -f grid[123456789].cl
