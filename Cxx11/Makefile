include ../common/Cxx11.defs
include ../common/PRKVERSION

ifndef RADIUS
  RADIUS=2
endif

STARFLAG   = -DSTAR
#DOUBLEFLAG = -DDOUBLE

CPPFLAGS  = -DPRKVERSION=$(PRKVERSION)
CPPFLAGS += -DRADIUS=$(RADIUS) $(STARFLAG) $(DOUBLEFLAG)
CPPFLAGS += -I.

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
KOKKOSFLAGS = $(KOKKOSFLAG) $(BOOSTFLAGS)
RAJAFLAGS = $(RAJAFLAG)

.PHONY: all clean run vector valarray openmp target opencl cilk tbb pstl rangefor kokkos raja cuda

ifeq ($(shell uname -s),Darwin)
ifneq ($(CXX),icpc)
TARGET = target
endif
else
TARGET = target
endif
all: vector valarray openmp opencl $(TARGET)

vector: p2p-vector stencil-vector transpose-vector

valarray: transpose-valarray

openmp: p2p-wavefront-openmp p2p-vector-doacross-openmp p2p-tasks-openmp stencil-vector-openmp transpose-vector-openmp

target: p2p-openmp-target stencil-openmp-target transpose-openmp-target

opencl: stencil-opencl transpose-opencl

cilk: stencil-vector-cilk transpose-vector-cilk

tbb: p2p-vector-tbb stencil-vector-tbb transpose-vector-tbb

pstl: stencil-vector-pstl transpose-vector-pstl

rangefor: stencil-vector-rangefor transpose-vector-rangefor

kokkos: stencil-kokkos transpose-kokkos

#raja: stencil-vector-raja transpose-vector-raja
raja: transpose-vector-raja

cuda: transpose-cuda

transpose-opencl: transpose-opencl.cc transpose.cl prk_util.h prk_opencl.h
	$(CXX) $(CXXFLAGS) $< $(OPENCLFLAGS) -o $@

%-opencl: %-opencl.cc prk_util.h prk_opencl.h
	$(CXX) $(CXXFLAGS) $< $(OPENCLFLAGS) -o $@

%-target: %-target.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(OMPFLAGS) $(TARGETFLAGS) -o $@

%-openmp: %-openmp.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(OMPFLAGS) -o $@

%-cilk: %-cilk.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(CILKFLAGS) -o $@

%-tbb: %-tbb.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(TBBFLAGS) -o $@

%-pstl: %-pstl.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(PSTLFLAGS) -o $@

%-rangefor: %-rangefor.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(BOOSTFLAGS) -o $@

%-kokkos: %-kokkos.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(KOKKOSFLAGS) -o $@

%-raja: %-raja.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< $(RAJAFLAGS) -o $@

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
	-rm -f *-opencl
	-rm -f *-cilk
	-rm -f *-tbb
	-rm -f *-pstl
	-rm -f *-rangefor
	-rm -f *-kokkos
	-rm -f *-raja
	-rm -f *-cuda

cleancl:
	-rm -f star[123456789].cl
	-rm -f grid[123456789].cl
