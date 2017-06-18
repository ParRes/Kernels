include ../common/Cxx11.defs
include ../common/PRKVERSION

ifndef RADIUS
  RADIUS=2
endif

STARFLAG   = -DSTAR
#DOUBLEFLAG = -DDOUBLE

CXXFLAGS  = $(DEFAULT_OPT_FLAGS)
CXXFLAGS += -DPRKVERSION=$(PRKVERSION)
CXXFLAGS += -DRADIUS=$(RADIUS) $(STARFLAG) $(DOUBLEFLAG)
CXXFLAGS += -I.

# debugging
#CXXFLAGS += -DVERBOSE

OMPFLAGS = $(OPENMPFLAG)
TARGETFLAGS = $(OFFLOADFLAG)
OPENCLFLAGS = $(OPENCLFLAG)
ORNLACCFLAGS = $(ORNLACCFLAG)
CILKLAGS = $(CILKFLAG)
TBBFLAGS = $(TBBFLAG) -DUSE_TBB

.PHONY: all clean run vector valarray openmp target opencl cilk

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

openmp: p2p-vector-doacross-openmp p2p-tasks-openmp stencil-vector-openmp transpose-vector-openmp

target: p2p-openmp-target stencil-openmp-target transpose-openmp-target

opencl: stencil-opencl transpose-opencl

cilk: stencil-vector-cilk transpose-vector-cilk

tbb: p2p-vector-tbb stencil-vector-tbb transpose-vector-tbb

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

%: %.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	-rm -f *.o
	-rm -f *.optrpt
	-rm -f *.dwarf
	-rm -rf *.dSYM
	-rm -f *-vector
	-rm -f *-valarray
	-rm -f *-openmp
	-rm -f *-target
	-rm -f *-opencl
	-rm -f *-cilk
	-rm -f *-tbb

cleancl:
	-rm -f star[123456789].cl
	-rm -f grid[123456789].cl
