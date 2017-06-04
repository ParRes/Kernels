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

OMPFLAGS = $(OPENMPFLAG)
TARGETFLAGS = $(OFFLOADFLAG)

.PHONY: all clean run vector valarray openmp target

all: vector valarray openmp target

vector: p2p-vector stencil-vector transpose-vector

valarray: transpose-valarray

openmp: stencil-vector-openmp transpose-vector-openmp

target: stencil-openmp-target transpose-openmp-target

%-target: %-target.cc
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(TARGETFLAGS) $< -o $@

%-openmp: %-openmp.cc
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

%: %.cc
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	-rm -f *.o
	-rm -f *.optrpt
	-rm -f *.dwarf
	-rm -rf *.dSYM
	-rm -f p2p
	-rm -f *-vector
	-rm -f *-valarray
	-rm -f *-openmp
	-rm -f *-target

