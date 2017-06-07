include ../common/FORTRAN.defs
include ../common/PRKVERSION

ifndef RADIUS
  RADIUS=2
endif

STARFLAG   = -DSTAR

FCFLAGS  = $(DEFAULT_OPT_FLAGS)
FCFLAGS += -DPRKVERSION=$(PRKVERSION)
FCFLAGS += -DRADIUS=$(RADIUS) $(STARFLAG)

.PHONY: all clean plain pretty openmp coarray

all: plain pretty openmp coarray

plain: p2p stencil transpose

pretty: stencil-pretty transpose-pretty

openmp: p2p-openmp stencil-openmp transpose-openmp

coarray: p2p-coarray stencil-coarray transpose-coarray

%: %.f90
	$(FC) $(FCFLAGS) $< -o $@

%-pretty: %-pretty.f90
	$(FC) $(FCFLAGS) $< -o $@

%-openmp: %.f90
	$(FC) $(FCFLAGS) $(OPENMPFLAG) $< -o $@

%-coarray: %-coarray.f90
	$(CAFCOMPILER) $(FCFLAGS) $< $(COARRAYFLAG) -o $@

clean:
	-rm -f *.o
	-rm -f *.i90
	-rm -f *.optrpt
	-rm -f *.dwarf
	-rm -rf *.dSYM
	-rm -f p2p         stencil         transpose
	-rm -f p2p-pretty  stencil-pretty  transpose-pretty
	-rm -f p2p-openmp  stencil-openmp  transpose-openmp
	-rm -f p2p-coarray stencil-coarray transpose-coarray
