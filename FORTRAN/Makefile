include ../common/FORTRAN.defs
include ../common/PRKVERSION

ifndef RADIUS
  RADIUS=2
endif

STARFLAG   = -DSTAR

FCFLAGS  = $(DEFAULT_OPT_FLAGS)
FCFLAGS += -DPRKVERSION=$(PRKVERSION)
FCFLAGS += -DRADIUS=$(RADIUS) $(STARFLAG)

.PHONY: all clean serial pretty openmp coarray

all: serial pretty openmp coarray target

serial: p2p stencil transpose

pretty: stencil-pretty transpose-pretty

openmp: p2p-openmp stencil-openmp transpose-openmp

coarray: p2p-coarray stencil-coarray transpose-coarray

target: transpose-target

%: %.f90
	$(FC) $(FCFLAGS) $< -o $@

%-pretty: %-pretty.f90
	$(FC) $(FCFLAGS) $< -o $@

%-openmp: %.f90
	$(FC) $(FCFLAGS) $(OPENMPFLAG) $< -o $@

%-coarray: %-coarray.f90
	$(CAFCOMPILER) $(FCFLAGS) $< $(COARRAYFLAG) -o $@

%-target: %-target.f90
	$(FC) $(FCFLAGS) $(OPENMPFLAG) $(OFFLOADFLAG) $< -o $@

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
