PROGRAM=stencil

include ../../common/FORTRAN.defs
include ../../common/PRKVERSION

ifndef RADIUS
  RADIUS=2
endif

STARFLAG   = -DSTAR

FCFLAGS  = $(DEFAULT_OPT_FLAGS)
FCFLAGS += -DPRKVERSION=$(FPRKVERSION)
FCFLAGS += -DRADIUS=$(RADIUS) $(STARFLAG)

.PHONY: all clean run serial openmp coarray

all: serial openmp coarray

serial: $(PROGRAM)
openmp: $(PROGRAM)-omp
coarray: $(PROGRAM)-coarray

$(PROGRAM): $(PROGRAM).f90
	$(FC) $(FCFLAGS) $< -o $@

$(PROGRAM)-omp: $(PROGRAM).f90
	$(FC) $(FCFLAGS) $(OPENMPFLAG) $< -o $@

$(PROGRAM)-coarray: $(PROGRAM)-coarray.f90
	$(CAFCOMPILER) $(FCFLAGS) $< $(COARRAYFLAG) -o $@

run: $(PROGRAM)
	$(PROGRAM) 10 4000 32

clean:
	-rm -f *.o
	-rm -f *.i90
	-rm -f *.optrpt
	-rm -f *.dwarf
	-rm -rf *.dSYM
	-rm -f $(PROGRAM)
	-rm -f $(PROGRAM)-omp
	-rm -f $(PROGRAM)-coarray

