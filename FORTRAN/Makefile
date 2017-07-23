include ../common/make.defs

ifndef CAFC
  CAFC=$(FC)
endif

ifndef RADIUS
  RADIUS=2
endif

STARFLAG   = -DSTAR

FCFLAGS  = $(DEFAULT_OPT_FLAGS)
FCFLAGS += -DRADIUS=$(RADIUS) $(STARFLAG)

.PHONY: all clean serial pretty openmp coarray target ornlacc

ifeq ($(shell uname -s),Darwin)
ifneq ($(findstring ifort,$(FC)),ifort)
COARRAY = coarray
TARGET = target
endif
else
COARRAY = coarray
TARGET = target
endif

all: serial pretty openmp $(COARRAY) $(TARGET)

serial: p2p stencil transpose p2p-innerloop

pretty: stencil-pretty transpose-pretty

openmp: p2p-tasks-openmp p2p-doacross-openmp p2p-innerloop-openmp stencil-openmp transpose-openmp

tasks: p2p-tasks-openmp stencil-taskloop-openmp transpose-tasks-openmp transpose-taskloop-openmp

coarray: p2p-coarray stencil-coarray transpose-coarray

target: p2p-openmp-target stencil-openmp-target transpose-openmp-target

ornlacc: p2p-ornlacc stencil-ornlacc transpose-ornlacc

%: %.f90
	$(FC) $(FCFLAGS) $< -o $@

stencil: stencil.f90 stencil_serial.f90
	$(FC) $(FCFLAGS) -c stencil_serial.f90 -o stencil_serial.o
	$(FC) $(FCFLAGS) $< -o $@

%-pretty: %-pretty.f90
	$(FC) $(FCFLAGS) $< -o $@

%-openmp: %.f90
	$(FC) $(FCFLAGS) $(OPENMPFLAG) $< -o $@

%-openmp: %-openmp.f90
	$(FC) $(FCFLAGS) $(OPENMPFLAG) $< -o $@

%-coarray: %-coarray.f90
	$(CAFC) $(FCFLAGS) $< $(COARRAYFLAG) -o $@

%-target: %-target.f90
	$(FC) $(FCFLAGS) $(OPENMPFLAG) $(OFFLOADFLAG) $< -o $@

%-ornlacc: %-ornlacc.f90
	$(FC) $(FCFLAGS) $(ORNLACCFLAG) $< -o $@

clean:
	-rm -f *.o
	-rm -f *.i90
	-rm -f *__genmod.f90 # Intel Fortran
	-rm -f *__genmod.mod # Intel Fortran
	-rm -f *.optrpt
	-rm -f *.dwarf
	-rm -rf *.dSYM # Mac
	-rm -f p2p stencil transpose
	-rm -f p2p-innerloop
	-rm -f *-pretty
	-rm -f *-coarray
	-rm -f *-openmp
	-rm -f *-target
	-rm -f *-ornlacc
