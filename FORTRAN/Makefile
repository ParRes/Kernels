include ../common/make.defs

ifndef CAFC
  CAFC=$(FC)
endif

ifndef RADIUS
  RADIUS=2
endif

STARFLAG = $(XLFPP)-DSTAR

FCFLAGS  = $(DEFAULT_OPT_FLAGS)
FCFLAGS += $(XLFPP)-DRADIUS=$(RADIUS) $(STARFLAG)

ifeq ($(findstring ifort,$(FC)),ifort)
  BLASFLAGS += -heap-arrays
endif

.PHONY: all clean serial pretty openmp coarray target ornlacc

# Intel
ifeq ($(findstring ifort,$(FC)),ifort)
  ifeq ($(shell uname -s),Darwin)
    EXTRA = taskloop
  else
    EXTRA = taskloop target coarray
  endif
endif
# GCC (also matches pgfortran so PGI must come after)
ifeq ($(findstring gfortran,$(FC)),gfortran)
  EXTRA = target coarray taskloop ornlacc
endif
# PGI and LLVM Flang
ifeq ($(findstring pgf,$(FC)),pgf)
  EXTRA = target ornlacc
  FCFLAGS += -DPGI
endif
ifeq ($(findstring flang,$(FC)),flang)
  EXTRA = target ornlacc
  FCFLAGS += -DPGI
endif
ifeq ($(findstring xlf,$(FC)),xlf)
  EXTRA = target
  FCFLAGS += $(XLFPP)-DXLF
endif

all: serial pretty openmp tasks $(EXTRA)

serial: p2p stencil transpose nstream dgemm p2p-innerloop

pretty: stencil-pretty transpose-pretty nstream-pretty dgemm-pretty

openmp: p2p-tasks-openmp p2p-innerloop-openmp stencil-openmp transpose-openmp nstream-openmp dgemm-openmp

tasks: p2p-tasks-openmp transpose-tasks-openmp

taskloop: stencil-taskloop-openmp transpose-taskloop-openmp nstream-taskloop-openmp dgemm-taskloop-openmp

coarray: p2p-coarray stencil-coarray transpose-coarray

target: stencil-openmp-target transpose-openmp-target nstream-openmp-target dgemm-openmp-target

ornlacc: p2p-ornlacc p2p-innerloop-ornlacc stencil-ornlacc transpose-ornlacc nstream-ornlacc

%: %.F90
	$(FC) $(FCFLAGS) $< -o $@

stencil: stencil.F90 stencil_serial.F90
	#$(FC) $(FCFLAGS) -c stencil_serial.F90 -o stencil_serial.o
	$(FC) $(FCFLAGS) $< -o $@

dgemm-pretty: dgemm-pretty.F90
	$(FC) $(FCFLAGS) $(BLASFLAGS) $< -o $@

%-pretty: %-pretty.F90
	$(FC) $(FCFLAGS) $< -o $@

%-openmp: %.F90
	$(FC) $(FCFLAGS) $(OPENMPFLAG) $< -o $@

%-openmp: %-openmp.F90
	$(FC) $(FCFLAGS) $(OPENMPFLAG) $< -o $@

%-coarray: %-coarray.F90
	$(CAFC) $(FCFLAGS) $< $(COARRAYFLAG) -o $@

%-target: %-target.F90
	$(FC) $(FCFLAGS) $(OPENMPFLAG) $(OFFLOADFLAG) $< -o $@

%-ornlacc: %-ornlacc.F90
	$(FC) $(FCFLAGS) $(ORNLACCFLAG) $< -o $@

clean:
	-rm -f *.o
	-rm -f *.i90
	-rm -f *.dbg
	-rm -f *__genmod.f90 # Intel Fortran
	-rm -f *__genmod.mod # Intel Fortran
	-rm -f *.optrpt
	-rm -f *.dwarf
	-rm -rf *.dSYM # Mac
	-rm -f p2p stencil transpose nstream dgemm
	-rm -f p2p-innerloop
	-rm -f *-pretty
	-rm -f *-coarray
	-rm -f *-openmp
	-rm -f *-target
	-rm -f *-ornlacc
