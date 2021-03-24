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

BLASFLAGS = $(BLASFLAG)

ifeq ($(findstring ifort,$(FC)),ifort)
  BLASFLAGS += -heap-arrays
endif
ifeq ($(findstring ifx,$(FC)),ifx)
  BLASFLAGS += -heap-arrays
endif

.PHONY: all clean serial pretty openmp coarray target openacc ga

# Intel
ifeq ($(findstring ifort,$(FC)),ifort)
  ifeq ($(shell uname -s),Darwin)
    EXTRA = taskloop
  else
    EXTRA = taskloop coarray
  endif
endif
# OpenMP target stuff is only in IFX now
ifeq ($(findstring ifx,$(FC)),ifx)
  EXTRA = target
endif
# GCC (also matches pgfortran so PGI must come after)
ifeq ($(findstring gfortran,$(FC)),gfortran)
  EXTRA = target coarray taskloop openacc
endif
# PGI and LLVM Flang
ifeq ($(findstring pgf,$(FC)),pgf)
  EXTRA = target openacc
  FCFLAGS += -DPGI
endif
ifeq ($(findstring flang,$(FC)),flang)
  EXTRA = target openacc
  FCFLAGS += -DPGI
endif
ifeq ($(findstring xlf,$(FC)),xlf)
  EXTRA = target
  FCFLAGS += $(XLFPP)-DXLF
endif

all: serial pretty openmp tasks $(EXTRA)

serial: p2p stencil transpose nstream dgemm p2p-innerloop pic pic_soa

pretty: stencil-pretty transpose-pretty nstream-pretty dgemm-pretty

openmp: p2p-tasks-openmp p2p-innerloop-openmp stencil-openmp transpose-openmp nstream-openmp dgemm-openmp pic-openmp pic_soa-openmp

tasks: p2p-tasks-openmp transpose-tasks-openmp

taskloop: stencil-taskloop-openmp transpose-taskloop-openmp nstream-taskloop-openmp dgemm-taskloop-openmp

coarray: nstream-coarray p2p-coarray stencil-coarray transpose-coarray

mpi: nstream-mpi

mpi-openmp: nstream-mpi-openmp

ga: nstream-ga transpose-ga dgemm-ga

target: stencil-openmp-target transpose-openmp-target nstream-openmp-target dgemm-openmp-target

openacc: p2p-openacc p2p-innerloop-openacc stencil-openacc transpose-openacc nstream-openacc

stdpar: nstream-stdpar stencil-stdpar transpose-stdpar

blas: dgemm-blas

%: %.F90
	$(FC) $(FCFLAGS) $< -o $@

stencil: stencil.F90 stencil_serial.F90
	#$(FC) $(FCFLAGS) -c stencil_serial.F90 -o stencil_serial.o
	$(FC) $(FCFLAGS) $< -o $@

dgemm-pretty: dgemm-pretty.F90
	$(FC) $(FCFLAGS) $< $(BLASFLAGS) $(STDPARFLAG) -o $@

dgemm-blas: dgemm-blas.F90
	$(FC) $(FCFLAGS) $< $(BLASFLAGS) -o $@

%-pretty: %-pretty.F90
	$(FC) $(FCFLAGS) $(STDPARFLAG) $< -o $@

#%-openmp: %.F90
#	$(FC) $(FCFLAGS) $(OPENMPFLAG) $< -o $@

%-openmp: %-openmp.F90
	$(FC) $(FCFLAGS) $(OPENMPFLAG) $< -o $@

%-ga: %-ga.F90
	$(MPIFORT) $(FCFLAGS) $< $(GAFLAG) -o $@

%-mpi-openmp: %-mpi.F90
	$(MPIFORT) $(FCFLAGS) $(OPENMPFLAG) $< -o $@

%-mpi: %-mpi.F90
	$(MPIFORT) $(FCFLAGS) $< -o $@

%-coarray: %-coarray.F90
	$(CAFC) $(FCFLAGS) $< $(COARRAYFLAG) -o $@

%-target: %-target.F90
	$(FC) $(FCFLAGS) $(OPENMPFLAG) $(OFFLOADFLAG) $< -o $@

%-openacc: %-openacc.F90
	$(FC) $(FCFLAGS) $(OPENACCFLAG) $< -o $@

%-stdpar: %-stdpar.F90
	$(FC) $(FCFLAGS) $(STDPARFLAG) $< -o $@

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
	-rm -f *-blas
	-rm -f *-coarray
	-rm -f *-mpi
	-rm -f *-ga
	-rm -f *-openmp
	-rm -f *-target
	-rm -f *-openacc
	-rm -f *-stdpar
	-rm -f pic pic_soa
