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

.PHONY: all clean serial pretty openmp coarray target stdpar blas ga

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
  EXTRA = target tasks taskloop
endif
# GCC (also matches pgfortran so PGI must come after)
ifeq ($(findstring gfortran,$(FC)),gfortran)
  EXTRA = target coarray taskloop openacc
endif
# PGI and LLVM Flang
ifeq ($(findstring flang,$(FC)),flang)
  EXTRA = openacc
  #FCFLAGS += -DPGI
endif
ifeq ($(findstring pgf,$(FC)),pgf)
  EXTRA = target openacc cufortran
  FCFLAGS += -DPGI
endif
ifeq ($(findstring nvf,$(FC)),nvf)
  EXTRA = target openacc cufortran
  FCFLAGS += -DNVHPC
endif
ifeq ($(findstring xlf,$(FC)),xlf)
  EXTRA = target
  FCFLAGS += $(XLFPP)-DXLF
endif

all: serial pretty openmp tasks blas mpi mpi-openmp $(EXTRA)

serial: p2p stencil transpose nstream dgemm p2p-innerloop pic pic_soa

pretty: stencil-pretty transpose-pretty nstream-pretty dgemm-pretty

openmp: p2p-tasks-openmp p2p-innerloop-openmp stencil-openmp transpose-openmp nstream-openmp dgemm-openmp pic-openmp pic_soa-openmp

tasks: p2p-tasks-openmp transpose-tasks-openmp

taskloop: stencil-taskloop-openmp transpose-taskloop-openmp nstream-taskloop-openmp dgemm-taskloop-openmp

coarray: nstream-coarray p2p-coarray stencil-coarray transpose-coarray

mpi: nstream-mpi transpose-a2a-mpi transpose-p2p-mpi transpose-get-mpi transpose-acc-mpi

mpi-openmp: nstream-mpi-openmp

ga: nstream-ga transpose-ga dgemm-ga

target: stencil-openmp-target transpose-openmp-target nstream-openmp-target dgemm-openmp-target

openacc: p2p-openacc p2p-innerloop-openacc stencil-openacc transpose-openacc nstream-openacc


stdpar: nstream-stdpar stencil-stdpar transpose-stdpar dgemm-stdpar

cuf: cufortran
cufortran: nstream-cufortran transpose-cufortran

blas: dgemm-blas

%: %.F90 prk.mod prk_mod.o
	$(FC) $(FCFLAGS) $< prk_mod.o -o $@

prk.mod prk_mod.o: prk_mod.F90
	$(FC) $(FCFLAGS) -c $< -o prk_mod.o

stencil: stencil.F90 prk.mod
	$(FC) $(FCFLAGS) -c stencil_serial.F90
	$(FC) $(FCFLAGS) stencil.F90 stencil_serial.o prk_mod.o -o $@

dgemm-pretty: dgemm-pretty.F90 prk.mod
	$(FC) $(FCFLAGS) $< prk_mod.o $(BLASFLAGS) $(STDPARFLAG) -o $@

dgemm-blas: dgemm-blas.F90 prk.mod
	$(FC) $(FCFLAGS) $< prk_mod.o $(BLASFLAGS) -o $@

%-pretty: %-pretty.F90 prk.mod
	$(FC) $(FCFLAGS) $(STDPARFLAG) $< prk_mod.o -o $@

#%-openmp: %.F90 prk.mod
#	$(FC) $(FCFLAGS) $(OPENMPFLAG) $< prk_mod.o -o $@

%-openmp: %-openmp.F90 prk.mod
	$(FC) $(FCFLAGS) $(OPENMPFLAG) $< prk_mod.o -o $@

%-ga: %-ga.F90 prk.mod
	$(MPIFORT) $(FCFLAGS) $< prk_mod.o $(GAFLAG) -o $@

%-mpi-openmp: %-mpi.F90 prk.mod
	$(MPIFORT) $(FCFLAGS) $(OPENMPFLAG) $< prk_mod.o -o $@

%-mpi: %-mpi.F90 prk.mod
	$(MPIFORT) $(FCFLAGS) $< prk_mod.o -o $@

%-coarray: %-coarray.F90 prk.mod
	$(CAFC) $(FCFLAGS) $< prk_mod.o $(COARRAYFLAG) -o $@

%-target: %-target.F90 prk.mod
	$(FC) $(FCFLAGS) $(OPENMPFLAG) $(OFFLOADFLAG) $< prk_mod.o -o $@

%-openacc: %-openacc.F90 prk.mod
	$(FC) $(FCFLAGS) $(OPENACCFLAG) $< prk_mod.o -o $@

%-cufortran: %-cufortran.cuf prk.mod
	$(FC) $(FCFLAGS) $(CUFORTFLAG) $< prk_mod.o -o $@

%-stdpar: %-stdpar.F90 prk.mod
	$(FC) $(FCFLAGS) $(STDPARFLAG) $< prk_mod.o -o $@

clean:
	-rm -f prk.mod
	-rm -f prk.f18.mod
	-rm -f *.o
	-rm -f *.i90
	-rm -f *.dbg
	-rm -f *__genmod.f90 # Intel Fortran
	-rm -f *__genmod.mod # Intel Fortran
	-rm -f flang_unparsed_file*.f90 # F18 Flang
	-rm -f *.optrpt
	-rm -f *.dwarf
	-rm -rf *.dSYM # Mac
	-rm -f p2p stencil transpose nstream dgemm
	-rm -f transpose-pointer
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
	-rm -f *-cufortran
	-rm -f pic pic_soa
	-rm -f a.out
