include ../common/make.defs
include ../common/PRKVERSION

CPPFLAGS = -DPRKVERSION=$(PRKVERSION)

# debugging
ifdef VERBOSE
    CPPFLAGS += -DVERBOSE
endif

CFLAGS = $(DEFAULT_OPT_FLAGS) $(CPPFLAGS)


ifdef PRK_USE_MMAP
  CFLAGS += -DPRK_USE_MMAP
endif

ifeq ($(findstring musl,$(CC)),musl)
  CFLAGS += -DUSE_C11_THREADS
endif

#ASMFLAGS = -fsource-asm -fverbose-asm -fasm-blocks -fcode-asm $(CFLAGS)
ASMFLAGS = -fverbose-asm $(CFLAGS)

OMPFLAGS = $(OPENMPFLAG)
TARGETFLAGS = $(OFFLOADFLAG)
CILKFLAGS = $(CILKFLAG)
ISPCFLAGS = $(ISPCFLAG)
OPENACCFLAGS = $(OPENACCFLAG)

.PHONY: all clean serial thread openmp tasks target taskloop ispc

EXTRA=
ifdef ($(ISPC))
  EXTRA += ispc
endif
ifneq ($(CILKFLAG),)
  EXTRA += cilk
endif
ifneq ($(OPENACCFLAG),)
  EXTRA += openacc
endif

all: serial thread openmp $(EXTRA)

serial: nstream \
        p2p p2p-2d p2p-tasks-2d \
	p2p-hyperplane p2p-hyperplane-2d \
	stencil stencil-2d \
	transpose transpose-2d

vector: p2p-sse p2p-avx p2p-avx-tasks-openmp

thread: transpose-thread

openmp: nstream-openmp \
	p2p-hyperplane-openmp p2p-hyperplane-2d-openmp \
	stencil-openmp stencil-2d-openmp \
	transpose-openmp transpose-2d-openmp

tasks: p2p-tasks-openmp p2p-tasks-2d-openmp

mpi: nstream-mpi transpose-a2a-mpi transpose-p2p-mpi

petsc: nstream-petsc transpose-petsc

memkind: nstream-memkind nstream-memkind-openmp

mmap: nstream-mmap nstream-mmap-openmp

target: nstream-target stencil-target transpose-target nstream-alloc-target nstream-usm-target nstream-memcpy-target nstream-ua-target

taskloop: nstream-taskloop stencil-taskloop transpose-taskloop

openacc: nstream-openacc stencil-openacc transpose-openacc

cilk: stencil-cilk transpose-cilk

ispc: transpose-ispc

p2p-hyperplane: p2p-hyperplane-openmp.c prk_util.h
	$(CC) $(CFLAGS) $< $(EXTRA_CLIBS) -o $@

nstream: nstream-openmp.c prk_util.h
	$(CC) $(CFLAGS) $< $(EXTRA_CLIBS) -o $@

stencil: stencil-openmp.c prk_util.h
	$(CC) $(CFLAGS) $< $(EXTRA_CLIBS) -o $@

transpose: transpose-openmp.c prk_util.h
	$(CC) $(CFLAGS) $< $(EXTRA_CLIBS) -o $@

p2p-2d: p2p-2d.c prk_util.h
	$(CC) $(CFLAGS) $< $(EXTRA_CLIBS) -o $@

%-2d: %-2d-openmp.c prk_util.h
	$(CC) $(CFLAGS) $< $(EXTRA_CLIBS) -o $@

%-mpi: %-mpi.c prk_util.h
	$(MPICC) $(CFLAGS) $(MPIINC) $< $(MPILIB) $(EXTRA_CLIBS) -o $@

%-petsc: %-petsc.c prk_util.h
	$(CC) $(CFLAGS) $(MPIINC) $< $(PETSCFLAG) $(MPILIB) $(EXTRA_CLIBS) -o $@

%-memkind: %-memkind.c prk_util.h
	$(CC) $(CFLAGS) $< $(EXTRA_CLIBS) $(MEMKINDFLAGS) -o $@

%-memkind-openmp: %-memkind.c prk_util.h prk_openmp.h
	$(CC) $(CFLAGS) $(OMPFLAGS) $< $(EXTRA_CLIBS) $(MEMKINDFLAGS) -o $@

%-mmap: %-mmap.c prk_util.h
	$(CC) $(CFLAGS) $< $(EXTRA_CLIBS) -o $@

%-mmap-openmp: %-mmap.c prk_util.h prk_openmp.h
	$(CC) $(CFLAGS) $(OMPFLAGS) $< $(EXTRA_CLIBS) -o $@

%-target: %-target.c prk_util.h prk_openmp.h
	$(CC) $(CFLAGS) $< $(OMPFLAGS) $(TARGETFLAGS) $(EXTRA_CLIBS) -o $@

%-taskloop: %-taskloop.c prk_util.h prk_openmp.h
	$(CC) $(CFLAGS) $< $(OMPFLAGS) $(EXTRA_CLIBS) -o $@

%-openmp: %-openmp.c prk_util.h prk_openmp.h
	$(CC) $(CFLAGS) $< $(OMPFLAGS) $(EXTRA_CLIBS) -o $@

%-openacc: %-openacc.c prk_util.h
	$(CC) $(CFLAGS) $< $(OPENACCFLAGS) $(EXTRA_CLIBS) -o $@

%-cilk: %-cilk.c prk_util.h
	$(CC) $(CFLAGS) $< $(CILKFLAGS) $(EXTRA_CLIBS) -o $@

%.ispo: %.ispc
	$(ISPC) $(CPPFLAGS) $< $(ISPCFLAGS) --pic -o $@

%-ispc: %.ispo %-ispc.c
	$(CC) $(CFLAGS) $^ $(OMPFLAGS) -o $@

%: %.c prk_util.h
	$(CC) $(CFLAGS) $< $(EXTRA_CLIBS) -o $@

%-openmp.s: %-openmp.c prk_util.h
	$(CC) $(ASMFLAGS) $(OMPFLAGS) -S $< -o $@

%.s: %.c prk_util.h
	$(CC) $(ASMFLAGS) -S $< -o $@

clean:
	-rm -f *.o
	-rm -f *.s
	-rm -f *.ll # Coriander
	-rm -f *.optrpt
	-rm -f *.dwarf
	-rm -rf *.dSYM # Mac
	-rm -f nstream p2p p2p-hyperplane stencil transpose
	-rm -f p2p-sse p2p-avx p2p-avx3 p2p-avx-tasks-openmp
	-rm -f *-2d
	-rm -f *-openmp
	-rm -f *-openacc
	-rm -f *-mpi
	-rm -f *-petsc
	-rm -f *-target
	-rm -f *-taskloop
	-rm -f *-cilk
	-rm -f *-thread
	-rm -f *-ispc
	-rm -f nstream-mmap nstream-memkind
	-rm -f nstream-mmap-openmp nstream-memkind-openmp

cleancl:
	-rm -f star[123456789].cl
	-rm -f grid[123456789].cl
