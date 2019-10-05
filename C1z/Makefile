include ../common/make.defs
include ../common/PRKVERSION

CPPFLAGS = -DPRKVERSION=$(PRKVERSION)

CFLAGS = $(DEFAULT_OPT_FLAGS) $(CPPFLAGS)

# debugging
ifdef VERBOSE
  CFLAGS += -DVERBOSE
endif

ifdef PRK_USE_MMAP
  CFLAGS += -DPRK_USE_MMAP
endif

ifeq ($(findstring musl,$(CC)),musl)
  CFLAGS += -DUSE_C11_THREADS
endif

ASMFLAGS = -fsource-asm -fverbose-asm -fasm-blocks -fcode-asm $(CFLAGS)

OMPFLAGS = $(OPENMPFLAG)
TARGETFLAGS = $(OFFLOADFLAG)
ORNLACCFLAGS = $(ORNLACCFLAG)
CILKFLAGS = $(CILKFLAG)
ISPCFLAGS = $(ISPCFLAG)

.PHONY: all clean serial thread openmp target taskloop ispc # cilk

EXTRA=
ifeq ($(shell uname -s),Darwin)
  ifneq ($(findstring icc,$(CC)),icc)
    EXTRA += target
  endif
else
  EXTRA += target
endif
ifdef ($(ISPC))
  EXTRA += ispc
endif
ifneq ($(CILKFLAG),)
  EXTRA += cilk
endif
ifeq ($(findstring xlc,$(CC)),xlc)
  EXTRA = target
  CFLAGS += -DXLC
endif

all: serial thread openmp taskloop $(EXTRA)

serial: nstream p2p p2p-hyperplane stencil transpose

thread: transpose-thread

openmp: nstream-openmp p2p-simd-openmp p2p-tasks-openmp stencil-openmp transpose-openmp p2p-hyperplane-openmp

mpi: nstream-mpi

memkind: nstream-memkind nstream-memkind-openmp

mmap: nstream-mmap nstream-mmap-openmp

target: nstream-target stencil-target transpose-target

taskloop: nstream-taskloop stencil-taskloop transpose-taskloop

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

%-mpi: %-mpi.c prk_util.h
	$(MPICC) $(CFLAGS) $< $(EXTRA_CLIBS) -o $@

%-memkind: %-memkind.c prk_util.h
	$(CC) $(CFLAGS) $< $(EXTRA_CLIBS) $(MEMKINDFLAGS) -o $@

%-memkind-openmp: %-memkind.c prk_util.h
	$(CC) $(CFLAGS) $(OMPFLAGS) $< $(EXTRA_CLIBS) $(MEMKINDFLAGS) -o $@

%-mmap: %-mmap.c prk_util.h
	$(CC) $(CFLAGS) $< $(EXTRA_CLIBS) -o $@

%-mmap-openmp: %-mmap.c prk_util.h
	$(CC) $(CFLAGS) $(OMPFLAGS) $< $(EXTRA_CLIBS) -o $@

%-target: %-target.c prk_util.h
	$(CC) $(CFLAGS) $< $(OMPFLAGS) $(TARGETFLAGS) $(EXTRA_CLIBS) -o $@

%-taskloop: %-taskloop.c prk_util.h
	$(CC) $(CFLAGS) $< $(OMPFLAGS) $(EXTRA_CLIBS) -o $@

%-openmp: %-openmp.c prk_util.h
	$(CC) $(CFLAGS) $< $(OMPFLAGS) $(EXTRA_CLIBS) -o $@

%-cilk: %-cilk.c prk_util.h
	$(CC) $(CFLAGS) $< $(CILKFLAGS) $(EXTRA_CLIBS) -o $@

%.ispo: %.ispc
	$(ISPC) $(CPPFLAGS) $< $(ISPCFLAGS) -o $@

%-ispc: %-ispc.o %.ispo
	$(CC) $(CFLAGS) $^ -o $@

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
	-rm -f *-openmp
	-rm -f *-mpi
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
