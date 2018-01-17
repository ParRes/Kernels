include ../common/make.defs
include ../common/PRKVERSION

CPPFLAGS = -DPRKVERSION=$(PRKVERSION)

CFLAGS = $(DEFAULT_OPT_FLAGS) $(CPPFLAGS)

# debugging
ifdef VERBOSE
  CFLAGS += -DVERBOSE
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

.PHONY: all clean serial thread openmp target taskloop cilk ispc

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

all: serial thread openmp taskloop $(EXTRA)

serial: p2p p2p-innerloop stencil transpose

thread: transpose-thread

openmp: p2p-simd-openmp p2p-tasks-openmp p2p-innerloop-openmp stencil-openmp transpose-openmp

target: stencil-target transpose-target

taskloop: stencil-taskloop transpose-taskloop

cilk: stencil-cilk transpose-cilk

ispc: transpose-ispc

p2p-innerloop: p2p-innerloop-openmp.c prk_util.h
	$(CC) $(CFLAGS) $< $(EXTRA_CLIBS) -o $@

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
	-rm -f p2p p2p-innerloop stencil transpose
	-rm -f *-openmp
	-rm -f *-target
	-rm -f *-taskloop
	-rm -f *-cilk
	-rm -f *-thread
	-rm -f *-ispc

cleancl:
	-rm -f star[123456789].cl
	-rm -f grid[123456789].cl
