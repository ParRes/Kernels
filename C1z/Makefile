include ../common/make.defs
include ../common/PRKVERSION

CPPFLAGS  = -DPRKVERSION=$(PRKVERSION)

CFLAGS  = $(DEFAULT_OPT_FLAGS) $(CPPFLAGS)
#CFLAGS  = $(PRKCPPFLAGS) -g3 -O0

# debugging
#CFLAGS += -DVERBOSE

ifeq ($(findstring musl,$(CC)),musl)
CFLAGS += -DUSE_C11_THREADS
endif

ASMFLAGS = -fsource-asm -fverbose-asm -fasm-blocks -fcode-asm $(CFLAGS)

OMPFLAGS = $(OPENMPFLAG)
TARGETFLAGS = $(OFFLOADFLAG)
ORNLACCFLAGS = $(ORNLACCFLAG)
CILKFLAGS = $(CILKFLAG)

.PHONY: all clean serial thread openmp target taskloop cilk

ifeq ($(shell uname -s),Darwin)
ifneq ($(findstring icc,$(CC)),icc)
TARGET = target
endif
else
TARGET = target
endif
all: serial thread openmp cilk taskloop $(TARGET)

serial: p2p p2p-innerloop stencil transpose

thread: transpose-thread

openmp: p2p-simd-openmp p2p-tasks-openmp p2p-innerloop-openmp stencil-openmp transpose-openmp

target: stencil-target transpose-target

taskloop: stencil-taskloop transpose-taskloop

cilk: stencil-cilk transpose-cilk

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

%: %.c prk_util.h
	$(CC) $(CFLAGS) $< $(EXTRA_CLIBS) -o $@

%-openmp.s: %-openmp.c prk_util.h
	$(CC) $(ASMFLAGS) $(OMPFLAGS) -S $< -o $@

%.s: %.c prk_util.h
	$(CC) $(ASMFLAGS) -S $< -o $@

clean:
	-rm -f *.o
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

cleancl:
	-rm -f star[123456789].cl
	-rm -f grid[123456789].cl
