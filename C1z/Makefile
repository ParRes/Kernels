include ../common/make.defs
include ../common/PRKVERSION

CPPFLAGS  = -DPRKVERSION=$(PRKVERSION)

CFLAGS  = $(DEFAULT_OPT_FLAGS) $(CPPFLAGS)
#CFLAGS  = $(PRKCPPFLAGS) -g3 -O0

# debugging
#CFLAGS += -DVERBOSE

OMPFLAGS = $(OPENMPFLAG)
TARGETFLAGS = $(OFFLOADFLAG)
ORNLACCFLAGS = $(ORNLACCFLAG)
CILKFLAGS = $(CILKFLAG)

.PHONY: all clean serial openmp target taskloop cilk

ifeq ($(shell uname -s),Darwin)
ifneq ($(findstring icc,$(CC)),icc)
TARGET = target
endif
else
TARGET = target
endif
all: serial openmp cilk taskloop $(TARGET)

serial: p2p stencil transpose

openmp: p2p-tasks-openmp p2p-innerloop-openmp stencil-openmp transpose-openmp

target: stencil-target transpose-target

taskloop: stencil-taskloop transpose-taskloop

cilk: stencil-cilk transpose-cilk

%-target: %-target.c prk_util.h
	$(CC) $(CFLAGS) $< $(OMPFLAGS) $(TARGETFLAGS) $(EXTRA_CLIBS) -o $@

%-taskloop: %-taskloop.c prk_util.h
	$(CC) $(CFLAGS) $< $(OMPFLAGS) $(OMPFLAGS) $(EXTRA_CLIBS) -o $@

%-openmp: %-openmp.c prk_util.h
	$(CC) $(CFLAGS) $< $(OMPFLAGS) $(EXTRA_CLIBS) -o $@

%-cilk: %-cilk.c prk_util.h
	$(CC) $(CFLAGS) $< $(CILKFLAGS) $(EXTRA_CLIBS) -o $@

%: %.c prk_util.h
	$(CC) $(CFLAGS) $< $(EXTRA_CLIBS) -o $@

clean:
	-rm -f *.o
	-rm -f *.ll # Coriander
	-rm -f *.optrpt
	-rm -f *.dwarf
	-rm -rf *.dSYM # Mac
	-rm -f p2p stencil transpose
	-rm -f *-openmp
	-rm -f *-target
	-rm -f *-taskloop
	-rm -f *-cilk

cleancl:
	-rm -f star[123456789].cl
	-rm -f grid[123456789].cl
