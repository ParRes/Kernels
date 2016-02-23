PROGRAM=stencil

include ../../common/FORTRAN.defs
include ../../common/PRKVERSION

ifndef RADIUS
  RADIUS=2
endif

STARFLAG   = -DSTAR

CFLAGS  = $(DEFAULT_OPT_FLAGS)
CFLAGS += -DPRKVERSION=$(CPRKVERSION)
CFLAGS += -DRADIUS=$(RADIUS) $(STARFLAG)
CFLAGS += -I..

.PHONY: all clean run

all: $(PROGRAM) $(PROGRAM)-omp

$(PROGRAM): $(PROGRAM).c
	$(CC) $(CFLAGS) $< -o $@

$(PROGRAM)-omp: $(PROGRAM).c
	$(CC) $(CFLAGS) $(OPENMPFLAG) $< -o $@

run: $(PROGRAM)
	$(PROGRAM) 10 4000 32

clean:
	-rm -f *.o
	-rm -f *.optrpt
	-rm -f *.dwarf
	-rm -rf *.dSYM
	-rm -f $(PROGRAM)
	-rm -f $(PROGRAM)-omp

