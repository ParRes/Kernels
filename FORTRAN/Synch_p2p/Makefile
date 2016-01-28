PROGRAM=p2p

include ../../common/make.defs
include ../../common/PRKVERSION

FCFLAGS  = $(DEFAULT_OPT_FLAGS)
FCFLAGS += -DPRKVERSION=$(PRKVERSION)
FCFLAGS += -DRADIUS=$(RADIUS) $(STARFLAG)

.PHONY: all clean run

all: $(PROGRAM)

$(PROGRAM): $(PROGRAM).f90
	$(FC) $(FCFLAGS) $< -o $@

$(PROGRAM)-omp: $(PROGRAM).f90
	$(FC) $(FCFLAGS) $(OPENMPFLAG) $< -o $@

run: $(PROGRAM)
	$(PROGRAM) 10 4096 32

clean:
	-rm -f $(PROGRAM)
	-rm -f $(PROGRAM)-omp

