PROGRAM=transpose

include ../../common/FORTRAN.defs
include ../../common/PRKVERSION

CFLAGS  = $(DEFAULT_OPT_FLAGS)
CFLAGS += -DPRKVERSION=$(CPRKVERSION)
CFLAGS += -DRADIUS=$(RADIUS) $(STARFLAG)
CFLAGS += -I..

.PHONY: all clean run

all: $(PROGRAM) $(PROGRAM)-omp $(PROGRAM)-1d

extra:
	make PRK_LOOP_INDEX_TYPE=int32_t $(PROGRAM)-omp-int32_t
	make PRK_LOOP_INDEX_TYPE=uint32_t $(PROGRAM)-omp-uint32_t
	make PRK_LOOP_INDEX_TYPE=int64_t $(PROGRAM)-omp-int64_t
	make PRK_LOOP_INDEX_TYPE=uint64_t $(PROGRAM)-omp-uint64_t

$(PROGRAM): $(PROGRAM).c
	$(CC) $(CFLAGS) $< -o $@

$(PROGRAM)-omp: $(PROGRAM).c
	$(CC) $(CFLAGS) $(OPENMPFLAG) $< -o $@

$(PROGRAM)-omp-$(PRK_LOOP_INDEX_TYPE): $(PROGRAM).c
	$(CC) $(CFLAGS) $(OPENMPFLAG) -DPRK_LOOP_INDEX_TYPE=$(PRK_LOOP_INDEX_TYPE) $< -o $@

run: $(PROGRAM)
	$(PROGRAM) 10 4000 32

clean:
	-rm -f *.o
	-rm -f *.i90
	-rm -f *.optrpt
	-rm -f *.dwarf
	-rm -rf *.dSYM
	-rm -f $(PROGRAM)
	-rm -f $(PROGRAM)-omp
	-rm -f $(PROGRAM)-1d
	-rm -f $(PROGRAM)-omp-*int*_t

