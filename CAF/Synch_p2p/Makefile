PROGRAM=p2p

include ../../common/make.defs
include ../../common/PRKVERSION

FCFLAGS  = -fcoarray=lib -cpp
FCFLAGS += $(DEFAULT_OPT_FLAGS)
FCFLAGS += -DPRKVERSION=$(PRKVERSION)
FCFLAGS += -DRADIUS=$(RADIUS) $(STARFLAG)
FCFLAGS += -L$(OCAS)/src/mpi
LIBCAF   = -lcaf_mpi

.PHONY: all clean

all: $(PROGRAM)

$(PROGRAM): $(PROGRAM).f90
	$(MPIF90) $(FCFLAGS) $< -o $@ $(LIBCAF)

clean:
	-rm -f $(PROGRAM)
