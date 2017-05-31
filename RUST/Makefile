include ../common/RUST.defs
include ../common/PRKVERSION

ifndef RADIUS
  RADIUS=2
endif

STARFLAG = -DSTAR

RUSTC    = rustc
RCFLAGS  = $(DEFAULT_OPT_FLAGS)
RCFLAGS += -DPRKVERSION=$(PRKVERSION)
RCFLAGS += -DRADIUS=$(RADIUS) $(STARFLAG) $(DOUBLEFLAG)
RCFLAGS += -I.

.PHONY: all clean run

all: p2p stencil transpose

%: %.rs
	$(RUSTC) $(RCFLAGS) $< -o $@

clean:
	-rm -f *.o
	-rm -f *.optrpt
	-rm -f *.dwarf
	-rm -rf *.dSYM
	-rm -f p2p stencil transpose

