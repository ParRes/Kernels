include ../common/RUST.defs
include ../common/PRKVERSION

ifndef RADIUS
  RADIUS=2
endif

STARFLAG = -DSTAR

RUSTC    = rustc
RCFLAGS  = -g

# Enable verbose printing
#RCFLAGS += --cfg=VERBOSE

# Port these over to cfg
#RCFLAGS += #-DRADIUS=$(RADIUS) $(STARFLAG) $(DOUBLEFLAG)

.PHONY: all clean run

#all: p2p stencil transpose
all: transpose

%: %.rs
	$(RUSTC) $(RCFLAGS) $< -o $@

clean:
	-rm -f *.o
	-rm -f *.optrpt
	-rm -f *.dwarf
	-rm -rf *.dSYM
	-rm -f p2p stencil transpose

