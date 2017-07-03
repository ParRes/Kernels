include ../common/RUST.defs
include ../common/PRKVERSION

ifndef RADIUS
  RADIUS=2
endif

RUSTC    = rustc
RCFLAGS  = -g

# Enable verbose printing
#RCFLAGS += --cfg "VERBOSE"

# This is now a runtime option
# Stencil radius
#RCFLAGS += --cfg radius="$(RADIUS)"

# Stencil shape: star is default, uncomment to switch to grid
#RCFLAGS += --cfg grid

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

