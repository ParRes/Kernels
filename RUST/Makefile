# Enable verbose printing
#RCFLAGS += --cfg "VERBOSE"

# This is now a runtime option
# Stencil radius
#RCFLAGS += --cfg radius="$(RADIUS)"

# Stencil shape: star is default, uncomment to switch to grid
#RCFLAGS += --cfg grid

.PHONY: all clean

all:
	cd p2p && cargo build
	cd stencil && cargo build
	cd transpose && cargo build

clean:
	cd p2p && cargo clean
	cd stencil && cargo clean
	cd transpose && cargo clean

