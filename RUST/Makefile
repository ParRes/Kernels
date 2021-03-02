# Enable verbose printing
#RCFLAGS += --cfg "VERBOSE"

# This is now a runtime option
# Stencil radius
#RCFLAGS += --cfg radius="$(RADIUS)"

# Stencil shape: star is default, uncomment to switch to grid
#RCFLAGS += --cfg grid

RCFLAGS += --release

.PHONY: all clean

all:
	cd nstream 	&& cargo build $(RCFLAGS)
	cd nstream-unsafe 	&& cargo build $(RCFLAGS)
	cd nstream-iter 	&& cargo build $(RCFLAGS)
	cd p2p 		&& cargo build $(RCFLAGS)
	cd stencil 	&& cargo build $(RCFLAGS)
	cd transpose 	&& cargo build $(RCFLAGS)
	cd dgemm 	&& cargo build $(RCFLAGS)

clean:
	cd nstream 	&& cargo clean
	cd nstream-unsafe 	&& cargo clean
	cd nstream-iter 	&& cargo clean
	cd p2p 		&& cargo clean
	cd stencil 	&& cargo clean
	cd transpose 	&& cargo clean
	cd dgemm 	&& cargo clean

