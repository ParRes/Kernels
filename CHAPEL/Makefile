CHPL      = chpl
CHPLFLAGS =

.PHONY: serial bigthree all clean

all: bigthree dgemm nstream sparse

serial: p2p-serial-fast stencil-serial transpose-serial

bigthree: p2p-serial-fast stencil-opt stencil-serial stencil transpose-serial transpose

pic: pic.chpl random_draw.h random_draw.c
	$(CHPL) $(CHPLFLAGS) $< -o $@

%: %.chpl
	$(CHPL) $(CHPLFLAGS) $< -o $@

clean:
	-rm -f dgemm
	-rm -f nstream
	-rm -f p2p-serial-fast
	-rm -f pic
	-rm -f sparse
	-rm -f stencil-opt
	-rm -f stencil-serial
	-rm -f stencil
	-rm -f transpose-serial
	-rm -f transpose
