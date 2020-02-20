CHPL      = chpl
CHPLFLAGS = --fast

ifeq ($(CHPL_LLVM),llvm)
    echo "CHPL_LLVM=llvm"
    CHPLFLAGS += --llvm
else
    ifeq ($(CHPL_LLVM),system)
	echo "CHPL_LLVM=system"
	CHPLFLAGS += --llvm
    endif
endif


.PHONY: serial bigthree allstencil all clean

all: bigthree dgemm nstream sparse

serial: p2p-serial-fast stencil-serial transpose-serial

allstencil: stencil-serial stencil-opt stencil-defaultdist stencil-blockdist stencil-stencildist

bigthree: p2p-serial-fast allstencil transpose-serial transpose

pic: pic.chpl random_draw.h random_draw.c
	$(CHPL) $(CHPLFLAGS) $< -o $@

stencil-defaultdist: stencil.chpl
	$(CHPL) $(CHPLFLAGS) $< -o $@

stencil-blockdist: stencil.chpl
	$(CHPL) $(CHPLFLAGS) $< -o $@

stencil-stencildist: stencil.chpl
	$(CHPL) $(CHPLFLAGS) $< -o $@

%: %.chpl
	$(CHPL) $(CHPLFLAGS) $< -o $@

clean:
	-rm -f p2p-serial-fast
	-rm -f stencil-opt stencil-serial
	-rm -f stencil-defaultdist stencil-blockdist stencil-stencildist
	-rm -f transpose-serial transpose
	-rm -f dgemm nstream pic sparse

