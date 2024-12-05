# Chapel Makefile

# TODO Find a way to make this fit into the existing Makefile system

OPTFLAGS = --fast #-sassertNoSlicing
#MODULEFLAGS = --module-dir .

BLAS = -lblas

# Debugging
DEBUGFLAGS =
ifdef DEBUG
DEBUGFLAGS = --savec c
endif

CHPL = chpl

SOURCES = dgemm.chpl dgemm-summa.chpl nstream.chpl p2p-serial-fast.chpl pic.chpl sparse.chpl \
          stencil.chpl stencil-serial.chpl stencil-opt.chpl transpose.chpl transpose-serial.chpl

EXECUTABLE = $(SOURCES:.chpl=) dgemm-summa-blas

all: $(EXECUTABLE)

%: %.chpl
	$(CHPL) $(OPTFLAGS) $(MODULEFLAGS) $(DEBUGFLAGS) $*.chpl -o $*

dgemm-summa: dgemm-summa.chpl
	$(CHPL) $(OPTFLAGS) -s useBlockDist=true -s blasImpl=off $(MODULEFLAGS) $(DEBUGFLAGS) $^ -o $@

dgemm-summa-blas: dgemm-summa.chpl
	$(CHPL) $(OPTFLAGS) -s useBlockDist=true -s blasImpl=blas $(MODULEFLAGS) $(DEBUGFLAGS) $^ $(BLAS) -o $@

clean:
	rm -f $(EXECUTABLE)
