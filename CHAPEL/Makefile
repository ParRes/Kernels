# Chapel Makefile

# TODO Find a way to make this fit into the existing Makefile system

OPTFLAGS = --fast #-sassertNoSlicing
#OPTFLAGS_SUMMA = -suseBlockDist=true -sblasImpl=blas
OPTFLAGS_SUMMA = -s useBlockDist=true -s blasImpl=off -s lapackImpl=off
#MODULEFLAGS = --module-dir .

# Debugging
DEBUGFLAGS =
ifdef DEBUG
DEBUGFLAGS = --savec c
endif

CHPL = chpl

SOURCES = dgemm.chpl dgemm-summa.chpl nstream.chpl p2p-serial-fast.chpl pic.chpl sparse.chpl \
          stencil.chpl stencil-serial.chpl stencil-opt.chpl transpose.chpl transpose-serial.chpl

EXECUTABLE = $(SOURCES:.chpl=)

all: $(EXECUTABLE)

%: %.chpl
	$(CHPL) $(OPTFLAGS) $(MODULEFLAGS) $(DEBUGFLAGS) $*.chpl -o $*

dgemm-summa: dgemm-summa.chpl
	$(CHPL) $(OPTFLAGS) $(OPTFLAGS_SUMMA) $(MODULEFLAGS) $(DEBUGFLAGS) $^ -o $@

clean:
	rm -f $(EXECUTABLE)
