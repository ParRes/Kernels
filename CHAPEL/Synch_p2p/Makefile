PROGRAM=p2p

include ../../common/CHAPEL.defs

CHPLFLAGS=-O

.PHONY: all clean run

all: $(PROGRAM)

$(PROGRAM): $(PROGRAM)-serial-fast.chpl
	$(CHPL) $(CHPLFLAGS) $< -o $@

run: $(PROGRAM)
	$(PROGRAM) --iterations=10 --m=1000 --n=1000

clean:
	-rm -f $(PROGRAM)

