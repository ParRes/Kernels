PROGRAM=transpose

include ../../common/CHAPEL.defs

CHPLFLAGS=-O

.PHONY: all clean run

all: $(PROGRAM) $(PROGRAM)-serial

$(PROGRAM): $(PROGRAM).chpl
	$(CHPL) $(CHPLFLAGS) $< -o $@

$(PROGRAM)-serial: $(PROGRAM)-serial.chpl
	$(CHPL) $(CHPLFLAGS) $< -o $@

run: $(PROGRAM)
	$(PROGRAM) --iterations=10 --order=4000 --tileSize=32

clean:
	-rm -f $(PROGRAM)
	-rm -f $(PROGRAM)-serial

