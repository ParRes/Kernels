PROGRAM=transpose

include ../../common/make.defs

.PHONY: all clean run

all: $(PROGRAM)

$(PROGRAM): $(PROGRAM).f90
	$(FC) $(DEFAULT_OPT_FLAGS) $< -o $@

run: $(PROGRAM)
	$(PROGRAM) 10 4096 32

clean:
	-rm -f $(PROGRAM).o
	-rm -f $(PROGRAM)

