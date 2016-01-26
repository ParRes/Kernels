PROGRAM=stencil

include ../../common/make.defs

ifndef RADIUS
  RADIUS=2
endif

STARFLAG   = -DSTAR

.PHONY: all clean run

all: $(PROGRAM)

$(PROGRAM): $(PROGRAM).f90
	$(FC) $(DEFAULT_OPT_FLAGS) -DRADIUS=$(RADIUS) $< -o $@

run: $(PROGRAM)
	$(PROGRAM) 10 4096 32

clean:
	-rm -f $(PROGRAM).o
	-rm -f $(PROGRAM)

