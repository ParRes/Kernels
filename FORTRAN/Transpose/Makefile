include ../../common/make.defs

.PHONY: all clean run

all: transpose

transpose: transpose.f90
	$(FC) $(DEFAULT_OPT_FLAGS) $< -o $@

run: transpose
	./transpose 10 4096 32

clean:
	-rm -f transpose.o
	-rm -f transpose

