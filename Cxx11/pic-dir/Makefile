

all: pic-dpcpp

pic-dpcpp: pic-dpcpp.cc random_draw.o
	dpcpp -DPRKVERSION=\"ALCF\" -DUSE_CPU -g -O3 pic-dpcpp.cc random_draw.o -o pic-dpcpp

random_draw.o: random_draw.c random_draw.h
	clang -O3 -c random_draw.c -o random_draw.o

clean:
	-rm -f random_draw.o
	-rm -f pic-dpcpp.o
	-rm -f pic-dpcpp
