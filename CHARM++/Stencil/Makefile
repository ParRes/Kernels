CHARMTOP=/home/rfvander/charm/mpi-linux-x86_64-ifort-smp-mpicxx
OPTS=-Ofast -fno-alias -DRADIUS=2 -DSTAR -g -vec_report
CHARMC=$(CHARMTOP)/bin/charmc $(OPTS)
CHARMCci=$(CHARMTOP)/bin/charmc $(OPTS) -E

OBJS = stencil.o

all: stencil

stencil: $(OBJS)
	$(CHARMC) -language charm++ -o stencil $(OBJS)

stencil.decl.h: stencil.ci
	$(CHARMCci)  stencil.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o stencil charmrun *~ *optrpt

stencil.o: stencil.C stencil.decl.h
	$(CHARMC) -c stencil.C
