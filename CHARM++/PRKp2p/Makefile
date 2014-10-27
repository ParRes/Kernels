CHARMTOP=/home/rfvander/charm-6.6.0_10_13_2014/mpi-linux-x86_64-ifort-smp-mpicxx
OPTS=-Ofast -g
CHARMC=$(CHARMTOP)/bin/charmc $(OPTS)
CHARMCci=$(CHARMTOP)/bin/charmc $(OPTS) -E

OBJS = p2p.o

all: p2p

p2p: $(OBJS)
	$(CHARMC) -language charm++ -o p2p $(OBJS)

p2p.decl.h: p2p.ci
	$(CHARMCci)  p2p.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o p2p charmrun *~

p2p.o: p2p.C p2p.decl.h
	$(CHARMC) -c p2p.C
