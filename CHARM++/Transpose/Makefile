CHARMTOP=/home/rfvander/charm/mpi-linux-x86_64-ifort-smp-mpicxx
OPTS=-Ofast -fno-alias -g
CHARMC=$(CHARMTOP)/bin/charmc $(OPTS)
CHARMCci=$(CHARMTOP)/bin/charmc $(OPTS) -E

OBJS = transpose.o

all: transpose

transpose: $(OBJS)
	$(CHARMC) -language charm++ -o transpose $(OBJS)

transpose.decl.h: transpose.ci
	$(CHARMCci)  transpose.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o transpose charmrun *~

transpose.o: transpose.C transpose.decl.h
	$(CHARMC) -c transpose.C
