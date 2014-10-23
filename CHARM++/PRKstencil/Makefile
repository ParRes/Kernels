OPTS=-Ofast -DRADIUS=2 -DSTAR -g
CHARMC=../../../bin/charmc $(OPTS)
CHARMCci=../../../bin/charmc $(OPTS) -E

OBJS = stencil.o

all: stencil

stencil: $(OBJS)
	$(CHARMC) -language charm++ -o stencil $(OBJS)

stencil.decl.h: stencil.ci
	$(CHARMCci)  stencil.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o stencil charmrun *~

stencil.o: stencil.C stencil.decl.h
	$(CHARMC) -c stencil.C
