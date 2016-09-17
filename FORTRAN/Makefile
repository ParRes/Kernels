SUBDIRS = Synch_p2p Stencil Transpose

.PHONY: all clean serial openmp coarray pretty

all:
	for d in $(SUBDIRS) ; do echo "executing make $@ in $$d" ; make -C $$d $@ ; done

serial:
	for d in $(SUBDIRS) ; do echo "executing make $@ in $$d" ; make -C $$d $@ ; done

openmp:
	for d in $(SUBDIRS) ; do echo "executing make $@ in $$d" ; make -C $$d $@ ; done

coarray:
	for d in $(SUBDIRS) ; do echo "executing make $@ in $$d" ; make -C $$d $@ ; done

pretty:
	for d in $(SUBDIRS) ; do echo "executing make $@ in $$d" ; make -C $$d $@ ; done

clean:
	for d in $(SUBDIRS) ; do echo "executing make $@ in $$d" ; make -C $$d $@ ; done

