SUBDIRS = Synch_p2p Stencil Transpose

.PHONY: all clean

all:
	for d in $(SUBDIRS) ; do echo "executing make $@ in $$d" ; make -C $$d $@ ; done

clean:
	for d in $(SUBDIRS) ; do echo "executing make $@ in $$d" ; make -C $$d $@ ; done

