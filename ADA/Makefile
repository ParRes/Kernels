AC = gnat
ACFLAGS = make

all: nstream p2p stencil transpose

%: %.adb
	$(AC) $(ACFLAGS) $< -o $@

clean:
	-rm -f *.i *.o *.ali
	-rm -f nstream p2p stencil transpose

