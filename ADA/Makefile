AC = gnat
ACFLAGS = make

all: hello args

%: %.adb
	$(AC) $(ACFLAGS) $< -o $@

clean:
	-rm -f *.i *.o *.ali
	-rm -f hello args

