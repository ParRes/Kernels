AC = gnat
ACFLAGS = make

all: hello

%: %.adb
	$(AC) $(ACFLAGS) $< -o $@

clean:
	-rm -f *.i *.o *.ali
	-rm hello

