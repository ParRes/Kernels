AC = gnat
ACFLAGS = make

all: hello args nstream nstream_array transpose

%: %.adb
	$(AC) $(ACFLAGS) $< -o $@

clean:
	-rm -f *.i *.o *.ali
	-rm -f hello args nstream nstream_array transpose

