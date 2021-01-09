all: hello.exe

%.exe: %.cs
	mcs $< -out:$@

clean:
	-rm -f *.exe
