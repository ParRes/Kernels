all: hello.exe nstream.exe transpose.exe

%.exe: %.cs
	mcs $< -out:$@

clean:
	-rm -f *.exe
