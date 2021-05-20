
all: nstream.class p2p.class stencil.class transpose.class

%.class: %.scala
	scalac -deprecation $<

clean:
	rm -f transpose*.class
	rm -f stencil*.class
	rm -f nstream*.class
	rm -f p2p*.class

