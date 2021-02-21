
all: nstream.class p2p.class stencil.class

%.class: %.scala
	scalac $<

clean:
	rm -f stencil*.class
	rm -f nstream*.class
	rm -f p2p*.class

