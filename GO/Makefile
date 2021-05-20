all: nstream transpose dgemm

%: %.go
	go build $<

# installed in ~/go
# not quick enough to call every time (~14s for repeat invocation)
deps:
	go get -u gonum.org/v1/gonum/...

clean:
	-rm -f nstream
	-rm -f transpose
	-rm -f dgemm

