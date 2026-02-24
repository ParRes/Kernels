# COBOL Parallel Research Kernels Makefile
#
# This makefile builds COBOL implementations of PRK benchmarks using GNU COBOL
#

# GNU COBOL compiler
COBC := cobc

# Compiler flags for optimization
COBCFLAGS := -O3 -x

# Default target
all: nstream transpose p2p dgemm

# Build nstream benchmark
nstream: nstream.cob
	$(COBC) $(COBCFLAGS) -o nstream nstream.cob

# Build transpose benchmark  
transpose: transpose.cob
	$(COBC) $(COBCFLAGS) -o transpose transpose.cob

# Build p2p benchmark
p2p: p2p.cob
	$(COBC) $(COBCFLAGS) -o p2p p2p.cob

# Build dgemm benchmark
dgemm: dgemm.cob
	$(COBC) $(COBCFLAGS) -o dgemm dgemm.cob

# Run tests with default parameters
test: test-nstream test-transpose test-p2p test-dgemm

test-nstream: nstream
	@echo "Testing nstream..."
	./nstream 10 100000

test-transpose: transpose
	@echo "Testing transpose..."
	./transpose 100 10

test-p2p: p2p
	@echo "Testing p2p..."
	./p2p 10 50 50

test-dgemm: dgemm
	@echo "Testing dgemm..."
	./dgemm 10 100

# Run benchmarks with larger parameters
benchmark: benchmark-nstream benchmark-transpose benchmark-p2p benchmark-dgemm

benchmark-nstream: nstream
	@echo "Benchmarking nstream..."
	./nstream 10 1000000

benchmark-transpose: transpose
	@echo "Benchmarking transpose..."
	./transpose 500 10

benchmark-p2p: p2p
	@echo "Benchmarking p2p..."
	./p2p 10 200 200

benchmark-dgemm: dgemm
	@echo "Benchmarking dgemm..."
	./dgemm 10 200

# Clean build artifacts
clean:
	rm -f nstream transpose p2p dgemm

# Install GNU COBOL (requires package manager)
install-cobol-ubuntu:
	@echo "Installing GNU COBOL on Ubuntu..."
	sudo apt-get update
	sudo apt-get install -y gnucobol

install-cobol-macos:
	@echo "Installing GNU COBOL on macOS via Homebrew..."
	brew install gnu-cobol

install-cobol-fedora:
	@echo "Installing GNU COBOL on Fedora..."
	sudo dnf install -y gnucobol

# Check GNU COBOL installation
check-cobol:
	@echo "Checking GNU COBOL installation..."
	@which cobc >/dev/null 2>&1 && echo "GNU COBOL found at: $$(which cobc)" || echo "GNU COBOL not found"
	@cobc --version 2>/dev/null || echo "GNU COBOL version check failed"

# Help target
help:
	@echo "COBOL PRK Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build all benchmarks"
	@echo "  nstream          - Build nstream benchmark"
	@echo "  transpose        - Build transpose benchmark"
	@echo "  p2p              - Build p2p benchmark"
	@echo "  dgemm            - Build dgemm benchmark"
	@echo "  test             - Run tests with small parameters"
	@echo "  benchmark        - Run benchmarks with larger parameters"
	@echo "  clean            - Remove build artifacts"
	@echo "  install-cobol-*  - Install GNU COBOL (ubuntu/macos/fedora)"
	@echo "  check-cobol      - Check GNU COBOL installation"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  GNU COBOL (cobc) compiler"
	@echo ""
	@echo "Usage examples:"
	@echo "  make all"
	@echo "  make test"
	@echo "  ./nstream 10 100000"
	@echo "  ./transpose 100 10"

.PHONY: all test benchmark clean install-cobol-ubuntu install-cobol-macos install-cobol-fedora check-cobol help test-nstream test-transpose test-p2p test-dgemm benchmark-nstream benchmark-transpose benchmark-p2p benchmark-dgemm
