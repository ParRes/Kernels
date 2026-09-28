# Swift Parallel Research Kernels Makefile
#
# This makefile builds Swift implementations of PRK benchmarks
#

SWIFT := swift
SWIFTC := swiftc

# Compiler flags for optimization
SWIFTFLAGS := -O -whole-module-optimization

# Default target
all: nstream transpose stencil dgemm dgemm-accelerate p2p nstream-metal transpose-metal dgemm-metal p2p-metal xgemm xgemm-accelerate xgemm-metal

# Build nstream benchmark
nstream: nstream.swift
	$(SWIFTC) $(SWIFTFLAGS) -o nstream nstream.swift

# Build transpose benchmark  
transpose: transpose.swift
	$(SWIFTC) $(SWIFTFLAGS) -o transpose transpose.swift

# Build stencil benchmark
stencil: stencil.swift
	$(SWIFTC) $(SWIFTFLAGS) -o stencil stencil.swift

# Build dgemm benchmark
dgemm: dgemm.swift
	$(SWIFTC) $(SWIFTFLAGS) -o dgemm dgemm.swift

# Build dgemm-accelerate benchmark (using Accelerate framework)
dgemm-accelerate: dgemm-accelerate.swift
	$(SWIFTC) $(SWIFTFLAGS) -o dgemm-accelerate dgemm-accelerate.swift

# Build p2p benchmark
p2p: p2p.swift
	$(SWIFTC) $(SWIFTFLAGS) -o p2p p2p.swift

# Build Metal GPU benchmarks
nstream-metal: nstream-metal.swift
	$(SWIFTC) $(SWIFTFLAGS) -o nstream-metal nstream-metal.swift

transpose-metal: transpose-metal.swift
	$(SWIFTC) $(SWIFTFLAGS) -o transpose-metal transpose-metal.swift

dgemm-metal: dgemm-metal.swift
	$(SWIFTC) $(SWIFTFLAGS) -o dgemm-metal dgemm-metal.swift

p2p-metal: p2p-metal.swift
	$(SWIFTC) $(SWIFTFLAGS) -o p2p-metal p2p-metal.swift

# Build xgemm multi-precision benchmarks
xgemm: xgemm.swift
	$(SWIFTC) $(SWIFTFLAGS) -o xgemm xgemm.swift

xgemm-accelerate: xgemm-accelerate.swift
	$(SWIFTC) $(SWIFTFLAGS) -o xgemm-accelerate xgemm-accelerate.swift

xgemm-metal: xgemm-metal.swift
	$(SWIFTC) $(SWIFTFLAGS) -o xgemm-metal xgemm-metal.swift

# Run tests with default parameters
test: test-nstream test-transpose test-stencil test-dgemm test-dgemm-accelerate test-p2p test-xgemm test-xgemm-accelerate

# Run xgemm tests
test-xgemm: xgemm
	@echo "Testing xgemm..."
	./xgemm 10 500

test-xgemm-accelerate: xgemm-accelerate
	@echo "Testing xgemm-accelerate..."
	./xgemm-accelerate 10 500

test-xgemm-metal: xgemm-metal
	@echo "Testing xgemm-metal..."
	./xgemm-metal 10 500

# Run Metal GPU tests
test-metal: test-nstream-metal test-transpose-metal test-dgemm-metal test-p2p-metal test-xgemm-metal

test-nstream: nstream
	@echo "Testing nstream..."
	./nstream 10 1000000

test-transpose: transpose
	@echo "Testing transpose..."
	./transpose 10 500

test-stencil: stencil
	@echo "Testing stencil..."
	./stencil 10 500

test-dgemm: dgemm
	@echo "Testing dgemm..."
	./dgemm 10 500

test-dgemm-accelerate: dgemm-accelerate
	@echo "Testing dgemm-accelerate..."
	./dgemm-accelerate 10 500

test-p2p: p2p
	@echo "Testing p2p..."
	./p2p 10 100 100

test-nstream-metal: nstream-metal
	@echo "Testing nstream-metal..."
	./nstream-metal 10 1000000

test-transpose-metal: transpose-metal
	@echo "Testing transpose-metal..."
	./transpose-metal 10 500

test-dgemm-metal: dgemm-metal
	@echo "Testing dgemm-metal..."
	./dgemm-metal 10 500

test-p2p-metal: p2p-metal
	@echo "Testing p2p-metal..."
	./p2p-metal 10 100 100

# Run benchmarks with larger parameters
benchmark: benchmark-nstream benchmark-transpose benchmark-stencil benchmark-dgemm benchmark-dgemm-accelerate benchmark-p2p

benchmark-nstream: nstream
	@echo "Benchmarking nstream..."
	./nstream 10 100000000

benchmark-transpose: transpose
	@echo "Benchmarking transpose..."
	./transpose 10 2000

benchmark-stencil: stencil
	@echo "Benchmarking stencil..."
	./stencil 10 2000

benchmark-dgemm: dgemm
	@echo "Benchmarking dgemm..."
	./dgemm 10 1000

benchmark-dgemm-accelerate: dgemm-accelerate
	@echo "Benchmarking dgemm-accelerate..."
	./dgemm-accelerate 10 1000

benchmark-p2p: p2p
	@echo "Benchmarking p2p..."
	./p2p 10 2000 2000

# Clean build artifacts
clean:
	rm -f nstream transpose stencil dgemm dgemm-accelerate p2p nstream-metal transpose-metal dgemm-metal p2p-metal xgemm xgemm-accelerate xgemm-metal

# Install Swift (requires Homebrew)
install-swift:
	@echo "Installing Swift via Homebrew..."
	brew install swift

# Check Swift installation
check-swift:
	@echo "Checking Swift installation..."
	@which swift >/dev/null 2>&1 && echo "Swift found at: $$(which swift)" || echo "Swift not found"
	@swift --version 2>/dev/null || echo "Swift version check failed"

# Help target
help:
	@echo "Swift PRK Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all           - Build all benchmarks"
	@echo "  nstream       - Build nstream benchmark"
	@echo "  transpose     - Build transpose benchmark"
	@echo "  test          - Run tests with small parameters"
	@echo "  benchmark     - Run benchmarks with larger parameters"
	@echo "  clean         - Remove build artifacts"
	@echo "  install-swift - Install Swift via Homebrew"
	@echo "  check-swift   - Check Swift installation"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make all"
	@echo "  make test"
	@echo "  ./nstream 10 1000000"
	@echo "  ./transpose 10 1000"

.PHONY: all test benchmark clean install-swift check-swift help test-nstream test-transpose test-stencil test-dgemm test-dgemm-accelerate test-p2p test-xgemm test-xgemm-accelerate test-xgemm-metal benchmark-nstream benchmark-transpose benchmark-stencil benchmark-dgemm benchmark-dgemm-accelerate benchmark-p2p test-metal
