# Swift Parallel Research Kernels (PRK)

This directory contains Swift implementations of the Parallel Research Kernels benchmarks: `nstream`, `transpose`, `stencil`, `p2p`, `dgemm` (plus an Accelerate-framework variant), and `xgemm` (multi-precision GEMM, plus an Accelerate-framework variant). Metal GPU variants of `nstream`, `transpose`, `p2p`, `dgemm`, and `xgemm` are also provided.

## What is Swift?

Swift is a powerful and modern programming language developed by Apple. Originally created for iOS and macOS development, Swift is now available on multiple platforms including Linux. It combines the performance of compiled languages with the expressiveness and safety of modern programming languages.

## Prerequisites

- macOS 10.15 (Catalina) or later
- Xcode 11.0 or later (for full Swift toolchain)
- OR Swift toolchain installed via Homebrew

## Installation on macOS

### Option 1: Install Xcode (Recommended)

The easiest way to get Swift on macOS is through Xcode:

1. **Install Xcode from App Store**
   ```bash
   # Open App Store and search for "Xcode"
   # Or use the command line:
   mas install 497799835  # Xcode
   ```

2. **Install Xcode Command Line Tools**
   ```bash
   xcode-select --install
   ```

3. **Verify Swift installation**
   ```bash
   swift --version
   swiftc --version
   ```

### Option 2: Install Swift via Homebrew

If you prefer a lighter installation without the full Xcode:

```bash
# Install Swift
brew install swift

# Verify installation
swift --version
swiftc --version
```

### Option 3: Download Swift Toolchain

Download the official Swift toolchain from [swift.org](https://swift.org/download/):

1. Download the `.pkg` file for macOS
2. Run the installer
3. Add Swift to your PATH:
   ```bash
   export PATH="/Library/Developer/Toolchains/swift-latest.xctoolchain/usr/bin:$PATH"
   ```

## Building the Benchmarks

### Using Make

```bash
# Navigate to the SWIFT directory
cd PRK/SWIFT

# Check Swift installation
make check-swift

# Build all benchmarks
make all

# Or build individual benchmarks
make nstream
make transpose
make stencil
make p2p
make dgemm
make xgemm
```

### Manual Compilation

```bash
# Compile with optimization
swiftc -O -whole-module-optimization -o nstream nstream.swift
swiftc -O -whole-module-optimization -o transpose transpose.swift

# Compile for debugging
swiftc -g -o nstream nstream.swift
swiftc -g -o transpose transpose.swift
```

## Running the Benchmarks

### NSTREAM (Stream Triad)

The nstream benchmark measures memory bandwidth using the stream triad operation: `A = B + scalar * C`

```bash
# Syntax: ./nstream <iterations> <vector_length>

# Quick test
./nstream 10 1000000

# Longer benchmark
./nstream 100 10000000
```

**Example Output:**
```
Parallel Research Kernels
Swift STREAM triad: A = B + scalar * C
Number of iterations = 10
Vector length        = 1000000
Solution validates
Rate (MB/s): 8543.210987 Avg time (s): 0.003756
```

### Transpose

The transpose benchmark measures the time for matrix transpose: `B = A^T`

```bash
# Syntax: ./transpose <iterations> <matrix_order>

# Quick test
./transpose 10 1000

# Longer benchmark  
./transpose 100 2000
```

**Example Output:**
```
Parallel Research Kernels
Swift Matrix transpose: B = A^T
Number of iterations = 10
Matrix order         = 1000
Solution validates
Rate (MB/s): 2456.789123 Avg time (s): 0.006543
```

### Stencil

The stencil benchmark measures the time to apply a space-invariant, linear filter (a `star` or `grid`-shaped stencil) to a 2D grid.

```bash
# Syntax: ./stencil <iterations> <grid_dimension> [<star/grid> <radius> <tile_size>]

# Quick test
./stencil 10 1000

# Longer benchmark
./stencil 100 2000 star 2
```

**Example Output:**
```
Parallel Research Kernels
Swift stencil execution on 2D grid
Number of iterations = 10
Grid size             = 1000
Type of stencil       = star
Untiled
radius of stencil     = 2
Data type             = double precision
Solution validates
Rate (MFlops/s): 4567.891234 Avg time (s): 0.002345
```

## Using the Makefile

The provided Makefile includes several convenient targets:

```bash
# Build everything
make all

# Run quick tests
make test

# Run performance benchmarks
make benchmark

# Clean build artifacts
make clean

# Check Swift installation
make check-swift

# Install Swift via Homebrew
make install-swift

# Show help
make help
```

## Performance Characteristics

### NSTREAM Performance Factors

- **Vector Length**: Larger vectors generally show higher bandwidth (until memory limits)
- **Iterations**: More iterations provide more accurate timing measurements
- **Memory Hierarchy**: Performance depends on L1/L2/L3 cache sizes and main memory bandwidth

### Transpose Performance Factors

- **Matrix Size**: Larger matrices may show cache effects
- **Memory Access Pattern**: Transpose involves non-contiguous memory access
- **Cache Blocking**: For very large matrices, tiled algorithms perform better

## Troubleshooting

### Common Issues

1. **"swift: command not found"**
   ```bash
   # Check if Swift is installed
   which swift
   
   # Install Swift via Homebrew
   brew install swift
   
   # Or install Xcode from App Store
   ```

2. **"No such module 'Foundation'"**
   - This usually means Swift is not properly installed
   - Reinstall Swift or Xcode

3. **Permission denied when running executables**
   ```bash
   chmod +x nstream transpose
   ```

4. **Poor performance compared to C++**
   - Ensure you're compiling with optimization flags: `-O -whole-module-optimization`
   - Swift performance is generally competitive with C++ when optimized

### Performance Tuning

1. **Compile with optimization**:
   ```bash
   swiftc -O -whole-module-optimization -o nstream nstream.swift
   ```

2. **For maximum performance**, also try:
   ```bash
   swiftc -O -whole-module-optimization -Xcc -O3 -o nstream nstream.swift
   ```

3. **Profile your code**:
   ```bash
   # Compile with debug info for profiling
   swiftc -O -g -o nstream nstream.swift
   
   # Use Instruments (macOS) for detailed profiling
   instruments -t "Time Profiler" ./nstream 100 10000000
   ```

## Implementation Notes

### Design Decisions

1. **Array vs UnsafePointer**: Used Swift Arrays for safety and ease of use
2. **CFAbsoluteTime**: Used for high-precision timing measurements
3. **Functional Style**: Leveraged Swift's functional programming features where appropriate
4. **Memory Management**: Relied on Swift's automatic reference counting (ARC)

### Comparison with Other Languages

- **vs C++**: Swift provides similar performance with better safety guarantees
- **vs Python**: Swift is significantly faster, closer to C++ performance levels
- **vs Java**: Swift typically shows better performance and lower memory overhead

## Further Reading

- [Swift Programming Language Guide](https://swift.org/documentation/)
- [Swift Performance Tips](https://github.com/apple/swift/blob/main/docs/OptimizationTips.rst)
- [Parallel Research Kernels Project](https://github.com/ParRes/Kernels)

## Files

- `nstream.swift`: Swift implementation of the STREAM triad benchmark
- `transpose.swift`: Swift implementation of the matrix transpose benchmark
- `stencil.swift`: Swift implementation of the 2D stencil benchmark
- `p2p.swift`: Swift implementation of the pipeline (wavefront) benchmark
- `dgemm.swift` / `dgemm-accelerate.swift`: Swift implementations of double-precision matrix multiplication (naive and Accelerate-framework)
- `xgemm.swift` / `xgemm-accelerate.swift`: Swift implementations of multi-precision matrix multiplication (naive and Accelerate-framework)
- `*-metal.swift`: Metal GPU variants of the above benchmarks
- `Makefile`: Build system for compiling and testing
- `README.md`: This documentation file
