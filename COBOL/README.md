# COBOL Parallel Research Kernels

This directory contains COBOL implementations of the Parallel Research Kernels (PRK) benchmarks using GNU COBOL (GnuCOBOL).

## Benchmarks

### Available Benchmarks

1. **nstream** - STREAM triad: `A = B + scalar*C`
   - Tests memory bandwidth with vector operations
   - Usage: `./nstream <iterations> <vector_length>`

2. **transpose** - Matrix transpose: `B = A^T`
   - Tests efficiency of matrix transposition
   - Usage: `./transpose <matrix_order> <iterations>`

3. **p2p** - Pipeline execution on 2D grid
   - Tests stencil computation patterns
   - Usage: `./p2p <iterations> <grid_dim1> <grid_dim2>`

4. **dgemm** - Dense matrix-matrix multiplication: `C += A × B`
   - Tests floating-point computation performance
   - Usage: `./dgemm <iterations> <matrix_order>`

## Prerequisites

### GNU COBOL Installation

#### macOS (via Homebrew)
```bash
brew install gnu-cobol
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install gnucobol
```

#### Fedora/RHEL/CentOS
```bash
sudo dnf install gnucobol
# or on older systems:
# sudo yum install gnucobol
```

#### From Source
```bash
# Download from https://sourceforge.net/projects/gnucobol/
wget https://sourceforge.net/projects/gnucobol/files/gnucobol/3.2/gnucobol-3.2.tar.xz
tar -xf gnucobol-3.2.tar.xz
cd gnucobol-3.2
./configure
make
sudo make install
```

### Verification
```bash
cobc --version
```

## Building and Running

### Build All Benchmarks
```bash
make all
```

### Build Individual Benchmarks
```bash
make nstream
make transpose
make p2p
make dgemm
```

### Run Tests (Small Parameters)
```bash
make test
```

### Run Benchmarks (Larger Parameters)
```bash
make benchmark
```

### Individual Test Examples
```bash
# STREAM triad with 10 iterations on vectors of length 100,000
./nstream 10 100000

# Matrix transpose of 100x100 matrix with 10 iterations
./transpose 100 10

# Pipeline on 50x50 grid with 10 iterations
./p2p 10 50 50

# Matrix multiplication of 100x100 matrices with 10 iterations
./dgemm 10 100
```

## Implementation Notes

### COBOL Language Features Used

- **Fixed-point arithmetic** with `COMP-3` (packed decimal) for precision
- **Multi-dimensional arrays** with `OCCURS` clauses
- **Indexed access** for array operations
- **Intrinsic functions** for mathematical operations
- **Command-line argument processing**

### Array Size Limitations

The implementations use statically allocated arrays with these maximum sizes:
- **nstream**: Up to 1,000,000 elements per vector
- **transpose**: Up to 1,000×1,000 matrices
- **p2p**: Up to 500×500 grids
- **dgemm**: Up to 300×300 matrices

These limits can be increased by modifying the `OCCURS` clauses in the source files, but may require more memory.

### Performance Considerations

1. **Compilation**: Use `-O3` for optimization
2. **Memory**: COBOL uses significant memory for large arrays
3. **Precision**: Uses packed decimal for numerical stability
4. **Timing**: Uses `CURRENT-DATE` function (limited precision)

### COBOL-Specific Adaptations

- **Array indexing**: COBOL uses 1-based indexing (converted from 0-based C)
- **Variable naming**: Uses COBOL naming conventions with hyphens
- **Error handling**: Uses COBOL `STOP RUN` for error conditions
- **I/O**: Uses `DISPLAY` for output formatting

## Troubleshooting

### Common Issues

1. **"cobc: command not found"**
   - Install GNU COBOL using package manager or from source

2. **Compilation errors about array sizes**
   - Reduce the problem size or increase array limits in source code

3. **Runtime errors with large arrays**
   - Check available system memory
   - Reduce array sizes in the benchmark parameters

4. **Timing precision issues**
   - COBOL's `CURRENT-DATE` has limited precision for very fast operations
   - Use larger problem sizes for meaningful timing results

### Platform-Specific Notes

- **macOS**: GNU COBOL works well with Homebrew installation
- **Linux**: Package manager installations are generally reliable
- **Windows**: Consider using WSL or Cygwin for GNU COBOL

## Validation

Each benchmark includes validation routines that check:
- Computational correctness using reference checksums
- Numerical precision within acceptable tolerances
- Proper algorithm implementation

Success is indicated by "Solution validates" message.

## Performance Expectations

COBOL performance characteristics:
- **Strengths**: Excellent decimal arithmetic precision, robust I/O
- **Limitations**: Generally slower than compiled C/Fortran for numerical computing
- **Use case**: Demonstrates algorithm implementation in business-oriented language

## Contributing

When modifying the COBOL implementations:
1. Maintain COBOL coding standards and conventions
2. Keep array size limits reasonable for typical systems
3. Preserve numerical accuracy and validation routines
4. Update documentation for any parameter changes

## References

- [GNU COBOL Documentation](https://gnucobol.sourceforge.io/)
- [COBOL Language Reference](https://www.ibm.com/docs/en/cobol-zos)
- [Parallel Research Kernels](https://github.com/ParRes/Kernels)
