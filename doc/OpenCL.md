These are just some notes.  This information may not be accurate
in the future, so please determine the accuracy of this information
experimentally.

# Apple OpenCL

This works very nicely.  No known issues.

# POCL

POCL is a portable, open-source implementation of OpenCL.
We have only tested it on Mac so far.

[GitHub](https://github.com/pocl/pocl)
[Documentation](http://portablecl.org/docs/html/index.html)

The POCL available from Homebrew did not work when we tried it (Sept 2017).
Building from source as follows did.  As you can see, we used the Homebrew
LLVM 4.0 package.  The LLVM 3.8 package didn't work.

```
cmake .. \
-DCMAKE_C_COMPILER=/usr/local/Cellar/llvm@4/4.0.1/bin/clang \
-DCMAKE_CXX_COMPILER=/usr/local/Cellar/llvm@4/4.0.1/bin/clang++ \
-DCMAKE_INSTALL_PREFIX=/opt/pocl/latest && \
make && make test && make install
```

# Linux

We have tested against Beignet, Intel OpenCL, and NVIDIA OpenCL on
Ubuntu 16.04 LTS.  The CPU implementations work well, although
we have only tested Intel Core i7 and Xeon E5 (both Haswell)
processors.

We have seen one issue with NVIDIA OpenCL so far:
- https://github.com/ParRes/Kernels/issues/183

We have not yet tested OpenCL on Intel/Altera FPGAs, AMD GPUs,
or any other hardware.
