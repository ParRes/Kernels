![PRK logo.](https://github.com/ParRes/Kernels/blob/default/logo/PRK%20logo.png)

[![license](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/ParRes/Kernels/blob/master/COPYING)
[![GitHub contributors](https://img.shields.io/github/contributors/ParRes/Kernels.svg)]()
[![GitHub language count](https://img.shields.io/github/languages/count/ParRes/Kernels.svg)]()
[![GitHub top language](https://img.shields.io/github/languages/top/ParRes/Kernels.svg)]()

# Overview

This suite contains a number of kernel operations, called Parallel
Research Kernels, plus a simple build system intended for a Linux-compatible environment.
Most of the code relies on open standard programming models and thus can be
executed on many computing systems.

These programs should not be used as benchmarks.  They are operations to 
explore features of a hardware platform, but they do not define 
fixed problems that can be used to rank systems.  Furthermore 
they have not been optimized for the features of any particular system.

# Build Instructions

To build the codes the user needs to make certain changes by editing text
files. Assuming the source tree is untarred in directory `$PRK`, the
following file needs to be copied to `$PRK/common/make.defs` and edited.

`$PRK/common/make.defs.in` -- This file specifies the names of the C
compiler (`CC`), and of the MPI (Message Passing Interface) compiler `MPICC`
or compile script. If MPI is not going to be used, the user can ignore
the value of `MPICC`. The compilers should already be in your path. That
is, if you define `CC=icc`, then typing `which icc` should show a
valid path where that compiler is installed.
Special instructions for building and running codes using Charm++, Grappa, 
OpenSHMEM, or Fine-Grain MPI are in `README.special`.

We provide examples of working examples for a number of programming environments.
Some of these are tested more than others.
If you are looking for the simplest option, try `make.defs.gcc`.

| File (in `./common/`) | Environment |  
|----------------------|-------------------------|  
| `make.defs.cray`     | Cray toolchain (rarely tested). |
| `make.defs.cuda`     | GCC with the CUDA compiler (only used in C++/CUDA implementation). |
| `make.defs.gcc`      | GCC compiler toolchain, which supports essentially all implementations (tested often). |
| `make.defs.freebsd`  | FreeBSD (rarely tested). |
| `make.defs.ibmbg`    | IBM Blue Gene/Q compiler toolchain (deprecated). |
| `make.defs.ibmp9nv`  | IBM compilers for POWER9 and NVIDIA Volta platforms (rarely tested). |
| `make.defs.intel`    | Intel Parallel Studio toolchain, which supports most implementations (tested often). |
| `make.defs.llvm`     | LLVM compiler toolchain, which supports most implementations (tested often). |
| `make.defs.musl`     | GCC compiler toolchain with MUSL as the C standard library, which was required to use C11 threads. |
| `make.defs.nvhpc`    | [NVIDIA HPC SDK](https://developer.nvidia.com/nvidia-hpc-sdk-downloads), which supports most implementations (tested often). |
| `make.defs.oneapi`   | Intel [oneAPI](https://software.intel.com/oneapi/hpc-kit). |
| `make.defs.pgi`      | PGI compiler toolchain (infrequently tested). |
| `make.defs.hip`      | HIP compiler toolchain (infrequently tested). |

Some of the C++ implementations require you to install Boost, RAJA, Kokkos, Parallel STL, respectively,
and then modify `make.defs` appropriately.  Please see the documentation in the
[documentation](https://github.com/ParRes/Kernels/tree/default/doc) (`doc`) subdirectory.

You can refer to the `travis` subdirectory for install scripts that can be readily modified
to install any of the dependencies in your local environment.

# Supported Programming Models

The suite of kernels currently has complete parallel implementations in 
[OpenMP](http://openmp.org/), 
[MPI](http://www.mpi-forum.org/), Adaptive MPI and 
[Fine-Grain MPI](http://www.cs.ubc.ca/~humaira/fgmpi.html). 
There is also a SERIAL reference implementation. 

The suite is currently being extended to include 
[Charm++](http://charm.cs.illinois.edu/research/charm),
MPI+OpenMP, 
[OpenSHMEM](http://openshmem.org/), UPC, and
[Grappa](http://grappa.io/), 
Fortran with coarrays,
as well as three new variations of MPI: 
  1. MPI with one-sided communications (MPIRMA) 
  2. MPI with direct use of shared memory inside coherency domains (MPISHM)
  3. MPI with OpenMP inside coherency domains (MPIOPENMP)
These extensions are not yet complete.

More recently, we have implemented many single-node programming models in modern languages.

## Modern C++

y = yes

i = in-progress, incomplete, incorrect, or incredibly slow

f = see footnotes

| Parallelism          | p2p | stencil | transpose | nstream | sparse | dgemm | PIC |
|----------------------|-----|---------|-----------|---------|--------|-------|-----|
| None                 |  y  |    y    |     y     |    y    |    y   |   y   |  y  |
| C++11 threads, async |     |         |     y     |         |        |       |     |
| OpenMP               |  y  |    y    |     y     |    y    |        |       |     |
| OpenMP tasks         |  y  |    y    |     y     |    y    |        |       |     |
| OpenMP target        |  y  |    y    |     y     |    y    |        |       |     |
| OpenCL 1.x           |  i  |    y    |     y     |    y    |        |       |     |
| SYCL                 |  i  |    y    |     y     |    y    |        |   y   |  y  |
| Boost.Compute        |     |         |           |    y    |        |       |     |
| Parallel STL         |  y  |    y    |     y     |    y    |        |       |     |
| Thrust               |     |         |     i     |    y    |        |       |     |
| TBB                  |  y  |    y    |     y     |    y    |        |       |     |
| Kokkos               |  y  |    y    |     y     |    y    |        |       |     |
| RAJA                 |  y  |    y    |     y     |    y    |        |       |     |
| CUDA                 |  i  |    y    |     y     |    y    |        |       |     |
| CUBLAS               |     |         |     y     |    y    |        |   y   |     |
| HIP                  |  i  |    y    |     y     |    y    |        |       |     |
| HIPBLAS              |     |         |     y     |    y    |        |   y   |     |
| CBLAS                |     |         |     y     |         |        |   y   |     |
| OpenACC              |  y  |         |           |         |        |       |     |
| MPI (RMA)            |     |         |           |    y    |        |       |     |

* [SYCL](http://sycl.tech/)
* [Boost.Compute](http://boostorg.github.io/compute/)
* [TBB](https://www.threadingbuildingblocks.org/)
* [Kokkos](https://github.com/kokkos/kokkos)
* [RAJA](https://github.com/LLNL/RAJA)

## Modern C

| Parallelism          | p2p | stencil | transpose | nstream | sparse |
|----------------------|-----|---------|-----------|---------|--------|
| None                 |  y  |    y    |     y     |    y    |        |
| C11 threads          |     |         |     y     |         |        |
| OpenMP               |  y  |    y    |     y     |    y    |        |
| OpenMP tasks         |  y  |    y    |     y     |    y    |        |
| OpenMP target        |  y  |    y    |     y     |    y    |        |
| Cilk                 |     |    y    |     y     |         |        |
| ISPC                 |     |         |     y     |         |        |
| MPI                  |     |         |           |    y    |        |
| PETSc                |     |         |     i     |    y    |        |

There are versions of nstream with OpenMP that support memory allocation
using [mmap](http://man7.org/linux/man-pages/man2/mmap.2.html)
and [memkind](https://github.com/memkind/memkind), which can be used
for testing novel memory systems, including persistent memory.

* [ISPC](https://ispc.github.io/)

## Modern Fortran

| Parallelism          | p2p | stencil | transpose | nstream | sparse | dgemm |
|----------------------|-----|---------|-----------|---------|--------|-------|
| None                 |  y  |    y    |     y     |    y    |        |   y   |
| Intrinsics           |     |         |     y     |    y    |        |   y   |
| coarrays             |  y  |    y    |     y     |         |        |       |
| Global Arrays        |     |         |     y     |    y    |        |       |
| OpenMP               |  y  |    y    |     y     |    y    |        |   y   |
| OpenMP tasks         |  y  |    y    |     y     |    y    |        |       |
| OpenMP target        |  y  |    y    |     y     |    y    |        |       |
| OpenACC              |     |    y    |     y     |    y    |        |       |

By intrinsics, we mean the language built-in features, such as colon notation or the `TRANSPOSE` intrinsic.
We use `DO CONCURRENT` in a few places.

## Other languages

x = externally supported (in the Chapel repo)

| Parallelism          | p2p | stencil | transpose | nstream | sparse | dgemm |
|----------------------|-----|---------|-----------|---------|--------|-------|
| Python 3             |  y  |    y    |     y     |    y    |    y   |   y   |
| Python 3 w/ Numpy    |  y  |    y    |     y     |    y    |    y   |   y   |
| Python 3 w/ mpi4py   |     |    y    |     y     |    y    |        |       |
| Julia                |  y  |    y    |     y     |    y    |        |   y   |
| Octave (Matlab)      |  y  |    y    |     y     |         |        |       |
| Rust                 |  y  |    y    |     y     |         |        |       |
| Go                   |     |         |     y     |    y    |        |   y   |
| C#                   |     |         |     y     |    y    |        |       |
| Chapel               |  x  |    x    |     x     |         |        |       |
| Java                 |  y  |    y    |     y     |    y    |        |       |
| Lua                  |     |         |           |    y    |        |       |

## Global make

Please run `make help` in the top directory for the latest information.

To build all available kernels of a certain version, type in the root
directory:

| Command              | Effect |  
|----------------------|-------------------------|  
| `make all`           | builds all kernels. |  
| `make allserial`     | builds all serial kernels. |  
| `make allopenmp`     | builds all OpenMP kernels. |  
| `make allmpi`        | builds all conventional two-sided MPI kernels. |  
| `make allmpi1`       | builds all MPI kernels. |  
| `make allfgmpi`      | builds all Fine-Grain MPI kernels. | 
| `make allampi`       | builds all Adaptive MPI kernels. |  
| `make allmpiopenmp`  | builds all hybrid MPI+OpenMP kernels. |  
| `make allmpirma`     | builds all MPI-3 kernels with one-sided communications. |  
| `make allmpishm`     | builds all kernels with MPI-3 shared memory. | 
| `make allshmem`      | builds all OpenSHMEM kernels. |  
| `make allupc`        | builds all Unified Parallel C (UPC) kernels. |  
| `make allcharm++`    | builds all Charm++ kernels. |  
| `make allgrappa`     | builds all Grappa kernels. |  
| `make allfortran`    | builds all Fortran kernels. |
| `make allc1x`        | builds all C99/C11 kernels. |
| `make allcxx`        | builds all C++11 kernels. |

The global make process uses a single set of optimization flags for all
kernels. For more control, the user should consider individual makes
(see below), carefully choosing the right parameters in each Makefile.
If a a single set of optimization flags different from the default is
desired, the command line can be adjusted:
`make all<version> default_opt_flags=<list of optimization flags>` 

The global make process uses some defaults for the Branch kernel
(see Makefile in that directory). These can be overridden by adjusting
the command line: 
`make all<version> matrix_rank=<n> number_of_functions=<m>`
Note that no new values for `matrix_rank` or `number_of_functions` will
be used unless a `make veryclean` has been issued.

## Individual make

Descend into the desired sub-tree and `cd` to the kernel(s) of interest. 
Each kernel has its own Makefile. There are a number of parameters 
that determine the behavior of the kernel that need to be known at 
compile time. These are explained succinctly in the Makefile itself. Edit 
the Makefile to activate certain parameters, and/or to set their values.

Typing `make` without parameters in each leaf directory will prompt
the user for the correct parameter syntax. Once the code has been
built, typing the name of the executable without any parameters will 
prompt the user for the correct parameter syntax.

# Running test suite

After the desired kernels have been built, they can be tested by
executing scripts in the 'scripts' subdirectory from the root of the
kernels package. Currently two types of run scripts are supported.
scripts/small: tests only very small examples that should complete in 
               just a few seconds. This merely tests functionality
               of kernels and installed runtimes
scripts/wide:  tests examples that will take up most memory on a 
               single node with 64 GB of memory. 

Only a few parameters can be changed globally; for rigorous testing, 
the user should run each kernel individually, carefully choosing the 
right parameters. This may involve editing the individual Makefiles 
and rerunning the kernels.

# Example build and runs

```sh
make all default_opt_flags="-O2" "matrix_rank=7" "number_of_functions=200" 
./scripts/small/runopenmp
./scripts/small/runmpi1
./scripts/wide/runserial
./scripts/small/runcharm++
./scripts/wide/runmpiopenmp
```

To exercise all kernels, type
```sh
./scripts/small/runall
./scripts/wide/runall
```

# Quality Control

We have a rather massive test matrix running in Travis CI.
Unfortunately, the Travis CI environment may vary with time and occasionally differs
from what we are running locally, which makes debugging tricky.
If the status of the project is not passing, please inspect the [details](https://travis-ci.org/ParRes/Kernels),
because this may not be an indication of an issue with our project, but rather
something in Travis CI.

# License

See [COPYING](https://github.com/ParRes/Kernels/blob/master/COPYING) for licensing information.

## Note on stream

Note that while our `nstream` operations are based on the well
known STREAM benchmark by John D. McCalpin, we modified the source 
code and do not follow the run-rules associated with this benchmark.
Hence, according to the rules defined in the STREAM license (see 
clause 3b), you must never report the results of our nstream 
operations as official "STREAM Benchmark" results. The results must 
be clearly labled whenever they are published.  Examples of proper 
labelling include: 

      "tuned STREAM benchmark results" 
      "based on a variant of the STREAM benchmark code" 

Other comparable, clear, and reasonable labelling is acceptable.
