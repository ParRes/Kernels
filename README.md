# License

See LICENSE for licensing information.

# Overview

This suite contains a number of kernel operations, called Parallel
Research Kernels, plus a simple build environment intended for Linux. 

These programs must not be used as benchmarks.  They are operations to 
explore features of a hardware platform, but they do not define 
fixed problems that can be used to rank systems.  Furthermore 
they have not been optimimzed for the features of any particular system.

### Note on stream

Note that while our "nstream" operations are based on the well
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


# Build Instructions

To build the codes the user needs to make certain changes by editing text
files. Assuming the source tree is untarred in directory HOME, the
following file needs to be copied to `HOME/common/make.defs` and edited.

`HOME/common/make.defs.in` -- This file specifies the names of the C
compiler (CC), and of the MPI (Message Passing Interface) compiler MPICC
or compile script. If MPI is not going to be used, the user can ignore
the value of MPICC. The compilers should already be in your path. That
is, if you define `CC=icc`, then typing `which icc` should show a
valid path where that compiler is installed.
Special instructions for building and running codes using Charm++, Grappa, 
OpenSHMEM, or Fine-Grain MPI are in `README.special`.

The suite of kernels currently has complete parallel implementations in 
[OpenMP](http://openmp.org/), 
[MPI](http://www.mpi-forum.org/), and 
[Fine-Grain MPI](http://www.cs.ubc.ca/~humaira/fgmpi.html). 
There is also a SERIAL reference implementation. 
The suite is currently being extended to include 
[Charm++](http://charm.cs.illinois.edu/research/charm),
MPI+OpenMP, 
[OpenSHMEM](http://openshmem.org/), and
[Grappa](http://grappa.io/), 
as well as two new variations of MPI: 
  1. MPI with one-sided communications (MPIRMA) 
  2. MPI with direct use of shared memory inside coherency domains (MPISHM)
These extensions are not yet complete.

## Global make

To build all available kernels of a certain version, type in the root
directory:

| Command              | Effect |  
|----------------------|-------------------------|  
| `make all`           | builds all kernels. |  
| `make allopenmp`     | builds all OpenMP kernels. |  
| `make allmpi`        | builds all MPI kernels. |  
| `make allfgmpi`      | builds all Fine-Grain MPI kernels. |  
| `make allserial`     | builds all serial kernels. |  
| `make allmpiopenmp`  | builds all hybrid MPI+OpenMP kernels. |  
| `make allmpirma`     | builds all MPI kernels with one-sided communications. |  
| `make allshmem`      | builds all OpenSHMEM kernels. |  
| `make allmpishm`     | builds all kernels with MPI3 shared memory. |  
| `make allcharm++`    | builds all Charm++ kernels. |  
| `make allgrappa`     | builds all Grappa kernels. |  

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

Descend into the desired sub-tree and cd to the kernel(s) of interest. 
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
executing scripts in the Run subdirectory from the root of the
kernels package. Only very small examples are provided that should
complete in just a few seconds. Only a few parameters can be changed
globally; for rigorous testing, the user should run each kernel 
individually, carefully choosing the right parameters. This may involve 
editing the individual Makefile and rerunning  the code.

# Example build and run

```
make all default_opt_flags="-xP -restrict" matrix_rank=7 number_of_functions=200
./Run/smallopenmp
./Run/smallmpi
./Run/smallserial
./Run/smallcharm++
./Run/smallmpiopenmp
```

To exercise all kernels, type
```
./Run/smallall
```

