# Style

Fortran implementations of the PRK should be written in ISO Fortran 2008, without reliance upon compiler extensions, with a few exceptions:
 * Preprocessing can be used, because the portable fallback is to use the C preprocessor if the Fortran compiler cannot handle this.
 * Directives can be used, because they are in comments and should have no effect if not supported.
 * OpenMP is not part of ISO Fortran, but Fortran OpenMP compilers have well-defined behavior, so long as one only uses Fortran features that are defined to be supported in OpenMP.

## OpenMP

One importance note on the prior point is that `DO CONCURRENT` is part of Fortran 2008 and its interaction with OpenMP is not defined.  Thus, one should not use `DO CONCURRENT` with OpenMP.  OpenMP supports simple `DO` loops and `WORKSHARE` for implicit loops (i.e. array notation), which should be used instead.

Side note: Intel Fortran maps `DO CONCURRENT` to something along the lines of `omp parallel do simd`.  It is likely that other compilers will do this.

## Fortran 2015

Support for Fortran 2015 coarray features are not yet widely available and may not be used.

## Documents

The [GCC Wiki](https://gcc.gnu.org/wiki/GFortranStandards) has an excellent collection of links to standards documents.

# Compiler Options

## Syntax and Standards

In `common/make.defs`, use the following to get the right language support.  In particular, one must explicitly preprocess Stencil.

Compiler|`FC`|`COARRAYFLAG`|`OPENMPFLAG`
---|---|---|---
Intel|`ifort -std08 -fpp`|see below|`-qopenmp` or `-fopenmp`
GCC|`gfortran-5 -std=f2008 -cpp`|see below|`-fopenmp`
Cray|`ftn -e F`|`-h caf`|`-h omp`
IBM|`xlf2008_r`|see below|`-qsmp=omp`

### Intel

Platform|`COARRAYFLAG`
---|---
Mac|Not supported
Linux shared-memory|`-coarray`
Linux distributed-memory|`-coarray=distributed`

You specify the number of images to use with
`FOR_COARRAY_NUM_IMAGES=N` at job execution time or
`-coarray_num_images=N` at compile time.
Please use the former.

See [Tutorial: Using Coarray Fortran](https://software.intel.com/en-us/compiler_15.0_coa_f) for details.

####  Debugging

When developing, it is useful to build with the Intel Fortran compiler using
```sh
FLAGS="-O0 -g3 -warn all -traceback -check bounds"
```
where `FLAGS` is `PRK_FLAGS` when building in the top directory and `DEFAULT_OPT_FLAGS` in a specific subdirectory.

#### Coarray images

Intel compiler:
```sh
export FOR_COARRAY_NUM_IMAGES=32
```
[Documentation](https://software.intel.com/en-us/node/532830)

### GCC

Purpose|Compiler|Library
---|---|---
Serial|`-fcoarray=single`|`-lcaf_single`
Parallel|`-fcoarray=lib`|`-lcaf_mpi`

See [OpenCoarrays Getting Started](https://github.com/sourceryinstitute/opencoarrays/blob/master/GETTING_STARTED.md) for details.

#### Installing OpenCoarrays

This is working now.  See our Travis scripts for details.

https://gcc.gnu.org/wiki/Coarray may be useful.

```sh
# MPICH_* override because MPICH was built against GCC-5
export MPICH_CC=gcc-6
export MPICH_FC=gfortran-6
export MPI_DIR=/opt/mpich/dev/gcc/default # Jeff's Mac
git clone https://github.com/sourceryinstitute/opencoarrays.git && cd opencoarrays && \
mkdir build && cd build && \
CC=$MPI_DIR/bin/mpicc FC=$MPI_DIR/bin/mpifort cmake .. \
-DCMAKE_INSTALL_PREFIX=/opt/opencoarrays/mpich-dev-gcc-6 \
-DMPI_C_COMPILER=$MPI_DIR/bin/mpicc -DMPI_Fortran_COMPILER=$MPI_DIR/bin/mpifort && \
make -j4 && \
ctest && \
make install
# This is required because OpenCoarrays "caf" compiler wrapper does not capture absolute path.
export CAFC=$MPI_DIR/bin/mpifort
```

### Cray

You need to set `XT_SYMMETRIC_HEAP_SIZE` to something much larger than the default to run nontrivial problems across more than one node (with one node, this setting does not matter).  This is no different from SHMEM or UPC programs on Cray systems.

Below is a series of examples, the last of which segfaults because of inadequate symmetric heap space for the Cray PGAS runtime.

Requesting too much symmetric heap space is also a problem, because this space is allocated during job launch.  Make sure that the symmetric heap space times the number of processing elements is less than the total memory available, and probably less than 50% of the total available.

```
jhammond@nid00292:~/PRK/git> OMP_NUM_THREADS=1 XT_SYMMETRIC_HEAP_SIZE=1G srun -N 2 -n 48 -c 1 ./FORTRAN/Transpose/transpose-coarray 10 12000 32
Parallel Research Kernels version     2.16
Fortran coarray Matrix transpose: B = A^T
Number of images     =       48
Matrix order         =    12000
Tile size            =       32
Number of iterations =       10
Solution validates
Rate (MB/s):  8392.037250   Avg time (s):   0.274546
jhammond@nid00292:~/PRK/git> OMP_NUM_THREADS=1 XT_SYMMETRIC_HEAP_SIZE=500M srun -N 2 -n 48 -c 1 ./FORTRAN/Transpose/transpose-coarray 10 12000 32
Parallel Research Kernels version     2.16
Fortran coarray Matrix transpose: B = A^T
Number of images     =       48
Matrix order         =    12000
Tile size            =       32
Number of iterations =       10
Solution validates
Rate (MB/s):  8865.425479   Avg time (s):   0.259886
jhammond@nid00292:~/PRK/git> OMP_NUM_THREADS=1 XT_SYMMETRIC_HEAP_SIZE=200M srun -N 2 -n 48 -c 1 ./FORTRAN/Transpose/transpose-coarray 10 12000 32
Parallel Research Kernels version     2.16
Fortran coarray Matrix transpose: B = A^T
Number of images     =       48
Matrix order         =    12000
Tile size            =       32
Number of iterations =       10
Solution validates
Rate (MB/s):  8801.282141   Avg time (s):   0.261780
jhammond@nid00292:~/PRK/git> OMP_NUM_THREADS=1 XT_SYMMETRIC_HEAP_SIZE=50M srun -N 2 -n 48 -c 1 ./FORTRAN/Transpose/transpose-coarray 10 12000 32
Parallel Research Kernels version     2.16
Fortran coarray Matrix transpose: B = A^T
Number of images     =       48
Matrix order         =    12000
Tile size            =       32
Number of iterations =       10
Solution validates
Rate (MB/s):  8431.348271   Avg time (s):   0.273266
jhammond@nid00292:~/PRK/git> OMP_NUM_THREADS=1 XT_SYMMETRIC_HEAP_SIZE=5M srun -N 2 -n 48 -c 1 ./FORTRAN/Transpose/transpose-coarray 10 12000 32
srun: error: nid00293: tasks 24-47: Segmentation fault
srun: Terminating job step 568052.39
slurmstepd: *** STEP 568052.39 ON nid00292 CANCELLED AT 2016-06-16T16:21:01 ***
srun: Job step aborted: Waiting up to 32 seconds for job step to finish.
srun: error: nid00292: tasks 0-23: Segmentation fault
```

### IBM XLF

Disclaimer: The basis for these comments is the IBM Blue Gene/Q compiler (version 14), which does not support Fortran 2008 coarrays.

The IBM compiler will not preprocess files with a lower-case "f" suffix.  Until we rename the files, you need to rename or symlink them yourself to `*.F90` instead of `*.f90`.

The IBM compiler does not directly support the C preprocessor.  You need to substitute `-DFOO=BAR` for `-WF,-DFOO=BAR` manually.

For example, the PRK build system will generate a build command like this:
```sh
bgxlf2008_r -O3 -DPRKVERSION="'2.16'" -DRADIUS=2 -DSTAR stencil.f90 -o stencil
bgxlf2008_r -O3 -DPRKVERSION="'2.16'" -DRADIUS=2 -DSTAR -qsmp=omp stencil.f90 -o stencil-omp
```

You need to manually convert it to this:
```sh
bgxlf2008_r -O3  -WF,-DPRKVERSION="'2.16'" -WF,-DRADIUS=2 -qfree=f90 stencil.F90 -o stencil
bgxlf2008_r -O3  -WF,-DPRKVERSION="'2.16'" -WF,-DRADIUS=2 -qfree=f90 -qsmp=omp stencil.F90 -o stencil-omp
```
