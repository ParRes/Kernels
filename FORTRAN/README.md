# Fortran coarrays

## Compilers

### Cray

`COARRAYFLAG=-h caf`

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

### GCC

Purpose|`COARRAYFLAG`|Library
---|---|---
Serial|`-fcoarray=single`|`-lcaf_single`
Parallel|`-fcoarray=lib`|`-lcaf_mpi`

See [OpenCoarrays Getting Started](https://github.com/sourceryinstitute/opencoarrays/blob/master/GETTING_STARTED.md) for details.
