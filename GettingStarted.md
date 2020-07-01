# Intro

The PRK project is designed to allow easy exploration and comparison of programming models.
Everybody will have a different level of comfort with the patterns it covers, but
we have found that once one gets comfortable with these patterns, it is possible to do
a port of the PRKs to new languages in one day, at least for languages that are conceptually
similar to the ones that are already supported.

Please do not be discouraged if things are not easy.
It took us a long time to build this and we got a lot of help along the way.
In some cases, we floundered for months before reaching out to someone
with special expertise who helped to make the implementation possible.

# Prerequisites

## Operating system

We assume a Linux-like environment, which includes Linux, MacOS and BSD.

There was a time in the past that the PRK project supported Windows,
but it has not been tested in that context in _many_ years and one
should not expect that to work.

If you want to use the PRK project on a Windows machine, you have a few options:
1) [Windows Subsystem for Linux](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux)
2) Cygwin
3) Linux virtual machines
4) Port the build system and anything else to the Windows programming environment.

The first three options should be indistinguishable from Linux except with respect
to hardware-specific code such as CUDA.

If you want to contribute 4, that would be great, but it's not a priority for us.

## Software tools

### Git

You are on GitHub so presumably you have some familiarity with Git,
but you do not need Git to use the PRK project.

GitHub supports downloading a Zip file, in which case you'll need a way
to unzip that file (https://github.com/ParRes/Kernels/archive/default.zip).

### GNU Make

We use GNU Make for compiling code.
For code that is not compiled, you do not need GNU Make.
The code that doesn't need to be compiled includes Python and Octave.

### Compilers

The PRK project builds with a wide range of compilers.
We test with Intel, Clang and GCC regularly.
Compilers that do not support standards can cause problems.

Please look at `common/make.defs.${toolchain}` to see if your
programming environment is supported.

### Other dependencies

Python and Octave implementations require the appropriate environment.
We won't try to document that here since there is great documentation
online for whatever platform you are using.

All of the libraries and frameworks supported by the PRK project
can be installed using the Travis CI infrastructure.
See `travis/install-${dependency}.sh` for details and look
at how the script is invoked by `travis/install-deps.sh` to
undestand the options.
In many cases, the only required argument is the path to the
target directory.  We often use `${PRK}/deps/` for this.

# Getting the source

Try this:
```sh
git clone https://github.com/ParRes/Kernels.git PRK
```

If you are planning to contribute via GitHub pull request,
you should fork the project and do something like this:
```sh
git clone https://github.com/ParRes/Kernels.git PRK
cd PRK
git remote rename origin upstream
git remote add <address of your fork>
git fetch --all
```

# Running existing implementations

## Function

The PRK project is designed to be self-verifying.
We don't claim it is perfect but implementations are
designed to not produce performance measurements unless
the answer is correct.

Here is an example:

1) Select the GCC toolchain: `cp common/make.defs.gcc common/make.defs`.
2) Compile stuff: `make -j``nproc`` -k`.
3) Run something simple:
```
$ ./C1z/nstream 10 100000000
Parallel Research Kernels version 2020
C11 STREAM triad: A = B + scalar * C
Number of iterations = 10
Vector length        = 100000000
Solution validates
Rate (MB/s): 27243.009853 Avg time (s): 0.117461
```

## Performance

While the PRK project is not intended to serve as a benchmark,
users can compare the performance of software systems by choosing
a particular configuration and evaluating different implementations.

We will add some examples later...

# Creating new implementations

The recommended method here is to start with an implementation
in a related programming model or language.
For example, if you want to port to a new language that looks
a lot like C++, start with the C++ codes and use search-and-replace
to address simple changes in syntax.

When porting code that uses multidimensional arrays, it is strongly recommended
to start from a language with the same base index.
Converting within {Fortran, Octave, Julia} and {Python, C, C++, Rust} is
a lot easier than between the two sets (we learned this the hard way).

## History

_You should probably skip this part._
_It probably doesn't belong here but we felt like writing._

Originally, all of the PRKs were written in C89, because that is
the most portable language dialect in HPC.
When we were working on the ISC 2016 paper, we ported to
UPC (C99), Charm++ (C++) and Grappa (C++).
While we made that work, it wasn't ideal and one of the Charm++
developers wisely noted that porting MPI-1 C89 code to Charm++
is unlikely to lead to ideal results.

At some point (version control can provide details), we decided
to start porting the PRKs to new languages.
There were a few motivations.
First, we wanted to evaluate Chapel
It does not make sense to compare Chapel to C89 MPI code.
We wanted to compare Chapel to other expressive languages like Python.
Second, we wanted to make it easier for people to contribute implementations
in parallel models based on C++, and the best way to do that was
to write an idiomatic sequential C++ implementation.
Finally, since we were planning on porting to C++ and Chapel,
we figured, "why not just port to all the languages?"
As you can see, progress towards "all the languages" is rather incomplete.

Inspired by the Chapel ports, we also tried to evaluate expressive
dialects of languages that support it.
For example, one can write Fortran with a bunch of loops like it's 1986,
or one can write array notation that looks almost identical to Numpy.
We followed the Chapel convention of naming these implementations "pretty".

Despite our hopes that idiomatic C++ implementations would lead to an
outpouring of support from the community with ports to all the C++ models
like Kokkos, RAJA and TBB, this did not happen, so we started writing them ourselves.
Fortunately, we got some help along the way, both in terms of code contributions
but also code review and bug fixes.

## Porting guide

All of the PRKs parse arguments to determine the problem configuration.
In most cases, these are just integers.
If the target language has an idiomatic way to parse arguments,
e.g. Chapel, use that.
Otherwise use the C-style method based on argc/argv (or equivalent).

We recommend porting `nstream` first.
It is the simplest and if you struggle to port it,
it is unlikely that you'll have more success with others.

After `nstream`, we usually port `transpose`, followed by `stencil`.
Porting `synch_p2p` aka `p2p` and `dgemm` come next.
We rarely porting anything else, for various reasons, although
`sparse` has been ported to C++ and we would like to port `random`.

We do not port `branch`, `synch_global` and `reduce` to new languages
because there isn't much to learn from this, at least as specified.
Some day, we hope to develop next-generation versions of these
that focus on divergence and collective synchronization patterns.

Finally, we note that Rob has created PIC and AMR kernels for
evaluating load-balancing and relaxed challenges.
This was a heroic effort and the rest of us have not been
able to reproduce the level of mental acuity required to port
those to any new languages.

### `nstream`

`nstream` requires one to implement dynamic allocation of
a one-dimensional array and basic arithmetic.
The essence of this kernel is below:
```
for i from 1 to length:
    A[i] += B[i] + scalar * C[i]
```

### `transpose`

`transpose` requires one to allocate a matrix and
iterate over it in both a contiguous and strided manner.
The essence of this kernel is below:
```
for i from 1 to order: 
    for j from 1 to order:
        B(i,j) += A(j,i)
        A(j,i) += 1.0
```

### `stencil`

`stencil` looks a lot like `transpose` except that
instead of combining the contiguous and strided
representations, one combines offset views of the matrix.

We use a variety of methods to implement stencils, including
code generation (C, C++) and array notation (Python, Fortran).
Below are examples of Kokkos and Numpy.
The code generators are crude Python scripts.

```c++
void star2(const int n, const int t, matrix & in, matrix & out) {
    auto inside = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({2,2},{n-2,n-2},{t,t});
    Kokkos::parallel_for(inside, KOKKOS_LAMBDA(int i, int j) {
              out(i,j) += +in(i,j-2) * -0.125
                          +in(i,j-1) * -0.25
                          +in(i-2,j) * -0.125
                          +in(i-1,j) * -0.25
                          +in(i+1,j) * 0.25
                          +in(i+2,j) * 0.125
                          +in(i,j+1) * 0.25
                          +in(i,j+2) * 0.125;
    });
}
```

```python
b = n-r
B[r:b,r:b] += W[r,r] * A[r:b,r:b]
for s in range(1,r+1):
    B[r:b,r:b] += W[r,r-s] * A[r:b,r-s:b-s] \
                + W[r,r+s] * A[r:b,r+s:b+s] \
                + W[r-s,r] * A[r-s:b-s,r:b] \
                + W[r+s,r] * A[r+s:b+s,r:b]
```

### `p2p`

Porting `synch_p2p` aka `p2p` is complicated because this algorithm
is not data-parallel in the traditional sense.
The sequential representation is quite simple:
```python
for i in range(1,m):
    for j in range(1,n):
        grid[i][j] = grid[i-1][j] + grid[i][j-1] - grid[i-1][j-1]
```

There are a few different methods for porting `p2p`:
* data parallelism using hyperplanes
* task parallelism using task dependencies
* SPMD parallelism using explicit synchronization

Here is an example (C++/OpenMP) of the hyperplane method,
where the connection to the sequential case is most
obvious when the hyperplane bandwidth is 1 (`nc==1`):
```c++
      if (nc==1) {
        for (int i=2; i<=2*n-2; i++) {
          #pragma omp for simd
          for (int j=std::max(2,i-n+2); j<=std::min(i,n); j++) {
            const int x = i-j+1;
            const int y = j-1;
            grid[x*n+y] = grid[(x-1)*n+y] + grid[x*n+(y-1)] - grid[(x-1)*n+(y-1)];
          
        }
      } else {
        for (int i=2; i<=2*(nb+1)-2; i++) {
          #pragma omp for
          for (int j=std::max(2,i-(nb+1)+2); j<=std::min(i,nb+1); j++) {
            const int ib = nc*(i-j+1-1)+1;
            const int jb = nc*(j-1-1)+1;
            sweep_tile(ib, std::min(n,ib+nc), jb, std::min(n,jb+nc), n, grid);
          }
        }
```

Here is an example (C++/OpenMP) using tasks with dependencies:
```c++
for (int i=1; i<m; i+=mc) {
    for (int j=1; j<n; j+=nc) {
        #pragma omp task depend(in:grid[(i-mc)*n+j:1],grid[i*n+(j-nc):1]) depend(out:grid[i*n+j:1])
        sweep_tile(i, std::min(m,i+mc), j, std::min(n,j+nc), n, grid);
    }
}
```

Here is an example (Fortran coarrays) using explicit synchronization:
```fortran
    do j=2,n
       if(me > 1) sync images(prev)
       do i=2,m_local
          grid(i,j) = grid(i-1,j) + grid(i,j-1) - grid(i-1,j-1)
       enddo
       if(me /= np) then
          grid(1,j)[next] = grid(m_local,j)
          sync images(next)
       endif
    enddo
```
