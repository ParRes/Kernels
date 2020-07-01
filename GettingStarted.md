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

