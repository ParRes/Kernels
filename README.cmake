# CMake build system

PRK has always built through hand-written, per-language Makefiles driven
by `common/make.defs.*` toolchain variant files. This CMake build is a
newer, parallel way to build the same kernels, currently covering
`FORTRAN/`, `C1z/`, and `Cxx11/` (the other language directories still
build only via their own Makefiles). It does not replace the legacy
Makefiles -- both live side by side in the same source tree, and neither
build clobbers the other's output.

## Quick start

```sh
# Build and test everything (all enabled languages) in one combined tree
cmake -S . -B build
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure

# Or build one language standalone
cmake -S Cxx11 -B Cxx11/build
cmake --build Cxx11/build -j$(nproc)
ctest --test-dir Cxx11/build --output-on-failure

# In-source also works (cmake . inside a language directory), see below
cd FORTRAN && cmake . && cmake --build . -j$(nproc)
```

## How the pieces fit together

### Root `CMakeLists.txt` is a thin wrapper

It declares `project(PRK NONE)` (no default language -- building only
FORTRAN shouldn't require a C++ compiler) and `add_subdirectory()`s
whichever languages are enabled via `option(PRK_ENABLE_<LANG> ... ON)`.
That's it. All the real work happens in each language's own
`CMakeLists.txt`.

### Each language is its own standalone CMake project

`FORTRAN/CMakeLists.txt`, `C1z/CMakeLists.txt`, and `Cxx11/CMakeLists.txt`
each have their own `cmake_minimum_required()`, `project()`, and
`enable_testing()`. This means:

- `cd Cxx11 && cmake -S . -B build` works on its own -- you don't need the
  rest of the repo, and you don't need to build languages you don't care
  about.
- They're also each `add_subdirectory()`'d from the root, so a single
  `cmake -S . -B build` at the repo root configures and builds all of
  them together in one tree, with one combined `ctest` run.
- Adding a new language later follows the same pattern: its own
  `CMakeLists.txt` structured like the existing ones, plus one
  `option(PRK_ENABLE_<LANG> ... ON)` + `add_subdirectory()` pair in the
  root file.

### In-source builds are supported

Each language directory still has its original hand-written Makefile --
renamed to `Makefile.legacy` specifically so that `cmake .` (an in-source
build) doesn't overwrite it. CMake's own generated `Makefile` and the
legacy one coexist; use `-f Makefile.legacy` with `make` to reach the old
one. (Prefer an out-of-source build, `cmake -S . -B build`, when you can
-- it's cleaner and some autobuilt dependencies below require it. In-source
builds still work for everything else.)

### Missing optional dependencies get built from source, not skipped

Where a make.defs-managed dependency (Global Arrays, OpenCoarrays, PETSc,
Kokkos, RAJA) isn't found on the system, `cmake/AutobuildDeps.cmake`
fetches and builds it automatically as part of the normal build --
`prk_autobuild_global_arrays()`, `prk_autobuild_opencoarrays()`,
`prk_autobuild_petsc()` (all via `ExternalProject_Add`, since they need a
real install step), and `prk_autobuild_kokkos()`/`prk_autobuild_raja()`
(via `FetchContent_MakeAvailable()`, since they're CMake-native and don't
need a separate install). Controlled by `option(PRK_AUTOBUILD_DEPS ON)`
if you'd rather it just skip cleanly instead.

Kokkos and RAJA's own build systems unconditionally refuse to configure
in-source (checked against the *top-level* `CMAKE_SOURCE_DIR`/
`CMAKE_BINARY_DIR`, which every `add_subdirectory()`/`FetchContent` call
shares -- there's no per-subdirectory override). `prk_autobuild_kokkos()`/
`prk_autobuild_raja()` detect this and skip cleanly with an actionable
message rather than let Kokkos's own confusing `FATAL_ERROR` surface --
use an out-of-source build (`cmake -S Cxx11 -B Cxx11/build`) if you need
them.

### Dependency and feature detection uses real probes, not assumptions

Beyond ordinary `find_package()`/`find_library()`, each language has its
own `cmake/PRK<Lang>Features.cmake` (`PRKFortranFeatures.cmake`,
`PRKCFeatures.cmake`, `PRKCxxFeatures.cmake`) with `try_compile()`/
`try_run()`-based probes for things that aren't just "is the library
there" questions: does this compiler's OpenMP target-offload actually
compile *and run* correctly (not just accept the flag), does the CUDASTF
checkout actually compile against this CUDA version, does the host
compiler understand nvcc's extended-lambda syntax, does `std::ranges`
actually support `views::cartesian_product` here or only `views::iota`.
A compiler accepting a flag doesn't mean the feature works -- this
project found several real, silent breakages (broken offload toolchains,
version-mismatched headers, wrong-CPU-kernel-detection bugs) that a
naive "flag accepted" check would have missed, so prefer a real probe
over an assumption whenever the two might disagree.

### Toolchain selection

By default CMake uses the system compiler. To use a different one:

```sh
cmake -S Cxx11 -B build -DCMAKE_CXX_COMPILER=icpx
# or, equivalently and more reproducibly:
cmake -S Cxx11 -B build -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/oneapi.cmake
```

`cmake/toolchains/` has one file per `common/make.defs.<name>` example
(`gcc.cmake`, `llvm.cmake`, `nvhpc.cmake`, `cuda.cmake`, `oneapi.cmake`,
`intel.cmake`, `arm.cmake`, `armgcc.cmake`, `boost.cmake`, `cray.cmake`,
`freebsd.cmake`, `hip.cmake`, `ibmbg.cmake`, `ibmp9nv.cmake`,
`musl.cmake`, `pgi.cmake`, `upcxx-hpx.cmake`), each mirroring that
make.defs' `CC=`/`CXX=`/`FC=` compiler selection. They deliberately don't
also copy over the `-std=`/`-pthread`/etc. flags from make.defs -- each
language's `CMakeLists.txt` already manages its own C++/Fortran standard
selection via `CMAKE_CXX_STANDARD` and the real feature probes above,
which is more portable than hardcoding one make.defs example's specific
`-std=` flag (and avoids a "two competing `-std=` flags, last one
silently wins" footgun this project hit once already).

`gcc.cmake`, `llvm.cmake`, `nvhpc.cmake`, and `oneapi.cmake` are verified
against the toolchains actually available on the machine this was
developed on (see "Verification" below); the rest mirror their
make.defs.* as faithfully as reasonable but aren't build-tested here (no
matching compiler installed) -- treat them as a documented starting point
for a machine that does have that toolchain, not a guarantee.

`oneapi.cmake` additionally sets `MKLROOT`/`TBBROOT`/`LD_LIBRARY_PATH`/
`I_MPI_ROOT`/`PATH`/`LIBRARY_PATH` so it works without sourcing Intel's
`setvars.sh` first. One real limitation to know about: a toolchain file's
`set(ENV{...})` only takes effect during the *configure* step -- it does
**not** carry over to the separate `cmake --build`/`ctest` invocations
that follow. Anything that needs those variables at build or run time
(not just configure-time detection) still needs them set in your actual
shell, e.g. by sourcing `setvars.sh` yourself before building/testing.

### Testing

Every registered test gets a 30-second `TIMEOUT` (`prk_fortran_test()`/
`prk_cxx_test()` helpers set this automatically) -- a hung or infinite-
looping kernel must not be able to block the rest of the suite
indefinitely. A handful of tests are wired in and known to currently fail
(e.g. `cxx11_nvshmem_transpose_put`, `cxx11_opencl_p2p_innerloop`), each
with a comment at its `prk_cxx_test()` call site explaining the specific
root cause -- they're tracked in CTest deliberately rather than silently
omitted, so a regression search doesn't have to first rediscover that
they were already broken.

## Verification

The CMake port's produced binaries were compared against the legacy
`Makefile.legacy` build's default `all` target, language by language, for
every toolchain with a matching compiler on the development machine (gcc,
llvm, nvhpc, oneapi) -- see `git log` for the commits from that pass (it
found and fixed eight real bugs: several generator-expression/flag-
concatenation issues, missing compiler-specific flags, missing include
paths, and one CMake-probe convention bug). CMake's target list is a
strict superset of each toolchain's legacy `all` target in every case
verified, since this port also covers several groups (CUDA/NVSHMEM/NCCL/
Kokkos/RAJA/TBB/PSTL/stdpar/OpenACC/OpenCL/Boost.Compute and friends) that
the legacy Makefiles gate behind an explicit `make <group>` rather than
building by default.
