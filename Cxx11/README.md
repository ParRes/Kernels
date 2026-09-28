# Cxx11 CMake build

This directory has a standalone CMake project (`CMakeLists.txt`) alongside
the legacy hand-written `Makefile.legacy`. It can be built either as part
of the top-level PRK build (`add_subdirectory(Cxx11)`, gated by the root
`PRK_ENABLE_CXX11` option) or on its own:

```sh
cmake -S Cxx11 -B Cxx11/build
cmake --build Cxx11/build -j$(nproc)
ctest --test-dir Cxx11/build --output-on-failure
```

Missing optional dependencies (Kokkos, RAJA) are autobuilt from source via
`FetchContent`; everything else (MPI, BLAS, CUDA, NVSHMEM, TBB, OpenCL,
Boost.Compute, range-v3, ...) is detected, and a real compile/link probe
(not just "the library exists") gates any target group whose actual
correctness depends on toolchain specifics -- OpenMP target offload,
OpenACC, stdpar, CUDASTF/CCCL-vs-CUDA-version compatibility, PSTL-vs-TBB
linkage, and nvcc's extended-lambda support all work this way. Each
target group prints a `-- Cxx11 <group>: ...` status line explaining what
was found or why it was skipped.

## Picking a toolchain

By default CMake uses the system compiler (`cc`/`c++`, i.e. GCC on this
machine). To build with a different compiler, set `CMAKE_CXX_COMPILER`
at configure time -- CMake doesn't auto-discover alternate toolchains
like Intel oneAPI or NVIDIA HPC SDK on its own, since they aren't on
`PATH` by default and (for oneAPI) need their own environment variables
set up first.

### Intel oneAPI (`icpx`)

Source oneAPI's environment script first so `find_package(BLAS)`/`TBB`
resolve MKL and TBB correctly, then point CMake at `icpx`:

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S Cxx11 -B Cxx11/build-oneapi -DCMAKE_CXX_COMPILER=icpx
cmake --build Cxx11/build-oneapi -j$(nproc)
```

Under `icpx`, `find_package(BLAS)` resolves to MKL rather than OpenBLAS,
so the cblas targets need `MKL`'s own API (`mkl_dgemm`/`mkl_domatcopy`
instead of `cblas_dgemm_batch`/`cblas_domatcopy`) -- CMakeLists.txt
detects this from the resolved `BLAS_LIBRARIES` path and defines `MKL`
automatically. TBB/PSTL is also usable here, which unlocks the PSTL and
stdpar target groups (OpenACC is not available -- Intel never shipped a
C++ OpenACC implementation).

### NVIDIA HPC SDK (`nvc++`)

No environment sourcing needed, just put the compiler on `PATH` (or pass
its full path) and select it:

```sh
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/<version>/compilers/bin:$PATH
cmake -S Cxx11 -B Cxx11/build-nvhpc -DCMAKE_CXX_COMPILER=nvc++
cmake --build Cxx11/build-nvhpc -j$(nproc)
```

`nvc++` gets real GPU-offloaded OpenACC (`-acc -target=gpu`) and stdpar
(`-stdpar=gpu`), the only compiler here with a working implementation of
either. Note `nvc++` also defines `__GNUC__` for compatibility, which
trips up naive `#ifdef __GNUC__` compiler detection in a couple of source
files (matched against real GCC-only behavior) -- CMakeLists.txt checks
`CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC"` instead of relying on `__GNUC__`
wherever this distinction matters. The plain TBB-backed PSTL target group
is skipped under NVHPC specifically: its `<execution>` header predefines
a global `namespace exec = std::experimental::execution` (its own
pre-standardization PSTL, with no `par_unseq`) that collides with
`prk_pstl.h`'s own `namespace exec = std::execution` alias -- NVHPC's
real parallel story is stdpar/OpenACC, not this group.

### GNU (`g++`, the default)

No flags needed. OpenMP target offload and OpenACC are both detected as
*not working* on this machine even though `g++` accepts `-fopenmp
-foffload=...`/`-fopenacc` -- both probes fail for the same reason (a
broken `nvptx` accelerator offload toolchain: `mkoffload` rejects
`-fcf-protection=full` for the nvptx target). stdpar still works here via
TBB-backed PSTL (same as Intel).

## What each toolchain unlocks

| Group | GNU | Intel oneAPI (`icpx`) | NVIDIA HPC SDK (`nvc++`) |
|---|---|---|---|
| OpenMP target offload | broken toolchain | broken toolchain (Clang offload not probed) | broken toolchain |
| OpenACC | broken toolchain | not available | real GPU offload |
| stdpar (`nstream-stdpar`) | TBB-backed PSTL | TBB-backed PSTL | real GPU offload |
| stdpar (`transpose-stdpar`) | skipped (raw `__device__` lambda, nvcc/nvc++ only) | skipped | real GPU offload |
| TBB / PSTL (`*-pstl`) | yes (needs TBB) | yes (needs `USE_ONEAPI_DPL`) | skipped (see above) |
| OpenCL / Boost.Compute | not found | found | not found |
| cblas | OpenBLAS API | MKL API (`MKL` defined automatically) | OpenBLAS API |

CUDA/cuBLAS/Thrust/NVSHMEM/Kokkos/RAJA are independent of the host C++
compiler choice (CUDA device code is always compiled with `nvcc`) and
build the same way under all three.

Verified end-to-end (`cmake --build` + `ctest`) on this machine: GNU
95/95, Intel oneAPI `icpx` 106/106, NVIDIA HPC SDK `nvc++` 108/108 tests
passing. Test counts differ because each toolchain unlocks different
optional target groups.
