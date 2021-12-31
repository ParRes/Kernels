# How to Install SYCL

## triSYCL

See https://github.com/triSYCL/triSYCL.  This is a header-only implementation, so you can use
any C++17 compiler (C++14 might be sufficient).  You need Boost, while OpenMP or TBB are optional
for threaded parallelism on the CPU.

## CodePlay ComputeCpp

See https://www.codeplay.com/products/computesuite/computecpp.

## Intel Data Parallel C++

This comes in two flavors.  You can compile the open-source version on GitHub and use `clang++ -fsycl`,
or you can install oneAPI and use the `dpcpp` driver, which is a wrapper around `clang++ -fsycl`.

### oneAPI Download

See https://software.intel.com/en-us/articles/installation-guide-for-intel-oneapi-toolkits.

### Linux packages

See https://software.intel.com/en-us/articles/oneapi-repo-instructions.

### Build from source

Install CMake 3.14 from source.  None of the package managers seem to have a new enough version.

Install Ninja from package manager or https://ninja-build.org/.

Install GCC 9 (something newer than GCC 4 is required, but why not install something reasonable?).

If you are on a system with limited memory or multiple users, replace `nproc` with something nicer.
```sh
git clean -dfx
export CC=gcc-9
export CXX=g++-9
python3 ./buildbot/configure.py [--arm] [--cuda] [--cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.0"]
python3 ./buildbot/compile.py -j`nproc`
python3 ./buildbot/check.py
```

The optional arguments are associated with platforms such as the Xavier AGX, as shown below.
```sh
python3 ./buildbot/configure.py --arm --cuda --cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.0"
```

It may also be necessary to apply the following (evil) `CPATH` hack on ARM systems.
```
export CPATH=/usr/include/aarch64-linux-gnu:$CPATH
```

## hipSYCL

See https://github.com/illuhad/hipSYCL/tree/master/doc for other options.

### Spack

https://github.com/spack/spack/pull/14051 is not merged yet but this works if you grab the PR.

```sh
./bin/spack install  hipsycl +cuda
```

### Celerity

This is what I did to install on CentOS 7...

Dependencies (may be incomplete):
```sh
yum install cmake3.x86_64 cmake3-doc.noarch
yum install hipSYCL.x86_64 hipSYCL-base.x86_64
yum install mpich-3.2.x86_64 mpich-3.2-devel.x86_64 mpich-3.2-doc.noarch \
            mpich-3.2-autoload.x86_64
yum install boost169.x86_64 boost169-jam.x86_64 boost169-build.noarch \
            boost169-devel.x86_64 boost169-mpich-devel.x86_64 boost169-static.x86_64
```

The documentation is wrong about how you specify where hipSYCL lives.  Below works.

Use the hipSYCL Clang C/C++ compilers for consistency.
```sh
cmake3 .. -DBOOST_INCLUDEDIR=/usr/include/boost169/ \
          -DCMAKE_PREFIX_PATH=/opt/hipSYCL/lib -DHIPSYCL_PLATFORM=cpu \
          -DCMAKE_CXX_COMPILER=/opt/hipSYCL/llvm/bin/clang++ \
          -DCMAKE_C_COMPILER=/opt/hipSYCL/llvm/bin/clang
```

In order for the tests to run properly, one has to specify the location of the LLVM OpenMP runtime, e.g. as follows:
```sh
export LD_LIBRARY_PATH=/opt/hipSYCL/llvm/lib:$LD_LIBRARY_PATH
```
