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

See https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedWithSYCLCompiler.md for details.

The following is my automation once the repo is cloned.

```sh
#!/bin/bash

export SYCL_HOME=$HOME/ISYCL

#cd $SYCL_HOME/llvm && time git checkout usmapi && time git pull
cd $SYCL_HOME/llvm && time git checkout sycl && time git pull

rm -rf $SYCL_HOME/build

mkdir -p $SYCL_HOME/build && \
    cd $SYCL_HOME/build && \
    time cmake \
        -DCMAKE_INSTALL_PREFIX=/opt/isycl \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS="clang;llvm-spirv;sycl" \
        -DLLVM_EXTERNAL_PROJECTS="llvm-spirv;sycl" \
        -DLLVM_EXTERNAL_SYCL_SOURCE_DIR=$SYCL_HOME/llvm/sycl \
        -DLLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR=$SYCL_HOME/llvm/llvm-spirv \
        -DLLVM_TOOL_SYCL_BUILD=ON \
        -DLLVM_TOOL_LLVM_SPIRV_BUILD=ON \
        $SYCL_HOME/llvm/llvm && \

time make -j4 sycl-toolchain

time make -j4 sycl-toolchain install #DESTDIR=/opt/isycl
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
