# Halide

# Notes

```
$ git clone https://github.com/halide/Halide.git
```

# MacOS

This works:
```
make CLANG=/usr/local/Cellar/llvm/8.0.0/bin/clang PREFIX=/opt/halide LLVM_CONFIG=/usr/local/Cellar/llvm/8.0.0/bin/llvm-config
```

# Ubuntu 18.10

This works:
```
make PREFIX=/opt/halide
```

This does not work:

```
$ make CXX=clang++ PREFIX=/opt/halide LLVM_CONFIG=/usr/local/Cellar/llvm/8.0.0/bin/llvm-config
```

This does not work:

```
$ make CC=/usr/local/Cellar/llvm/8.0.0/bin/clang CXX=/usr/local/Cellar/llvm/8.0.0/bin/clang++ CLANG=/usr/local/Cellar/llvm/8.0.0/bin/clang PREFIX=/opt/halide LLVM_CONFIG=/usr/local/Cellar/llvm/8.0.0/bin/llvm-config
```

# Issues

*TL;DR* Do not try to use non-default compilers.

https://github.com/halide/Halide/issues/3884

Mac:
```
$ make CC=gcc-9 CXX=g++-9 CLANG=/usr/local/Cellar/llvm/8.0.0/bin/clang PREFIX=/opt/halide LLVM_CONFIG=/usr/local/Cellar/llvm/8.0.0/bin/llvm-config
g++-9 -Wall -Werror -Wno-unused-function -Wcast-qual -Wignored-qualifiers -Wno-comment -Wsign-compare -Wno-unknown-warning-option -Wno-psabi -Wsuggest-override   -Woverloaded-virtual -fPIC -O3 -fno-omit-frame-pointer -DCOMPILING_HALIDE -std=c++11  -I/usr/local/Cellar/llvm/8.0.0/include -std=c++11 -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -I/tmp/llvm-20190320-85215-19esl1h/llvm-8.0.0.src/tools/lld/include -DLLVM_VERSION=80  -DWITH_PTX=1  -DWITH_ARM=1  -DWITH_HEXAGON=1  -DWITH_AARCH64=1  -DWITH_X86=1  -DWITH_OPENCL=1  -DWITH_METAL=1  -DWITH_OPENGL=1  -DWITH_D3D12=1  -DWITH_MIPS=1  -DWITH_POWERPC=1  -DWITH_WEBASSEMBLY=1  -DWITH_INTROSPECTION    -DWITH_AMDGPU=1     -funwind-tables -c ~/Work/Languages/Halide/src/Util.cpp -o bin/build/Util.o -MMD -MP -MF bin/build/Util.d -MT bin/build/Util.o
~/Work/Languages/Halide/src/Util.cpp: In function 'std::string Halide::Internal::running_program_name()':
~/Work/Languages/Halide/src/Util.cpp:80:19: error: 'PATH_MAX' was not declared in this scope
   80 |         char path[PATH_MAX] = { 0 };
      |                   ^~~~~~~~
~/Work/Languages/Halide/src/Util.cpp:81:32: error: 'path' was not declared in this scope
   81 |         uint32_t size = sizeof(path);
      |                                ^~~~
At global scope:
cc1plus: error: unrecognized command line option '-Wno-unknown-warning-option' [-Werror]
cc1plus: all warnings being treated as errors
make: *** [bin/build/Util.o] Error 1
```
