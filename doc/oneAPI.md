# oneMKL

## Intel

TODO

## ARM

### Build LAPACK

```
wget https://github.com/Reference-LAPACK/lapack/archive/v3.9.0.tar.gz
tar -xaf v3.9.0.tar.gz
cd lapack-v3.9.0
cp ./INSTALL/make.inc.gfortran make.inc
make -k
make -C CBLAS
```

### Build oneMKL
```
ubuntu@ubuntu:~/oneMKL/build$ git clean -dfx ; CXX=${HOME}/ISYCL/llvm/build/install/bin/clang++ cmake .. -DENABLE_NETLIB_BACKEND=1 -DENABLE_MKLCPU_BACKEND=0  -DENABLE_MKLGPU_BACKEND=0 -DCBLAS_LIB_DIR=${HOME}/lapack-3.9.0 -DNETLIB_CBLAS_LIBRARY=${HOME}/lapack-3.9.0/libcblas.a -DCBLAS_file=${HOME}/lapack-3.9.0/CBLAS/include/cblas.h
warning: failed to remove ./: Invalid argument
Removing ./oneMKLTargets.cmake
Removing ./cmake_install.cmake
Removing ./tests
Removing ./cmake
Removing ./CMakeFiles
Removing ./CMakeCache.txt
Removing ./Makefile
Removing ./oneMKLConfigVersion.cmake
Removing ./compile_commands.json
Removing ./CTestTestfile.cmake
Removing ./lib
Removing ./oneMKLConfig.cmake
Removing ./deps
Removing ./bin
-- CMAKE_BUILD_TYPE: None, set to Release by default
-- The CXX compiler identification is Clang 12.0.0
-- Check for working CXX compiler: /home/ubuntu/ISYCL/llvm/build/install/bin/clang++
-- Check for working CXX compiler: /home/ubuntu/ISYCL/llvm/build/install/bin/clang++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Performing Test is_dpcpp
-- Performing Test is_dpcpp - Success
-- Found SYCL:   
-- Found NETLIB: /home/ubuntu/lapack-3.9.0/libcblas.a  
-- Found NETLIB: /usr/lib/aarch64-linux-gnu/libblas.so  
-- Found NETLIB: /usr/include/aarch64-linux-gnu  
-- The C compiler identification is Clang 12.0.0
-- Check for working C compiler: /home/ubuntu/ISYCL/llvm/build/install/bin/clang
-- Check for working C compiler: /home/ubuntu/ISYCL/llvm/build/install/bin/clang -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Found PythonInterp: /usr/bin/python3.8 (found version "3.8.5") 
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE  
-- Found CBLAS: /home/ubuntu/lapack-3.9.0/CBLAS/include/cblas.h  
-- Found CBLAS: /usr/lib/aarch64-linux-gnu/libblas.so  
-- Found CBLAS: /usr/include/aarch64-linux-gnu  
-- Configuring done
-- Generating done
-- Build files have been written to: /home/ubuntu/oneMKL/build
```

## NVIDIA
