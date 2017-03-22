#!/bin/sh

set -e
set -x

if [ -f ~/use-intel-compilers ] ; then
    export CC=icc
    export CXX=icpc
    export FC=ifort
fi

TRAVIS_ROOT="$1"
SHMEM_ROOT=$TRAVIS_ROOT/sandia-openshmem

if [ ! -d "$SHMEM_ROOT" ]; then
    # master
    #git clone --depth 10 https://github.com/regrant/sandia-shmem.git sandia-shmem
    #git clone --depth 10 https://github.com/Sandia-OpenSHMEM/SOS.git sandia-shmem
    #git clone -b v1.3.0-rc1 --depth 10 https://github.com/Sandia-OpenSHMEM/SOS.git sandia-shmem
    #cd sandia-shmem
    # 1.3.2 release
    wget https://github.com/Sandia-OpenSHMEM/SOS/archive/v1.3.2.tar.gz
    tar -xzf v1.3.2.tar.gz
    cd SOS-1.3.2
    ./autogen.sh
    mkdir build
    cd build
    ../configure --with-ofi=$TRAVIS_ROOT/libfabric --with-ofi-libdir=$TRAVIS_ROOT/libfabric/lib \
                --disable-fortran \
                --enable-error-checking \
                --enable-remote-virtual-addressing \
                --enable-pmi-simple --with-pmi=$TRAVIS_ROOT/hydra \
                --prefix=$SHMEM_ROOT
    make && make install
else
    echo "Sandia OpenSHMEM installed..."
    find $SHMEM_ROOT -name shmem.h
fi
