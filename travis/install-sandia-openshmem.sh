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
    #git clone --depth 1 https://github.com/regrant/sandia-shmem.git sandia-shmem
    #git clone --depth 1 https://github.com/Sandia-OpenSHMEM/SOS.git sandia-shmem
    #git clone -b v1.3.2 --depth 1 https://github.com/Sandia-OpenSHMEM/SOS.git sandia-shmem
    #cd sandia-shmem
    # 1.3 release
    wget https://github.com/Sandia-OpenSHMEM/SOS/archive/v1.3.2.tar.gz
    tar -xzf v1.3.2.tar.gz
    cd SOS-1.3.2
    ./autogen.sh
    mkdir build
    cd build
    # Removed # --with-pmi=$TRAVIS_ROOT/hydra per Jim
    ../configure --with-libfabric=$TRAVIS_ROOT/libfabric \
                 --disable-fortran \
                 --enable-error-checking \
                 --enable-remote-virtual-addressing \
                 --enable-pmi-simple \
                 --prefix=$SHMEM_ROOT
    make
    make check | true
    make install
else
    echo "Sandia OpenSHMEM installed..."
    find $SHMEM_ROOT -name shmem.h
fi
