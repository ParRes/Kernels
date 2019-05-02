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
    # HEAD
    #git clone --depth 1 https://github.com/Sandia-OpenSHMEM/SOS.git sandia-shmem
    #cd sandia-shmem
    VERSION=1.4.2
    #git clone -b v$VERSION --depth 1 https://github.com/Sandia-OpenSHMEM/SOS.git SOS-$VERSION
    wget https://github.com/Sandia-OpenSHMEM/SOS/archive/v$VERSION.tar.gz
    tar -xzf v$VERSION.tar.gz
    cd SOS-$VERSION
    ./autogen.sh
    mkdir build
    cd build
    # Removed # --with-pmi=$TRAVIS_ROOT/hydra per Jim
    ../configure --with-libfabric=$TRAVIS_ROOT/libfabric \
                 --disable-fortran \
                 --enable-error-checking \
                 --enable-pmi-simple \
                 --prefix=$SHMEM_ROOT
                 #--enable-remote-virtual-addressing \
    make
    make check | true
    make install
else
    echo "Sandia OpenSHMEM installed..."
    find $SHMEM_ROOT -name shmem.h
fi
