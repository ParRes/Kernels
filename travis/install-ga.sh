#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

if [ ! -d "$TRAVIS_ROOT/ga" ]; then
    git clone -b develop https://github.com/GlobalArrays/ga.git $TRAVIS_ROOT/ga-src
    cd $TRAVIS_ROOT/ga-src
    ./autogen.sh
    mkdir build
    cd build
    #../configure CC=mpicc --prefix=$TRAVIS_ROOT/ga
    #../configure --with-mpi3 MPICC=mpiicc MPICXX=mpiicpc MPIFC=mpiifort MPIF77=mpiifort --prefix=$TRAVIS_ROOT/ga && make -j8 install
    ../configure --with-armci=${TRAVIS_ROOT}/armci-mpi MPICC=mpiicc MPICXX=mpiicpc MPIFC=mpiifort MPIF77=mpiifort --prefix=$TRAVIS_ROOT/ga && make -j8 install
    make
    make install
else
    echo "Global Arrays installed..."
    find $TRAVIS_ROOT/ga -name ga.h
fi
