#!/bin/sh

set -e
set -x

CI_ROOT="$1"

if [ ! -d "$CI_ROOT/ga" ]; then
    git clone -b develop https://github.com/GlobalArrays/ga.git $CI_ROOT/ga-src
    cd $CI_ROOT/ga-src
    ./autogen.sh
    mkdir build
    cd build
    #../configure CC=mpicc --prefix=$CI_ROOT/ga
    #../configure --with-mpi3 MPICC=mpiicc MPICXX=mpiicpc MPIFC=mpiifort MPIF77=mpiifort --prefix=$CI_ROOT/ga && make -j8 install
    ../configure --with-armci=${CI_ROOT}/armci-mpi MPICC=mpiicc MPICXX=mpiicpc MPIFC=mpiifort MPIF77=mpiifort --prefix=$CI_ROOT/ga && make -j8 install
    make
    make install
else
    echo "Global Arrays installed..."
    find $CI_ROOT/ga -name ga.h
fi
