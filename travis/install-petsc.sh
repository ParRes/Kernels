#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

if [ ! -f "$TRAVIS_ROOT/petsc/include/petsc.h" ]; then
    if [ -d "$TRAVIS_ROOT/petsc-src" ]; then
        cd $TRAVIS_ROOT/petsc-src
        git pull
    else
        git clone -b maint https://gitlab.com/petsc/petsc.git $TRAVIS_ROOT/petsc-src
    fi
    cd $TRAVIS_ROOT/petsc-src
    ./configure --prefix=$TRAVIS_ROOT/petsc \
                --with-blaslapack-dir=$MKLROOT \
                --with-mpi-dir=$I_MPI_ROOT \
                --with-cxx=0 --with-fc=0
    make PETSC_DIR=$TRAVIS_ROOT/petsc-src PETSC_ARCH=arch-linux-c-debug all
    make PETSC_DIR=$TRAVIS_ROOT/petsc-src PETSC_ARCH=arch-linux-c-debug install
else
    echo "PETSc installed..."
    cat $TRAVIS_ROOT/petsc/lib/petsc/conf/reconfigure*.py
fi
