#!/bin/sh

set -e
set -x

CI_ROOT="$1"

if [ ! -f "$CI_ROOT/petsc/include/petsc.h" ]; then
    if [ -d "$CI_ROOT/petsc-src" ]; then
        cd $CI_ROOT/petsc-src
        git pull
    else
        git clone -b maint https://gitlab.com/petsc/petsc.git $CI_ROOT/petsc-src
    fi
    cd $CI_ROOT/petsc-src
    ./configure --prefix=$CI_ROOT/petsc \
                --with-blaslapack-dir=$MKLROOT \
                --with-mpi-dir=$I_MPI_ROOT \
                --with-cxx=0 --with-fc=0
    make PETSC_DIR=$CI_ROOT/petsc-src PETSC_ARCH=arch-linux-c-debug all
    make PETSC_DIR=$CI_ROOT/petsc-src PETSC_ARCH=arch-linux-c-debug install
else
    echo "PETSc installed..."
    cat $CI_ROOT/petsc/lib/petsc/conf/reconfigure*.py
fi
