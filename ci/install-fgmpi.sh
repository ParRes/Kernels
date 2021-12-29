#!/bin/sh

set -e
set -x

CI_ROOT="$1"

if [ ! -d "$CI_ROOT/fgmpi" ]; then

    # TAR build
    wget --no-check-certificate -q http://www.cs.ubc.ca/~humaira/code/fgmpi-2.0.tar.gz
    tar -C $CI_ROOT -xzf fgmpi-2.0.tar.gz
    cd $CI_ROOT/fgmpi-2.0

    # GIT build
    #cd $CI_ROOT
    #git clone --depth 1 https://github.com/humairakamal/fgmpi.git fgmpi-source
    #cd fgmpi-source
    ## this may fail on older autotools
    #./autogen.sh

    # TAR or GIT
    mkdir build && cd build
    # Clang defaults to C99, which chokes on "Set_PROC_NULL"
    ../configure --disable-fortran --disable-romio CFLAGS="-std=gnu89 -w" --prefix=$CI_ROOT/fgmpi
    make -j2
    make install

    # Package install
    # TODO (restore from older version but unpack in $CI_ROOT without sudo)

else
    echo "FG-MPI installed..."
    find $CI_ROOT/fgmpi -name mpiexec
    find $CI_ROOT/fgmpi -name mpicc
    mpicc -show
fi
