#!/bin/sh

set -e
set -x

CI_ROOT="$1"

if [ ! -d "$CI_ROOT/oshmpi" ]; then
    git clone --depth 1 https://github.com/jeffhammond/oshmpi.git
    cd oshmpi
    ./autogen.sh
    ./configure CC=mpicc --prefix=$CI_ROOT/oshmpi
    make
    make install
else
    echo "OSHMPI installed..."
    find $CI_ROOT/oshmpi -name shmem.h
fi
