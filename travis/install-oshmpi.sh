#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

if [ ! -d "$TRAVIS_ROOT/oshmpi" ]; then
    git clone --depth 1 https://github.com/jeffhammond/oshmpi.git
    cd oshmpi
    ./autogen.sh
    ./configure CC=mpicc --prefix=$TRAVIS_ROOT/oshmpi
    make
    make install
else
    echo "OSHMPI installed..."
    find $TRAVIS_ROOT/oshmpi -name shmem.h
fi
