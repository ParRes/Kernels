#!/bin/sh

set -e
set -x

if [ -f ~/use-intel-compilers ] ; then
    export CC=icc
    export CXX=icpc
    export FC=ifort
fi

TRAVIS_ROOT="$1"

UPCXX_RELEASE=upcxx-2019.9.0
UPCXX_PREFIX=$TRAVIS_ROOT/$UPCXX_RELEASE

if [ ! -d "$UPCXX_PREFIX" ]; then
    cd $TRAVIS_ROOT
    wget --no-check-certificate -q https://bitbucket.org/berkeleylab/upcxx/downloads/${UPCXX_RELEASE}.tar.gz
    tar -xzf $UPCXX_RELEASE.tar.gz
    cd $UPCXX_RELEASE
    ./install $TRAVIS_ROOT/upcxx
else
    echo "UPC++ installed..."
    find $TRAVIS_ROOT/upcxx -name upcxx -type f
fi

