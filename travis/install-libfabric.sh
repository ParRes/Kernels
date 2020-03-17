#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

if [ ! -d "$TRAVIS_ROOT/libfabric" ]; then
    cd $TRAVIS_ROOT
    git clone --depth 1 https://github.com/ofiwg/libfabric.git libfabric-source
    #git clone -b 'v1.5.2' --depth 1 https://github.com/ofiwg/libfabric.git libfabric-source
    cd libfabric-source
    ./autogen.sh
    ./configure CC=cc --prefix=$TRAVIS_ROOT/libfabric
    make
    make install
    export FI_LOG_LEVEL=error
else
    echo "OFI/libfabric installed..."
    find $TRAVIS_ROOT -name "fi.h"
fi
