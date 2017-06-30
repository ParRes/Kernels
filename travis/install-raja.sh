#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

if [ ! -d "$TRAVIS_ROOT/raja" ]; then
    BRANCH=bugfix/jeffhammond/abort-and-getenv
    git clone --depth 10 -b ${BRANCH} https://github.com/LLNL/RAJA.git
    cd RAJA
    mkdir build
    cd build
    cmake .. -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_C_COMPILER=${CC} \
             -DCMAKE_INSTALL_PREFIX=${TRAVIS_ROOT}/raja \
             -DRAJA_ENABLE_OPENMP=On
    make -j2
    make install -j2
else
    echo "RAJA installed..."
    find $TRAVIS_ROOT/raja -name RAJA.hxx
fi
