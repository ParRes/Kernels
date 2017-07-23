#!/bin/sh

set -e
set -x

if [ -f ~/use-intel-compilers ] ; then
    export CC=icc
    export CXX=icpc
    export FC=ifort
fi

TRAVIS_ROOT="$1"

echo "compiler versions:"
$CC --version
$CXX --version

if [ ! -d "$TRAVIS_ROOT/legion" ]; then
    cd $TRAVIS_ROOT
    git clone -b master --depth 1 https://github.com/StanfordLegion/legion.git
else
    echo "Legion present..."
    find $TRAVIS_ROOT/legion
fi
