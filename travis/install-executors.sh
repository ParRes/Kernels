#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

git clone --depth 1 https://github.com/facebookexperimental/libunifex.git $TRAVIS_ROOT/libunifex
pushd  $TRAVIS_ROOT/libunifex
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_FLAGS="-std=c++20"
