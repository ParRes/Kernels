#!/bin/sh

set -e
set -x

CI_ROOT="$1"

git clone --depth 1 https://github.com/facebookexperimental/libunifex.git $CI_ROOT/libunifex
pushd  $CI_ROOT/libunifex
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_FLAGS="-std=c++20"
