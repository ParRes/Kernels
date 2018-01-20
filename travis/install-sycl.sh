#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

git clone --depth 1 https://github.com/triSYCL/triSYCL.git $TRAVIS_ROOT/triSYCL
