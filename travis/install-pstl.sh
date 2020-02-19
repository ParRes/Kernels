#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

git clone --depth 1 https://github.com/intel/parallelstl.git $TRAVIS_ROOT/pstl
