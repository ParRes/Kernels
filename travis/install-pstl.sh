#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

git clone https://github.com/intel/parallelstl.git $TRAVIS_ROOT/pstl
