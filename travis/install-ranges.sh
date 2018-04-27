#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

git clone --depth 1 https://github.com/ericniebler/range-v3.git $TRAVIS_ROOT/range-v3
