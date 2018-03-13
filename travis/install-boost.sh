#!/bin/sh

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

case "$os" in
    Darwin)
        brew update
        brew install boost || brew upgrade boost || true
        ;;

    Linux)
        # Boost.Compute is a header-only library
        #git clone --depth 1 https://github.com/kylelutz/compute.git ${TRAVIS_ROOT}/compute
        git clone --depth 1 https://github.com/boostorg/compute.git ${TRAVIS_ROOT}/compute
        ;;
esac
