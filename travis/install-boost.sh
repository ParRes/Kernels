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
        # We do not test Boost.Compute on Linux because of OpenCL issues...
        # Boost.Compute is a header-only library
        #git clone --depth 1 https://github.com/kylelutz/compute.git ${TRAVIS_ROOT}/compute
        for sp in core config multi_array optional log compute preprocessor circular_buffer type_index utility ; do
            git clone --depth 1 https://github.com/boostorg/${sp}.git ${TRAVIS_ROOT}/${sp}
        done
        ;;
esac
