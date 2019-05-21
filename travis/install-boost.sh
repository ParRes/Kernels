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
        for sp in  circular_buffer compute config core log array multi_array optional \
                   preprocessor type_index utility assert static_assert exception throw_exception \
                   concept_check type_traits iterator mpl detail functional move range ; do
            git clone --depth 1 https://github.com/boostorg/${sp}.git ${TRAVIS_ROOT}/${sp}
        done
        ;;
esac

