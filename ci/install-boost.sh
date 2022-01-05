#!/bin/sh

set -e
set -x

os=`uname`
CI_ROOT="$1"

case "$os" in
    Darwin)
        brew install boost || brew upgrade boost || true
        ;;

    Linux)
        sudo apt-get install libboost-all-dev
        # We do not test Boost.Compute on Linux because of OpenCL issues...
        # Boost.Compute is a header-only library
        #git clone --depth 1 https://github.com/kylelutz/compute.git ${CI_ROOT}/compute
        #git clone --depth 1 https://github.com/boostorg/compute.git ${CI_ROOT}/compute
        ;;
esac
