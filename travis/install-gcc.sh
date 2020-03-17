#!/bin/sh

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

if [ "${CC}" = "gcc" ] || [ "${CXX}" = "g++" ] ; then
    case "$os" in
        Darwin)
            echo "Mac"
            # this is 5.3.0 or later
            brew upgrade gcc || brew install gcc --force-bottle || true
            ;;
        Linux)
            echo "Linux"
            sudo apt-get install gcc g++ gfortran
        ;;
    esac
fi
