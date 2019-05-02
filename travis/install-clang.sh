#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

if [ "${CC}" = "clang" ] || [ "${CXX}" = "clang++" ] ; then
    os=`uname`
    case "$os" in
        Darwin)
            echo "Mac"
            brew update
            brew install llvm || brew upgrade llvm || true
            brew install libomp || brew upgrade libomp || true
            ;;
        Linux)
            echo "Linux Clang/LLVM builds not supported!"
            exit 18
        ;;
    esac
fi
