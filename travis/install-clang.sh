#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

if [ "${CC}" = "clang" ] || [ "${CXX}" = "clang++" ] ; then
    os=`uname`
    case "$os" in
        Darwin)
            echo "Mac"
            brew install llvm || brew upgrade llvm || true
            #brew install libomp || brew upgrade libomp || true
            ;;
        Linux)
            echo "Linux Clang/LLVM builds not supported!"
            set +e
            for v in "-11" "-10" "-9" "-8" "-7" "-6.0" "-5.0" "-4.0" ; do
                sudo apt-get install clang$v
            done
            set -e
        ;;
    esac
fi
