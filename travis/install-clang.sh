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
            for v in "-11" "-10" "-9" "-8" "-7" ; do
                sudo apt-get install clang$v && sudo apt-get install libomp$v-dev
                if [ -f /usr/lib/llvm$v/bin/clang-$v ] && [ -f /usr/lib/llvm$v/lib/libomp.so ] ; then
                    /usr/lib/llvm$v/bin/clang-$ -v
                    break
                fi
            done
            set -e
        ;;
    esac
fi
