#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"
CLANG_VERSION="$2"

if [ "${CC}" = "clang" ] || [ "${CXX}" = "clang++" ] ; then
    os=`uname`
    case "$os" in
        Darwin)
            echo "Mac"
            brew update
            case "$CLANG_VERSION" in
                omp)
                    brew install clang-omp
                    brew test clang-omp
                    # make sure that these are found before the system installation
                    # there are less evil but less local ways to impart this effect
                    if [ ! -d "$TRAVIS_ROOT/bin" ]; then
                        mkdir -p $TRAVIS_ROOT/bin
                    fi
                    # we should refer to clang-omp* explicitly so know it exists and works
                    ln -s `which clang-omp`   $TRAVIS_ROOT/bin/clang
                    ln -s `which clang-omp++` $TRAVIS_ROOT/bin/clang++
                    ;;
                3*)
                    #brew install llvm$CLANG_VERSION --with-clang --with-compiler-rt --with-libcxx --with-lld --without-assertions
                    brew install llvm@$CLANG_VERSION
                    brew test llvm@$CLANG_VERSION
                    ;;
                *)
                    echo "Unsupported version of Clang"
                    echo "Travis will continue and use the system default"
                    ;;
            esac
            ;;
        Linux)
            echo "Linux Clang/LLVM builds not supported!"
            exit 18
        ;;
    esac
fi
