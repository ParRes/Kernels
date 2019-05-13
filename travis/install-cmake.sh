#!/bin/sh

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

case "$os" in
    Darwin)
        echo "Mac"
        brew update || true
        brew upgrade cmake || brew install cmake || true
        #brew list cmake
        ;;

    Linux)
        echo "Linux"
        if [ ! -d "$TRAVIS_ROOT/cmake" ]; then
            mkdir -p $TRAVIS_ROOT/cmake
            cd $TRAVIS_ROOT
            wget --no-check-certificate -q https://github.com/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2-Linux-x86_64.sh
            sh ./cmake-3.13.2-Linux-x86_64.sh --prefix=$TRAVIS_ROOT/cmake --skip-license --exclude-subdir
        else
            echo "CMake installed..."
            find $TRAVIS_ROOT/cmake -name cmake
        fi
        ;;
esac
