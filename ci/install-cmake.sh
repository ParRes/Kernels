#!/bin/sh

set -e
set -x

os=`uname`
CI_ROOT="$1"

case "$os" in
    Darwin)
        echo "Mac"
        brew upgrade cmake || brew install cmake || true
        #brew list cmake
        ;;

    Linux)
        echo "Linux"
        if [ ! -d "$CI_ROOT/cmake" ]; then
            mkdir -p $CI_ROOT/cmake
            # from source
            #wget --no-check-certificate -q https://cmake.org/files/v3.4/cmake-3.4.1.tar.gz
            #tar -C $CI_ROOT -xzf cmake-3.4.1.tar.gz
            #cd ~/cmake-3.4.1
            #mkdir build && cd build
            #cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$CI_ROOT/cmake
            #make -j4 && make install
            # from binary
            cd $CI_ROOT
            wget --no-check-certificate -q https://github.com/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2-Linux-x86_64.sh
            sh ./cmake-3.13.2-Linux-x86_64.sh --prefix=$CI_ROOT/cmake --skip-license --exclude-subdir
        else
            echo "CMake installed..."
            find $CI_ROOT/cmake -name cmake
        fi
        ;;
esac
