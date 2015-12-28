#!/bin/sh
# This configuration file was taken originally from the mpi4py project
# <http://mpi4py.scipy.org/>, and then modified for Julia

set -e
set -x

TRAVIS_ROOT="$1"

case "$CC" in
    gcc)
        if [ ! -d "$TRAVIS_ROOT/gupc" ]; then
            wget --no-check-certificate -q http://www.gccupc.org/gupc-5201-1/32-gupc-5-2-0-1-source-release/file
            mv file upc-5.2.0.1.src.tar.bz2
            tar -xjf upc-5.2.0.1.src.tar.bz2
            cd upc-5.2.0.1
            ./contrib/download_prerequisites
            mkdir build && cd build
            ../configure --disable-multilib --enable-languages=c,c++ --prefix=$TRAVIS_ROOT/gupc
            # Travis has problems with how much output the GCC build creates
            make -j4 &> /dev/null
            make install
        else
            echo "GCC UPC installed..."
            find $TRAVIS_ROOT/gupc -name gupc
            gupc --version
            gcc  --version
        fi
        ;;

    clang)
        echo "Clang UPC not supported yet..."
        exit 60
        if [ ! -d "$TRAVIS_ROOT/clupc" ]; then
            # get source files
            mkdir build && cd build
            ../configure --disable-multilib --enable-languages=c,c++ --prefix=$TRAVIS_ROOT/clupc
            make -j4
            make install
        else
            echo "GCC UPC installed..."
            find $TRAVIS_ROOT/clupc -name clang
            clang --version
        fi
        ;;
esac
