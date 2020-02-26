#!/bin/sh

set -e
set -x

if [ -f ~/use-intel-compilers ] ; then
    export CC=icc
    export CXX=icpc
    export FC=ifort
fi

TRAVIS_ROOT="$1"

case "$TRAVIS_OS_NAME" in
    linux)
        ;;
    osx)
        set +e
        brew update
        for p in boost jemalloc gperftools ; do
            brew install $p || brew upgrade $p
        done
        set -e
        ;;
esac

if [ ! -d "$TRAVIS_ROOT/hpx" ]; then
    cd $TRAVIS_ROOT
    git clone --depth 1 https://github.com/STEllAR-GROUP/hpx.git hpx-source
    cd hpx-source
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$TRAVIS_ROOT/hpx -DCMAKE_MACOSX_RPATH=YES -DHPX_WITH_HWLOC=OFF
    make -j2
    # make check # target does not exist
    make install
else
    echo "HPX installed..."
    find $TRAVIS_ROOT/hpx
fi
