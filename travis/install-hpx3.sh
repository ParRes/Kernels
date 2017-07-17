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
        if [ "$USE_HPX_TARBALL" ] ; then
            export HPX_BOOST="homebrew/versions/boost155"
        else
            export HPX_BOOST="boost"
        fi
        for p in $HPX_BOOST jemalloc gperftools ; do
            brew install $p || brew upgrade $p
        done
        set -e
        ;;
esac

if [ ! -d "$TRAVIS_ROOT/hpx3" ]; then
    cd $TRAVIS_ROOT
    #if [ "$USE_HPX_TARBALL" ] ; then
    #    wget -q --no-check-certificate http://stellar.cct.lsu.edu/files/hpx_0.9.11.tar.bz2
    #    if [ `which md5` ] ; then
    #        echo "MD5 signature is:"
    #        md5 hpx_0.9.11.tar.bz2
    #        echo "MD5 signature should be:"
    #        echo "86a71189fb6344d27bf53d6aa2b33122"
    #    fi
    #    tar -xjf hpx_0.9.11.tar.bz2
    #    cd hpx_0.9.11
    #else
        git clone --depth 1 https://github.com/STEllAR-GROUP/hpx.git hpx3-source
        cd hpx3-source
    #fi
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$TRAVIS_ROOT/hpx3 -DCMAKE_MACOSX_RPATH=YES -DHPX_WITH_HWLOC=OFF
    make -j2
    # make check # target does not exist
    make install
else
    echo "HPX-3 installed..."
    find $TRAVIS_ROOT/hpx3
fi
