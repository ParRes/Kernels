#!/bin/sh

set -e
set -x

if [ -f ~/use-intel-compilers ] ; then
    export CC=icc
    export CXX=icpc
    export FC=ifort
fi

TRAVIS_ROOT="$1"

if [ ! -d "$TRAVIS_ROOT/hpx5" ] ; then
    cd $TRAVIS_ROOT
    if [ "0" = "1" ] ; then
        wget -q --no-check-certificate http://hpx.crest.iu.edu/release/HPX_Release_v2.0.0.tar.gz
        if [ `which shasum` ] ; then
            echo "SHA-256 signature is:"
            shasum -a 256 HPX_Release_v2.0.0.tar.gz
            echo "SHA-256 signature should be:"
            echo "647c5f0ef3618f734066c91d741021d7bd38cf21"
        fi
        tar -xzf HPX_Release_v2.0.0.tar.gz
        cd HPX_Release_v2.0.0/hpx
    else
       export GIT_SSL_NO_VERIFY=1
       git clone --depth 1 http://gitlab.crest.iu.edu/extreme/hpx.git hpx5-source
       cd hpx5-source
    fi
    ./bootstrap
    ./configure --prefix=$TRAVIS_ROOT/hpx5
    make -j2
    make check
    make install
else
    echo "HPX-5 installed..."
    find $TRAVIS_ROOT/hpx5 -name hpx-config
fi
