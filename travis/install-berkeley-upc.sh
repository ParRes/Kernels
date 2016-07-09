#!/bin/sh

set -e
set -x

if [ -f ~/use-intel-compilers ] ; then
    export CC=icc
    export CXX=icpc
    export FC=ifort
fi

TRAVIS_ROOT="$1"

# we can't handle this yet in build-run-prk.sh
#if [ "x$GASNET_CONDUIT" -eq "x" ] ; then
#    BUPC_PREFIX=bupc-$CC
#else
#    BUPC_PREFIX=bupc-$CC-$GASNET_CONDUIT
#fi
BUPC_PREFIX=$TRAVIS_ROOT/bupc-$CC

export BUPC_RELEASE=berkeley_upc-2.22.0

# On Mac (not Linux), we see this error:
#   upcrun: When network=smp compile with '-pthreads' or PSHM support to run with > 1 thread
os=`uname`
case $os in
    Darwin)
        BUPC_NO_PTHREADS="--disable-par" # --enable-pshm # configure test fails
        MPI_ROOT=/usr/local
        ;;
    Linux)
        BUPC_NO_PTHREADS=""
        MPI_ROOT=$TRAVIS_ROOT
        ;;
esac

if [ ! -d "$BUPC_PREFIX" ]; then
    wget --no-check-certificate -q http://upc.lbl.gov/download/release/$BUPC_RELEASE.tar.gz
    tar -xzf $BUPC_RELEASE.tar.gz
    cd $BUPC_RELEASE
    mkdir build && cd build
    # disable IBV just in case Linux has headers for it
    case "$GASNET_CONDUIT" in
        smp)
            ../configure --prefix=$BUPC_PREFIX --disable-aligned-segments $BUPC_NO_PTHREADS \
                         --enable-$GASNET_CONDUIT --disable-auto-conduit-detect
            ;;
        udp)
            ../configure --prefix=$BUPC_PREFIX --disable-aligned-segments $BUPC_NO_PTHREADS \
                         --enable-$GASNET_CONDUIT --disable-auto-conduit-detect
            ;;
        ofi)
            # TODO factor Hydra out of Sandia OpenSHMEM install so it can be used as spawner here
            ../configure --prefix=$BUPC_PREFIX --disable-aligned-segments $BUPC_NO_PTHREADS \
                         --enable-$GASNET_CONDUIT --with-ofihome=$TRAVIS_ROOT/libfabric \
                         --with-ofi-spawner=pmi --with-pmi=$TRAVIS_ROOT/hydra \
                         --disable-auto-conduit-detect
            ;;
        mpi)
            ../configure --prefix=$BUPC_PREFIX --disable-aligned-segments $BUPC_NO_PTHREADS \
                         --enable-$GASNET_CONDUIT --with-mpi-cc=$MPI_ROOT/bin/mpicc \
                         --disable-auto-conduit-detect
            ;;
        *)
            echo "GASNet conduit not specified - configure will guess."
            ../configure --prefix=$BUPC_PREFIX --disable-aligned-segments
            ;;
    esac
    make -j2
    make install
else
    echo "Berkeley UPC (w/ $CC) installed..."
    find $BUPC_PREFIX -name upcc -type f
fi

