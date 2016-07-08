#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

GASNET_PREFIX=$TRAVIS_ROOT/gasnet-$GASNET_CONDUIT

export GASNET_RELEASE=GASNet-1.26.3

# On Mac (not Linux), we see this error:
#   upcrun: When network=smp compile with '-pthreads' or PSHM support to run with > 1 thread
os=`uname`
case $os in
    Darwin)
        GASNET_NO_PTHREADS="--disable-par" # --enable-pshm # configure CC=cc test fails
        MPI_ROOT=/usr/local
        ;;
    Linux)
        GASNET_NO_PTHREADS=""
        MPI_ROOT=$TRAVIS_ROOT
        ;;
esac

if [ ! -d "$GASNET_PREFIX" ]; then
    wget --no-check-certificate -q https://gasnet.lbl.gov/$GASNET_RELEASE.tar.gz
    tar -xzf $GASNET_RELEASE.tar.gz
    cd $GASNET_RELEASE
    mkdir build && cd build
    # disable IBV just in case Linux has headers for it
    case "$GASNET_CONDUIT" in
        smp)
            ../configure CC=cc --prefix=$GASNET_PREFIX  --disable-aligned-segments $GASNET_NO_PTHREADS \
                               --enable-$GASNET_CONDUIT --disable-auto-conduit-detect
            ;;
        udp)
            ../configure CC=cc --prefix=$GASNET_PREFIX  --disable-aligned-segments $GASNET_NO_PTHREADS \
                               --enable-$GASNET_CONDUIT --disable-auto-conduit-detect
            ;;
        ofi)
            # TODO factor Hydra out of Sandia OpenSHMEM install so it can be used as spawner here
            ../configure CC=cc --prefix=$GASNET_PREFIX  --disable-aligned-segments $GASNET_NO_PTHREADS \
                               --enable-$GASNET_CONDUIT --with-ofihome=$TRAVIS_ROOT/libfabric \
                               --with-ofi-spawner=pmi   --with-pmi=$TRAVIS_ROOT/hydra \
                               --disable-auto-conduit-detect
            ;;
        mpi)
            ../configure CC=cc --prefix=$GASNET_PREFIX  --disable-aligned-segments $GASNET_NO_PTHREADS \
                               --enable-$GASNET_CONDUIT --with-mpi-cc=$MPI_ROOT/bin/mpicc \
                               --disable-auto-conduit-detect
            ;;
        *)
            echo "GASNet conduit not specified - configure CC=cc will guess."
            ../configure CC=cc --prefix=$GASNET_PREFIX --disable-aligned-segments
            ;;
    esac
    make -j2
    make install
else
    echo "GASNet installed..."
    find $GASNET_PREFIX -name "*gasnet*" -type f
fi

