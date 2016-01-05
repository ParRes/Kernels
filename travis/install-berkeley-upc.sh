set -e
set -x

TRAVIS_ROOT="$1"

MPI_IMPL=mpich

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
if [ "`uname`" -eq "Darwin" ]; then
    BUPC_NO_PTHREADS="--disable-par --enable-pshm"
else
    BUPC_NO_PTHREADS=""
fi

if [ ! -d "$BUPC_PREFIX" ]; then
    wget --no-check-certificate -q http://upc.lbl.gov/download/release/$BUPC_RELEASE.tar.gz
    tar -xzf $BUPC_RELEASE.tar.gz
    cd $BUPC_RELEASE
    mkdir build && cd build
    # disable IBV just in case Linux has headers for it
    case "$GASNET_CONDUIT" in
        smp)
            ../configure --prefix=$BUPC_PREFIX --disable-ibv --disable-aligned-segments \
                         --enable-$GASNET_CONDUIT --without-mpi-cc \
                         --disable-udp $BUPC_NO_PTHREADS
            ;;
        udp)
            ../configure --prefix=$BUPC_PREFIX --disable-ibv --disable-aligned-segments \
                         --enable-$GASNET_CONDUIT --without-mpi-cc \
                         --disable-smp $BUPC_NO_PTHREADS
            ;;
        ofi)
            # TODO factor Hydra out of Sandia OpenSHMEM install so it can be used as spawner here
            ../configure --prefix=$BUPC_PREFIX --disable-ibv --without-mpi-cc --disable-aligned-segments \
                         --enable-$GASNET_CONDUIT --with-ofihome=$TRAVIS_ROOT/libfabric --with-ofi-spawner=ssh \
                         --disable-smp --disable-udp $BUPC_NO_PTHREADS
            ;;
        mpi)
            ../configure --prefix=$BUPC_PREFIX --disable-ibv --disable-aligned-segments \
                         --enable-$GASNET_CONDUIT --with-mpi-cc=$TRAVIS_ROOT/$MPI_IMPL/bin/mpicc \
                         --disable-smp --disable-udp $BUPC_NO_PTHREADS
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

