set -e
set -x

TRAVIS_ROOT="$1"

# reasonable options include:
#  smp, udp, mpi, ofi
GASNET_CONDUIT="$2"

export BUPC_RELEASE=berkeley_upc-2.22.0

if [ ! -d "$TRAVIS_ROOT/bupc-$CC" ]; then
    wget --no-check-certificate -q http://upc.lbl.gov/download/release/$BUPC_RELEASE.tar.gz
    tar -xzf $BUPC_RELEASE.tar.gz
    cd $BUPC_RELEASE
    mkdir build && cd build
    # disable IBV just in case Linux has headers for it
    case "$GASNET_CONDUIT" in
        smp)
            ../configure --prefix=$TRAVIS_ROOT/bupc-$CC --disable-ibv --without-mpi-cc \
                         --enable-$GASNET_CONDUIT
            ;;
        udp)
            ../configure --prefix=$TRAVIS_ROOT/bupc-$CC --disable-ibv --without-mpi-cc \
                         --enable-$GASNET_CONDUIT
            ;;
        ofi)
            # TODO factor Hydra out of Sandia OpenSHMEM install so it can be used as spawner here
            sh ./travis/install-libfabric.sh $TRAVIS_ROOT
            ../configure --prefix=$TRAVIS_ROOT/bupc-$CC --disable-ibv --without-mpi-cc \
                         --enable-$GASNET_CONDUIT --with-ofihome=$TRAVIS_ROOT/libfabric \
                         --with-ofi-spawner=ssh
            ;;
        mpi)
            sh ./travis/install-mpi.sh $TRAVIS_ROOT mpich
            ../configure --prefix=$TRAVIS_ROOT/bupc-$CC --disable-ibv \
                         --enable-$GASNET_CONDUIT --with-mpi-cc=$TRAVIS_ROOT/mpich/bin/mpicc
            ;;
        *)
            echo "Invalid choice of GASNet conduit..."
            exit 85
            ;;
    esac
else
    echo "Berkeley UPC (w/ $CC) installed..."
    find $TRAVIS_ROOT/bupc -name upcc -type f
fi

