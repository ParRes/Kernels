set -e
set -x

TRAVIS_ROOT="$1"
SHMEM_ROOT=$TRAVIS_ROOT/sandia-openshmem

if [ ! -d "$SHMEM_ROOT" ]; then
    #git clone --depth 10 https://github.com/regrant/sandia-shmem.git sandia-shmem
    git clone --depth 10 https://github.com/Sandia-OpenSHMEM/SOS.git sandia-shmem
    cd sandia-shmem
    ./autogen.sh
    # must build in-place (https://github.com/regrant/sandia-shmem/issues/49)
    ./configure --with-ofi=$TRAVIS_ROOT/libfabric --with-ofi-libdir=$TRAVIS_ROOT/libfabric/lib \
                --disable-fortran --enable-error-checking \
                --enable-remote-virtual-addressing \
                --enable-pmi-simple --with-pmi=$TRAVIS_ROOT/hydra \
                --prefix=$SHMEM_ROOT
    make && make install
else
    echo "Sandia OpenSHMEM installed..."
    find $SHMEM_ROOT -name shmem.h
fi
