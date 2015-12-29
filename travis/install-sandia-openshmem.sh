set -e
set -x

TRAVIS_ROOT="$1"
SHMEM_ROOT=$TRAVIS_ROOT/sandia-openshmem

if [ ! -d "$SHMEM_ROOT" ]; then
    # install Hydra
    cd $TRAVIS_ROOT
    wget --no-check-certificate -q http://www.mpich.org/static/downloads/3.2/hydra-3.2.tar.gz
    tar xvzf hydra-3.2.tar.gz
    cd hydra-3.2
    ./configure --prefix=$SHMEM_ROOT
    make && make install

    # install Sandia OpenSHMEM
    cd $TRAVIS_ROOT
    git clone --depth 10 https://github.com/regrant/sandia-shmem.git
    cd sandia-shmem
    ./autogen.sh
    # must build in-place (https://github.com/regrant/sandia-shmem/issues/49)
    ./configure --with-ofi=$TRAVIS_ROOT/libfabric --with-ofi-libdir=$TRAVIS_ROOT/libfabric/lib \
                --disable-fortran --enable-error-checking \
                --enable-remote-virtual-addressing --enable-pmi-simple \
                --prefix=$SHMEM_ROOT
    make && make install
else
    echo "Sandia OpenSHMEM installed..."
    find $SHMEM_ROOT -name shmem.h
fi
