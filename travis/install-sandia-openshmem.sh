set -e
set -x

TRAVIS_ROOT="$1"
SHMEM_ROOT=$TRAVIS_ROOT/sandia-openshmem

if [ ! -d "$SHMEM_ROOT" ]; then
    # install OFI/libfabric
    cd $TRAVIS_ROOT
    git clone --depth 10 https://github.com/ofiwg/libfabric.git libfabric
    cd libfabric
    ./autogen.sh
    ./configure --prefix=$SHMEM_ROOT
    make && make install
    export FI_LOG_LEVEL=error

    # install Hydra
    cd $TRAVIS_ROOT
    wget http://www.mpich.org/static/downloads/3.2/hydra-3.2.tar.gz
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
    ./configure --with-ofi=$SHMEM_ROOT --with-ofi-libdir=$SHMEM_ROOT/lib --disable-fortran --enable-error-checking --enable-remote-virtual-addressing --enable-pmi-simple --prefix=$SHMEM_ROOT
    make && make install
else
    echo "Sandia OpenSHMEM installed..."
    find $SHMEM_ROOT -name shmem.h
fi
