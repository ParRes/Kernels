set -e
set -x

TRAVIS_ROOT="$1"
SHMEM_ROOT=$TRAVIS_ROOT/sandia-openshmem

if [ ! -d "$SHMEM_ROOT" ]; then
    # master
    #git clone --depth 10 https://github.com/regrant/sandia-shmem.git sandia-shmem
    #git clone --depth 10 https://github.com/Sandia-OpenSHMEM/SOS.git sandia-shmem
    #cd sandia-shmem
    # 1.3 release
    #wget https://github.com/Sandia-OpenSHMEM/SOS/archive/v1.3.0-beta2.tar.gz
    #tar -xzf v1.3.0-beta2.tar.gz
    #cd SOS-1.3.0-beta2
    # 1.2 release
    wget https://github.com/Sandia-OpenSHMEM/SOS/archive/v1.2.0.tar.gz
    tar -xzf v1.2.0.tar.gz
    cd SOS-1.2.0
    ./autogen.sh
    # must build in-place (https://github.com/regrant/sandia-shmem/issues/49)
    ./configure --with-ofi=$TRAVIS_ROOT/libfabric --with-ofi-libdir=$TRAVIS_ROOT/libfabric/lib \
                --disable-fortran \
                --enable-error-checking \
                --enable-remote-virtual-addressing \
                --enable-pmi-simple --with-pmi=$TRAVIS_ROOT/hydra \
                --prefix=$SHMEM_ROOT
    make && make install
else
    echo "Sandia OpenSHMEM installed..."
    find $SHMEM_ROOT -name shmem.h
fi
