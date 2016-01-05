set -e
set -x

TRAVIS_ROOT="$1"
HYDRA_ROOT=$TRAVIS_ROOT/hydra

if [ ! -d "$SHMEM_ROOT" ]; then
    cd $TRAVIS_ROOT
    wget --no-check-certificate -q http://www.mpich.org/static/downloads/3.2/hydra-3.2.tar.gz
    tar xvzf hydra-3.2.tar.gz
    cd hydra-3.2
    ./configure --prefix=$HYDRA_ROOT
    make && make install
else
    echo "MPICH Hydra installed..."
    find $HYDRA_ROOT -name pmi.h
fi
