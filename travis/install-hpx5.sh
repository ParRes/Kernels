set -e
set -x

TRAVIS_ROOT="$1"
MPI_IMPL="$2"

if [ ! -d "$TRAVIS_ROOT/hpx5" ]; then
    cd $TRAVIS_ROOT
    wget -q --no-check-certificate http://hpx.crest.iu.edu/release/HPX_Release_v2.0.0.tar.gz
    echo "SHA-256 signature is:"
    shasum -a 256 HPX_Release_v2.0.0.tar.gz
    echo "SHA-256 signature should be:"
    echo "647c5f0ef3618f734066c91d741021d7bd38cf21"
    tar -xzf HPX_Release_v2.0.0.tar.gz
    cd HPX_Release_v2.0.0/hpx
    ./bootstrap
    # This does not include MPI...
    ./configure --prefix=$TRAVIS_ROOT/hpx5
    make -j2
    make check
    make install
else
    echo "HPX-5 installed..."
    find $TRAVIS_ROOT/hpx5 -name hpx-config
fi
