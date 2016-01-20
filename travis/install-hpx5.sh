set -e
set -x

TRAVIS_ROOT="$1"

#export USE_HPX_TARBALL=0

if [ ! -d "$TRAVIS_ROOT/hpx5" ]; then
    cd $TRAVIS_ROOT
    #if [ "$USE_HPX_TARBALL" ] ; then
    #    wget -q --no-check-certificate http://hpx.crest.iu.edu/release/HPX_Release_v2.0.0.tar.gz
    #    if [ `which shasum` ] ; then
    #        echo "SHA-256 signature is:"
    #        shasum -a 256 HPX_Release_v2.0.0.tar.gz
    #        echo "SHA-256 signature should be:"
    #        echo "647c5f0ef3618f734066c91d741021d7bd38cf21"
    #    fi
    #    tar -xzf HPX_Release_v2.0.0.tar.gz
    #    cd HPX_Release_v2.0.0/hpx
    #else
        https://gitlab.crest.iu.edu/extreme/hpx.git hpx5-source
        cd hpx5-source
    #fi
    ./bootstrap
    ./configure --prefix=$TRAVIS_ROOT/hpx5
    make -j2
    make check
    make install
else
    echo "HPX-5 installed..."
    find $TRAVIS_ROOT/hpx5 -name hpx-config
fi
