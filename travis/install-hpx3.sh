set -e
set -x

TRAVIS_ROOT="$1"

if [ ! -d "$TRAVIS_ROOT/hpx3" ]; then
    cd $TRAVIS_ROOT
    wget -q --no-check-certificate http://stellar.cct.lsu.edu/files/hpx_0.9.11.tar.bz2
    if [ `which md5` ] ; then
        echo "MD5 signature is:"
        md5 hpx_0.9.11.tar.bz2
        echo "MD5 signature should be:"
        echo "86a71189fb6344d27bf53d6aa2b33122"
    fi
    tar -xjf hpx_0.9.11.tar.bz2
    cd hpx_0.9.11
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$TRAVIS_ROOT/hpx3
    make -j2
    make check
    make install
else
    echo "HPX-3 installed..."
    find $TRAVIS_ROOT/hpx3
fi
