set -e
set -x

os=`uname`

case "$os" in
    Darwin)
        echo "Mac"
        brew update
        # this is 5.3.0 or later
        brew install gcc --without-multilib
        ;;
    Linux)
        echo "Linux"
        cd $TRAVIS_ROOT
        wget -q ftp://gcc.gnu.org/pub/gcc/releases/gcc-5.3.0/gcc-5.3.0.tar.bz2
        tar -C $TRAVIS_ROOT -xjf gcc-5.3.0.tar.bz2
        ./contrib/download_prerequisites
        mkdir build && cd build
        ../configure --prefix=$TRAVIS_ROOT --enable-threads=posix --with-system-zlib --enable-__cxa_atexit --enable-languages=c,c++ --with-tune=native --enable-lto --disable-multilib
        make -j4 && make install
        ;;
esac

