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
        wget -q ftp://gcc.gnu.org/pub/gcc/releases/gcc-5.3.0/gcc-5.3.0.tar.bz2
        tar -C /tmp -xjf gcc-5.3.0.tar.bz2
        cd /tmp/gcc-5.3.0
        ./contrib/download_prerequisites
        mkdir build && cd build
        ../configure --prefix=$HOME --enable-threads=posix --with-system-zlib --enable-__cxa_atexit --enable-languages=c,c++ --with-tune=native --enable-lto --disable-multilib
        make -j4 && make install
        ;;
esac

