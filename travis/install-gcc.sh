#!/bin/sh

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

if [ "${CC}" = "gcc" ] || [ "${CXX}" = "g++" ] ; then
    case "$os" in
        Darwin)
            echo "Mac"
            brew update || true
            # this is 5.3.0 or later
            brew upgrade gcc || brew install gcc --force-bottle || true
            ;;
        DisableLinux)
            echo "Linux"
            if [ ! -d "$TRAVIS_ROOT/gcc" ]; then
                cd $TRAVIS_ROOT
                wget -q ftp://gcc.gnu.org/pub/gcc/releases/gcc-5.3.0/gcc-5.3.0.tar.bz2
                tar -xjf gcc-5.3.0.tar.bz2
                cd gcc-5.3.0
                ./contrib/download_prerequisites
                mkdir build && cd build
                ../configure --prefix=$TRAVIS_ROOT/gcc \
                             --enable-threads=posix --with-system-zlib --enable-__cxa_atexit \
                             --enable-languages=c,c++ --with-tune=native \
                             --enable-lto --disable-multilib
                make -j4
                make install
            else
                echo "GCC installed..."
                find $TRAVIS_ROOT -name gcc -type f
                gcc --version
            fi
        ;;
    esac
fi
