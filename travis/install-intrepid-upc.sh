#!/bin/sh
# This configuration file was taken originally from the mpi4py project
# <http://mpi4py.scipy.org/>, and then modified for Julia

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

case "$CC" in
    gcc)
        if [ ! -d "$TRAVIS_ROOT/gupc" ]; then
            case "$os" in
                Darwin)
                    echo "Mac"
                    # Travis uses Mac OSX 10.9, so this might not work...
                    mkdir $TRAVIS_ROOT/gupc
                    wget --no-check-certificate -q http://www.gccupc.org/gupc-5201-1/28-gupc-5201-x8664-mac-os-1010-yosemiti/file -O upc-5.2.0.1-x86_64-apple-macosx10.10.tar.gz
                    tar -C $TRAVIS_ROOT/gupc -xzvf upc-5.2.0.1-x86_64-apple-macosx10.10.tar.gz
                    find $TRAVIS_ROOT/gupc -name gupc -type f
                    ;;
                Linux)
                    echo "Linux"
                    mkdir $TRAVIS_ROOT/gupc
                    wget --no-check-certificate -q http://www.gccupc.org/gupc-5201-1/30-gupc-5201-x8664-ubuntu-1204/file -O upc-5.2.0.1-x86_64-linux-ubuntu12.4.tar.gz
                    tar -C $TRAVIS_ROOT/gupc -xzvf upc-5.2.0.1-x86_64-linux-ubuntu12.4.tar.gz
                    find $TRAVIS_ROOT/gupc -name gupc -type f
                    ;;
            esac
            # Building from source overflows Travis CI 4 MB output...
            #wget --no-check-certificate -q http://www.gccupc.org/gupc-5201-1/32-gupc-5-2-0-1-source-release/file -O upc-5.2.0.1.src.tar.bz2
            #tar -xjf upc-5.2.0.1.src.tar.bz2
            #cd upc-5.2.0.1
            #./contrib/download_prerequisites
            #mkdir build && cd build
            #../configure --disable-multilib --enable-languages=c,c++ --prefix=$TRAVIS_ROOT/gupc
            ## Travis has problems with how much output the GCC build creates
            #make -j4 &> /dev/null
            #make install
        else
            echo "GCC UPC installed..."
            find $TRAVIS_ROOT/gupc -name gupc -type f
        fi
        ;;

    clang)
        echo "Clang UPC not supported yet..."
        exit 60
        if [ ! -d "$TRAVIS_ROOT/clupc" ]; then
            # get source files
            mkdir build && cd build
            ../configure --disable-multilib --enable-languages=c,c++ --prefix=$TRAVIS_ROOT/clupc
            make -j4
            make install
        else
            echo "GCC UPC installed..."
            find $TRAVIS_ROOT/clupc -name clang
            clang --version
        fi
        ;;
    *)
        echo "This should not happen..."
        exit 70
        ;;
esac
