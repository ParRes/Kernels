#!/bin/sh
# This configuration file was taken originally from the mpi4py project
# <http://mpi4py.scipy.org/>, and then modified for Julia

set -e
set -x

os=`uname`

case "$os" in
    Darwin)
        echo "Mac"
        # This is for Mac OSX 10.10 - Travis currently uses 10.9.5
        #wget -q http://www.gccupc.org/gupc-5201-1/28-gupc-5201-x8664-mac-os-1010-yosemiti/file
        #tar -xzvf upc-5.2.0.1-x86_64-apple-macosx10.10.tar.gz
        wget -q http://www.gccupc.org/gupc-5201-1/32-gupc-5-2-0-1-source-release/file
        mv file upc-5.2.0.1.src.tar.bz2
        tar -xjf upc-5.2.0.1.src.tar.bz2
        cd upc-5.2.0.1
        mkdir build && cd build
        ../configure --disable-multilib --enable-languages=c,c++ --prefix=$HOME
        make && make install
        ;;

    Linux)
        echo "Linux"
        wget -q http://www.gccupc.org/gupc-5201-1/30-gupc-5201-x8664-ubuntu-1204/file
        mv file upc-5.2.0.1-x86_64-linux-ubuntu12.4.tar.gz
        sudo tar -C / -xzvf upc-5.2.0.1-x86_64-linux-ubuntu12.4.tar.gz
        ;;
esac
