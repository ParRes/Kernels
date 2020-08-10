#!/bin/sh

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

git clone --depth 1 https://github.com/llvm-mirror/pstl.git $TRAVIS_ROOT/llvm-pstl-git
cd $TRAVIS_ROOT/llvm-pstl-git
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$TRAVIS_ROOT/pstl
make -j2 install

#case "$os" in
#    Darwin)
#        echo "Mac"
#        brew upgrade parallelstl || brew install parallelstl
#        ;;
#    Linux)
#        echo "Linux"
#        if [ ! -d "$TRAVIS_ROOT/pstl" ]; then
#            git clone --depth 1 https://github.com/intel/parallelstl.git $TRAVIS_ROOT/pstl
#        fi
#        ;;
#esac
