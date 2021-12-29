#!/bin/sh

set -e
set -x

os=`uname`
CI_ROOT="$1"

git clone --depth 1 https://github.com/llvm-mirror/pstl.git $CI_ROOT/llvm-pstl-git
cd $CI_ROOT/llvm-pstl-git
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CI_ROOT/pstl
make -j2 install

#case "$os" in
#    Darwin)
#        echo "Mac"
#        brew upgrade parallelstl || brew install parallelstl
#        ;;
#    Linux)
#        echo "Linux"
#        if [ ! -d "$CI_ROOT/pstl" ]; then
#            git clone --depth 1 https://github.com/intel/parallelstl.git $CI_ROOT/pstl
#        fi
#        ;;
#esac
