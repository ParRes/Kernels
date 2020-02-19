#!/bin/sh

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

case "$os" in
    Darwin)
        echo "Mac"
        brew update
        brew upgrade parallelstl || brew install parallelstl
        ;;
    Linux)
        echo "Linux"
        if [ ! -d "$TRAVIS_ROOT/pstl" ]; then
            git clone --depth 1 https://github.com/intel/parallelstl.git $TRAVIS_ROOT/pstl
        fi
        ;;
esac
