#!/bin/sh

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

case "$os" in
    Darwin)
        echo "Mac"
        brew update
        brew install boost || brew update boost || true
        ;;

    Linux)
        echo "Linux"
        ;;
esac
