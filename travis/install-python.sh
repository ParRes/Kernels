#!/bin/sh

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

case "$os" in
    Darwin)
        echo "Mac"
        brew update
        brew upgrade python numpy || brew install python numpy
        ;;
    Linux)
        echo "Linux"
        ;;
esac
