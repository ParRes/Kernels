#!/bin/sh

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

case "$os" in
    Darwin)
        echo "Mac"
        brew update
        brew tap homebrew/core
        brew install rust || brew upgrade rust
        ;;
    Linux)
        echo "Linux not supported"
        ;;
esac
