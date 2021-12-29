#!/bin/sh

set -e
set -x

os=`uname`
CI_ROOT="$1"

case "$os" in
    Darwin)
        echo "Mac"
        brew tap homebrew/core
        brew install rust || brew upgrade rust
        ;;
    Linux)
        echo "Linux not supported"
        ;;
esac
