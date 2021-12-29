#!/bin/sh

set -e
set -x

os=`uname`
CI_ROOT="$1"

case "$os" in
    Darwin)
        echo "Mac"
        brew unlink python@2 || brew uninstall python@2
        brew upgrade python || brew install python
        brew upgrade numpy || brew install numpy
        brew link --overwrite python
        ;;
    Linux)
        echo "Linux"
        sudo apt-get install python3-numpy
        #sudo apt-get install python3-numba
        ;;
esac
