#!/bin/sh

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

case "$os" in
    Darwin)
        echo "Mac"
        brew tap homebrew/science
        brew install octave || brew upgrade octave
    ;;

    Linux)
        echo "Linux not supported"
    ;;
esac
