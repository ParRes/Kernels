#!/bin/sh

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

case "$os" in
    Darwin)
        echo "Mac"
        brew update
        brew cask upgrade julia || brew cask install julia
        ;;
    Linux)
        echo "Linux"
        JULIA_NAME=julia-1.3.1
        if [ ! -d "$TRAVIS_ROOT/$JULIA_NAME" ]; then
            cd $TRAVIS_ROOT
            wget --no-check-certificate -q https://julialang-s3.julialang.org/bin/linux/x64/1.3/julia-1.3.1-linux-x86_64.tar.gz
            tar -C $TRAVIS_ROOT -xzvf julia-1.3.1-linux-x86_64.tar.gz
            # symbolic link was not working for reasons i cannot explain
            ln -s $TRAVIS_ROOT/$JULIA_NAME $TRAVIS_ROOT/julia
            find $TRAVIS_ROOT -type f -name julia
        fi
        ;;
esac
