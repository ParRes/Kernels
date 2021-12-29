#!/bin/sh

set -e
set -x

os=`uname`
CI_ROOT="$1"

case "$os" in
    Darwin)
        echo "Mac"
        brew cask upgrade julia || brew cask install julia
        ;;
    Linux)
        echo "Linux"
        JULIA_NAME=julia-1.3.1
        if [ ! -d "$CI_ROOT/$JULIA_NAME" ]; then
            cd $CI_ROOT
            wget --no-check-certificate -q https://julialang-s3.julialang.org/bin/linux/x64/1.3/julia-1.3.1-linux-x86_64.tar.gz
            tar -C $CI_ROOT -xzvf julia-1.3.1-linux-x86_64.tar.gz
            # symbolic link was not working for reasons i cannot explain
            ln -s $CI_ROOT/$JULIA_NAME $CI_ROOT/julia
            find $CI_ROOT -type f -name julia
        fi
        ;;
esac
