#!/bin/sh

set -e
set -x

os=`uname`
CI_ROOT="$1"

case "$os" in
    Darwin)
        echo "Mac"
        brew upgrade --cask julia || brew install --cask julia
        ;;
    Linux)
        echo "Linux"
        # JULIA/p2p.jl uses the (@main)(args) entry-point convention added
        # in Julia 1.11; the old 1.3.1 pin predates that by 5 years.
        JULIA_NAME=julia-1.11.7
        if [ ! -d "$CI_ROOT/$JULIA_NAME" ]; then
            cd $CI_ROOT
            wget --no-check-certificate -q https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.7-linux-x86_64.tar.gz
            tar -C $CI_ROOT -xzf julia-1.11.7-linux-x86_64.tar.gz
            # symbolic link was not working for reasons i cannot explain
            ln -s $CI_ROOT/$JULIA_NAME $CI_ROOT/julia
            find $CI_ROOT -type f -name julia
        fi
        ;;
esac
