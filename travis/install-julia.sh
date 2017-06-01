#!/bin/sh

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

case "$os" in
    Darwin)
        echo "Mac"
        set +e
        brew update
        for p in julia Caskroom/cask/julia ; do
            if [ "x`brew ls --versions $p`" = "x" ] ; then
                echo "$p is not installed - installing it"
                brew install $p
            else
                echo "$p is installed - upgrading it"
                brew upgrade $p
            fi
        done
        set -e
    ;;

    Linux)
        echo "Linux"
        #JULIA_NAME=julia-2ac304dfba    # julia-0.4.5-linux-x86_64.tar.gz
        JULIA_NAME=julia-f4c6c9d4bb     # julia-0.5.2-linux-x86_64.tar.gz
        if [ ! -d "$TRAVIS_ROOT/$JULIA_NAME" ]; then
            cd $TRAVIS_ROOT
            #wget --no-check-certificate -q https://julialang.s3.amazonaws.com/bin/linux/x64/0.4/julia-0.4.5-linux-x86_64.tar.gz
            #tar -C $TRAVIS_ROOT -xzvf julia-0.4.5-linux-x86_64.tar.gz
            wget --no-check-certificate -q https://julialang-s3.julialang.org/bin/linux/x64/0.5/julia-0.5.2-linux-x86_64.tar.gz
            tar -C $TRAVIS_ROOT -xzvf julia-0.5.2-linux-x86_64.tar.gz
            # symbolic link was not working for reasons i cannot explain
            ln -s $TRAVIS_ROOT/$JULIA_NAME $TRAVIS_ROOT/julia
            find $TRAVIS_ROOT -type f -name julia
        fi
        ;;
esac
