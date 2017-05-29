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
        for p in octave ; do
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
        echo "Linux not supported"
        exit -1
    ;;
esac
