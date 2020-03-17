#!/bin/sh

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

if [ "${CC}" = "gcc" ] || [ "${CXX}" = "g++" ] ; then
    case "$os" in
        Darwin)
            echo "Mac"
            # this is 5.3.0 or later
            brew upgrade gcc || brew install gcc --force-bottle || true
            brew link --overwrite --dry-run gcc
            brew link --overwrite gcc || true
            ;;
        Linux)
            echo "Linux"
            set +e
            for v in "-10" "-9" "-8" "-7" "-6" "-5" ; do
                sudo apt-get install gcc$v g++$v gfortran$v
            done
            set -e
            ;;
    esac
fi
