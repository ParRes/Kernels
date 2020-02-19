#!/bin/sh

set -e
set -x

if [ -f ~/use-intel-compilers ] ; then
    export CC=icc
    export CXX=icpc
    export FC=ifort
fi

os=`uname`
TRAVIS_ROOT="$1"
# charm++ or AMPI
RUNTIME="$2"

# not here: pull this out of Travis environment
#CHARM_CONDUIT="$3"

# unused for now
case "$TRAVIS_OS_NAME" in
    linux)
        CHARM_OS=linux
        ;;
    osx)
        CHARM_OS=darwin
        ;;
esac

# unused for now
case "$CHARM_CONDUIT" in
    multicore)
        CHARM_CONDUIT_OPTIONS="multicore-linux64"
        ;;
    netlrts)
        CHARM_CONDUIT_OPTIONS="netlrts-$CHARM_OS-x86_64"
        ;;
    netlrts-smp)
        CHARM_CONDUIT_OPTIONS="netlrts-$CHARM_OS-x86_64 smp"
        ;;
esac

if [ ! -d "$TRAVIS_ROOT/charm" ]; then
    cd $TRAVIS_ROOT
    git clone --depth 1 -b v6.8.0 https://charm.cs.illinois.edu/gerrit/charm.git charm
    cd charm
    case "$os" in
        Darwin)
            echo "Mac"
            #./build $RUNTIME netlrts-darwin-x86_64 --with-production -j4
            ./build $RUNTIME netlrts-darwin-x86_64 smp --with-production -j4
            ;;

        Linux)
            echo "Linux"
            # This fails with: The authenticity of host 'localhost (127.0.0.1)' can't be established.
            #./build $RUNTIME netlrts-linux-x86_64 --with-production -j4
            ./build $RUNTIME netlrts-linux-x86_64 smp --with-production
            #./build $RUNTIME multicore-linux64 --with-production
            ;;
    esac
else
    echo "Charm++ or AMPI already installed..."
    case "$RUNTIME" in
        AMPI)
            find $TRAVIS_ROOT/charm -name charmrun
            find $TRAVIS_ROOT/charm -name ampicc
            ;;
        charm++)
            find $TRAVIS_ROOT/charm -name charmrun
            find $TRAVIS_ROOT/charm -name charmc
            ;;
    esac
fi
