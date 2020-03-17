#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

if [ "${TRAVIS_OS_NAME}" = "osx" ] || [ "${CHPL_COMM}" = "none" ] ; then
    echo "Mac single-locale"
    brew update
    brew install chapel || brew upgrade chapel
    brew test chapel
else
    # We could test Clang via the C back-end as well, but it seems silly.
    # Let GCC exercise C back-end and test the LLVM back-end for Clang.
    if [ "${CC}" = "clang" ] || [ "${CXX}" = "clang++" ] ; then
        CHPL_LLVM=llvm
    fi
    cd $TRAVIS_ROOT
    wget -q --no-check-certificate https://github.com/chapel-lang/chapel/releases/download/1.12.0/chapel-1.12.0.tar.gz
    tar -xzf chapel-1.12.0.tar.gz
    ln -s chapel-1.12.0 chapel
    cd chapel
    make
    ln -s `find $PWD -type f -name chpl` $TRAVIS_HOME/bin/chpl
fi
