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
        #CHPL_LLVM=llvm
        # Attempt to use built-in LLVM, since compiling it in Travis takes a while.
        CHPL_LLVM=system
    else
        CHPL_LLVM=none
    fi
    cd $TRAVIS_ROOT
    CHAPEL_RELEASE=1.15.0
    wget -q --no-check-certificate https://github.com/chapel-lang/chapel/releases/download/${CHAPEL_RELEASE}/chapel-${CHAPEL_RELEASE}.tar.gz
    tar -xzf chapel-${CHAPEL_RELEASE}.tar.gz
    ln -s chapel-${CHAPEL_RELEASE} chapel
    cd chapel
    make
    ln -s `find $PWD -type f -name chpl` $TRAVIS_HOME/bin/chpl
fi
