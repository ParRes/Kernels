#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

case "$CC" in
    gcc)
        for gccversion in "-6" "-5" "-5.3" "-5.2" "-5.1" "-4.9" "-4.8" "-4.7" "-4.6" "" ; do
            if [ -f "`which gcc$gccversion`" ]; then
                export PRK_CC="gcc$gccversion"
                export PRK_CXX="g++$gccversion"
                echo "Found GCC: $PRK_CC"
                break
            fi
        done
        ;;
    clang)
        for clangversion in "-3.9" "-3.8" "-3.7" "-3.6" "-3.5" "-3.4" "" ; do
            find /usr/local -name clang$clangversion
            if [ -f "`which clang$clangversion`" ]; then
                export PRK_CC="clang$clangversion"
                export PRK_CXX="clang++$clangversion"
                echo "Found Clang: $PRK_CC"
                break
            fi
        done
        ;;
    icc)
        export PRK_CC=icc
        export PRK_CXX=icpc
        ;;
esac

if [ "${TRAVIS_OS_NAME}" = "osx" ] && [ "${CHPL_COMM}" = "none" ] ; then
    echo "Mac single-locale"
    brew update
    brew install chapel || brew upgrade chapel
    # Chapel PRK depend upon recent features
    #brew install chapel --HEAD
    brew test chapel
else
    # We could test Clang via the C back-end as well, but it seems silly.
    # Let GCC exercise C back-end and test the LLVM back-end for Clang.
    if [ "${CC}" = "clang" ] || [ "${CXX}" = "clang++" ] ; then
        export CHPL_LLVM=llvm
    fi
    cd $TRAVIS_ROOT
    #wget -q --no-check-certificate https://github.com/chapel-lang/chapel/releases/download/1.12.0/chapel-1.12.0.tar.gz
    #tar -xzf chapel-1.12.0.tar.gz
    #ln -s chapel-1.12.0 chapel
    git clone --depth 10 https://github.com/chapel-lang/chapel.git
    cd chapel
    make CC=$PRK_CC CXX=$PRK_CXX -j2
    mkdir -p $TRAVIS_ROOT/bin
    ln -s `find $PWD -type f -name chpl` $TRAVIS_ROOT/bin/chpl
fi
