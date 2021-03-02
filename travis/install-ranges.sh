#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

if [ "${TRAVIS_OS_NAME}" = "osx" ] ; then
    git clone --depth 1 https://github.com/ericniebler/range-v3.git $TRAVIS_ROOT/range-v3
else
    sh ./travis/install-boost.sh $TRAVIS_ROOT
fi
