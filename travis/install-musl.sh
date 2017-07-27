#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

VERSION=musl-1.1.16
FILE=https://www.musl-libc.org/releases/${VERSION}.tar.gz

if [ "${TRAVIS_OS_NAME}" = "linux" ] ; then
    echo "Linux"
    wget --no-check-certificate -q ${FILE}
    tar -xzf ${FILE}
    cd ${VERSION}
    ./configure --prefix=${TRAVIS_ROOT}/musl
    make
    make install
else
    echo "Unsupported operating system"
fi
